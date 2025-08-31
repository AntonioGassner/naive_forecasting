"""
hour_of_week.py — Hour-of-Week Historical Average baselines for TPGNN CSV datasets.

- Loads V_*.csv shaped (T, N) (time x nodes), e.g., PeMS V_228.csv.
- Builds a weekly "slot-of-week" profile per node from TRAIN+VAL only.
- Forecasts multi-step horizons by profile lookup (optional last-week residual correction).
- Reports MAE / RMSE / MAPE per horizon and averaged.

Run examples:
    python naive_approaches/hour_of_week.py --data_path naive_approaches/V_228.csv --season weekly
    # with small residual correction (lambda=0.2):
    python naive_approaches/hour_of_week.py --data_path naive_approaches/V_228.csv --season weekly --lambda_resid 0.2
"""

import argparse
import os
import numpy as np
import pandas as pd

# ---------- Metrics (NaN-safe) ----------
def _pairwise_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (~np.isnan(y_true)) & (~np.isnan(y_pred))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = _pairwise_mask(y_true, y_pred)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = _pairwise_mask(y_true, y_pred)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    mask = _pairwise_mask(y_true, y_pred)
    if not mask.any():
        return float("nan")
    denom = np.clip(np.abs(y_true[mask]), eps, None)
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom)) * 100.0)

def corr_flat(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = _pairwise_mask(y_true, y_pred)
    if not mask.any():
        return float("nan")
    a = y_true[mask].reshape(-1)
    b = y_pred[mask].reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

# ---------- Data ----------
def load_tpgnn_matrix(data_path: str) -> np.ndarray:
    """
    Returns array shaped (T, N). If file is (N, T), it will be transposed.
    Assumes CSV has no header (as in PeMS V_228.csv).

    Note: No global forward/backward filling here to avoid leakage. NaNs are preserved
    and handled downstream by NaN-aware aggregation and metrics.
    """
    df = pd.read_csv(data_path, header=None, low_memory=False)
    X = df.values.astype(np.float32)
    if X.shape[0] < X.shape[1]:
        X = X.T
    return X  # (T, N)

# ---------- HA core ----------
def build_week_profile(Y, weekly_slots, hist_end, start_offset=0):
    _, num_nodes = Y.shape
    Y_hist = Y[:hist_end]

    slot_index = (np.arange(hist_end) + int(start_offset)) % int(weekly_slots)

    sums   = np.zeros((weekly_slots, num_nodes), dtype=np.float64)
    counts = np.zeros((weekly_slots, num_nodes), dtype=np.int64)

    # 1) valid mask first (Y_hist untouched)
    valid_mask = ~np.isnan(Y_hist)

    # 2) masked accumulation without mutating Y_hist
    Y_hist_filled = np.where(valid_mask, Y_hist, 0.0)
    np.add.at(sums,   slot_index, Y_hist_filled)
    np.add.at(counts, slot_index, valid_mask.astype(np.int64))

    with np.errstate(invalid="ignore", divide="ignore"):
        profile = sums / np.maximum(counts, 1)

    # 3) robust fallback for empty slots; handle all-NaN columns
    node_means = np.nanmean(Y_hist, axis=0, keepdims=True)  # (1, N)
    node_means = np.where(np.isnan(node_means), 0.0, node_means)
    empty_mask = counts == 0
    if np.any(empty_mask):
        profile[empty_mask] = np.broadcast_to(node_means, profile.shape)[empty_mask]

    return profile.astype(np.float32)


def ha_forecast(
    Y: np.ndarray,
    weekly_slots: int,
    n_pred: int,
    eval_start: int,
    eval_end: int,
    profile: np.ndarray,
    lambda_resid: float = 0.0,
    start_offset: int = 0,
):
    """
    Multi-step HA forecast with optional last-week residual correction.

    profile: (weekly_slots, N) built from TRAIN+VAL only.
    lambda_resid: forecast = profile[k] + lambda * (Y[t+h-m] - profile[(t+h-m+offset)%m])
    """
    T, N = Y.shape
    m = weekly_slots
    t_idx = np.arange(eval_start, eval_end)[:, None]              # (T_eval, 1)
    h_idx = np.arange(1, n_pred + 1)[None, :]                     # (1, H)

    # Truth
    y_true = Y[t_idx + h_idx, :]                                  # (T_eval, H, N)

    # Profile lookup for each (t, h) by slot-of-week
    k_now = ((t_idx + h_idx + int(start_offset)) % m).astype(int) # (T_eval, H)
    y_hat = profile[k_now, :]                                     # (T_eval, H, N)

    if lambda_resid != 0.0:
        # Residual from the same slot last week (available before time t)
        prev_idx = t_idx + h_idx - m                               # (T_eval, H)
        k_prev = ((prev_idx + int(start_offset)) % m).astype(int)
        r_prev = Y[prev_idx, :] - profile[k_prev, :]               # (T_eval, H, N)
        # NaN-safe residual: treat missing residuals as zero contribution
        r_prev = np.nan_to_num(r_prev, nan=0.0)
        y_hat = y_hat + lambda_resid * r_prev

    return y_true, y_hat

# ---------- Evaluation wrapper ----------
def evaluate_ha(
    data_path: str,
    day_slot: int = 288,                 # 5-min data: 288 per day
    days_train_val_test=(34, 5, 5),      # mirrors common PeMSD7 splits
    n_pred: int = 12,                    # 12 steps (~60 min for 5-min data)
    season: str = "weekly",              # HA usually uses weekly slots
    lambda_resid: float = 0.0,           # 0 = pure HA; try 0.1~0.3 for mild correction
    horizons_to_report=None,
    compute_corr: bool = False,
    start_offset: int = 0,
):
    if horizons_to_report is None:
        horizons_to_report = [3, 6, 9, 12]

    season = season.lower()
    if season == "weekly":
        weekly_slots = 7 * day_slot
    elif season == "daily":
        weekly_slots = day_slot
    else:
        try:
            weekly_slots = int(season)  # custom slot count
            if weekly_slots <= 0:
                raise ValueError
        except Exception:
            raise ValueError("season must be 'weekly', 'daily', or a positive integer of slots")

    # Load data
    Y = load_tpgnn_matrix(data_path)
    T, N = Y.shape

    # Splits (in samples)
    d_train, d_val, d_test = days_train_val_test
    n_train = int(d_train) * day_slot
    n_val   = int(d_val)   * day_slot
    requested_test = int(d_test) * day_slot

    # Adjust test length to fit data if needed
    remaining = T - (n_train + n_val)
    if remaining <= 0:
        raise ValueError("Not enough data after train+val for any testing.")
    n_test = min(requested_test, remaining)

    eval_start = n_train + n_val
    # exclusive end; ensures for all t in [eval_start, eval_end), t+h <= last test index
    eval_end   = n_train + n_val + n_test - n_pred

    if eval_start < weekly_slots:
        raise ValueError(f"Need at least one full season before eval_start: {eval_start} < {weekly_slots}")
    if eval_end <= eval_start:
        raise ValueError("Not enough test samples for the requested n_pred.")

    # Build profile from TRAIN+VAL only
    profile = build_week_profile(
        Y, weekly_slots=weekly_slots, hist_end=eval_start, start_offset=start_offset
    )

    # Metrics per horizon (streaming; avoid large (T_eval, H, N) arrays)
    results = {}
    t_range = np.arange(eval_start, eval_end)
    for h in range(1, n_pred + 1):
        # Truth for horizon h
        yt = Y[t_range + h, :]

        # Profile prediction for corresponding slots
        k_now = (t_range + h + int(start_offset)) % weekly_slots
        yp = profile[k_now, :]

        if lambda_resid != 0.0:
            prev_idx = t_range + h - weekly_slots
            k_prev = (prev_idx + int(start_offset)) % weekly_slots
            r_prev = Y[prev_idx, :] - profile[k_prev, :]
            r_prev = np.nan_to_num(r_prev, nan=0.0)
            yp = yp + lambda_resid * r_prev

        res = {
            "MAE":  mae(yt, yp),
            "RMSE": rmse(yt, yp),
            "MAPE": mape(yt, yp),
        }
        if compute_corr:
            res["CORR"] = corr_flat(yt, yp)
        results[h] = res

    # Pretty print
    title = f"HA (weekly_slots={weekly_slots}, lambda_resid={lambda_resid}, start_offset={start_offset}) on {data_path}"
    print(f"\n{title}")
    print(f"Sampling day_slot={day_slot}, splits(days)={days_train_val_test}, n_pred={n_pred}")
    print(f"Eval window: t in [{eval_start}, {eval_end})  (eval steps: {eval_end - eval_start})")
    print("-" * 68)
    cols = ["H", "MAE", "RMSE", "MAPE"] + (["CORR"] if compute_corr else [])
    fmt = "{:>3s}" + "  {:>10s}" * (len(cols) - 1)
    print(fmt.format(*cols))

    def row(h):
        vals = results[h]
        base = f"{h:3d}  {vals['MAE']:10.4f}  {vals['RMSE']:10.4f}  {vals['MAPE']:10.2f}"
        if compute_corr:
            base += f"  {vals['CORR']:10.4f}"
        return base

    for h in horizons_to_report:
        if 1 <= h <= n_pred:
            print(row(h))

    avg = {k: float(np.nanmean([results[h][k] for h in range(1, n_pred + 1)]))
           for k in results[1].keys()}
    base = f"AVG  {avg['MAE']:10.4f}  {avg['RMSE']:10.4f}  {avg['MAPE']:10.2f}"
    if compute_corr:
        base += f"  {avg['CORR']:10.4f}"
    print("-" * 68)
    print(base)
    print()
    
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    default_csv = os.path.join(os.path.dirname(__file__), "V_228.csv")
    p.add_argument("--data_path", type=str, default=default_csv)
    p.add_argument("--day_slot", type=int, default=288, help="samples per day (PeMS 5-min → 288)")
    p.add_argument("--splits", type=int, nargs=3, default=[34, 5, 5], metavar=("TRAIN_D", "VAL_D", "TEST_D"))
    p.add_argument("--n_pred", type=int, default=12)
    p.add_argument("--season", type=str, default="weekly", help="weekly | daily | <int slots>")
    p.add_argument("--lambda_resid", type=float, default=0.0, help="0.0 = pure HA; try 0.1~0.3 for mild correction")
    p.add_argument("--report", type=int, nargs="*", default=[3, 6, 9, 12])
    p.add_argument("--corr", action="store_true")
    p.add_argument("--start_offset", type=int, default=0, help="phase offset in slots for week alignment")
    args = p.parse_args()

    evaluate_ha(
        data_path=args.data_path,
        day_slot=args.day_slot,
        days_train_val_test=tuple(args.splits),
        n_pred=args.n_pred,
        season=args.season,
        lambda_resid=args.lambda_resid,
        horizons_to_report=args.report,
        compute_corr=args.corr,
        start_offset=args.start_offset,
    )
