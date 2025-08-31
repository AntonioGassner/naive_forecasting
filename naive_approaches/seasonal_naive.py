"""
snaive_tpgnn.py — Seasonal-Naïve baselines for TPGNN CSV datasets.

- Loads V_*.csv (time x nodes) used by TPGNN (e.g., PeMS V_228.csv).
- Uses TPGNN-style splits in days (default: 34/5/5) and 5-min sampling (day_slot=288).
- Computes S-Naïve forecasts with a chosen seasonal period m (daily=288, weekly=2016).
- Reports MAE, RMSE, MAPE (and optional CORR) per horizon and averaged.

Run:
    python snaive_tpgnn.py --data_path data/PeMS/V_228.csv --season daily
    # or weekly seasonality:
    python snaive_tpgnn.py --data_path data/PeMS/V_228.csv --season weekly
"""

import argparse
import numpy as np
import pandas as pd

# ---------- Metrics ----------
def _mask(y_true, y_pred):
    return (~np.isnan(y_true)) & (~np.isnan(y_pred))

def mae(y_true, y_pred):
    m = _mask(y_true, y_pred)
    return float(np.mean(np.abs(y_true[m] - y_pred[m]))) if m.any() else float("nan")

def rmse(y_true, y_pred):
    m = _mask(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m])**2))) if m.any() else float("nan")

def mape(y_true, y_pred, eps=1e-6):
    m = _mask(y_true, y_pred)
    if not m.any(): return float("nan")
    denom = np.clip(np.abs(y_true[m]), eps, None)
    return float(np.mean(np.abs((y_true[m] - y_pred[m]) / denom)) * 100.0)

def corr_flat(y_true, y_pred):
    m = _mask(y_true, y_pred)
    if not m.any(): return float("nan")
    a, b = y_true[m].ravel(), y_pred[m].ravel()
    if a.size == 0 or b.size == 0 or a.std() == 0 or b.std() == 0: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

# ---------- Data loading ----------
def load_tpgnn_matrix(data_path: str) -> np.ndarray:
    df = pd.read_csv(data_path, header=None, low_memory=False)
    X = df.values.astype(np.float32)
    if X.shape[0] < X.shape[1]:
        X = X.T
    return X

# ---------- Seasonal-Naïve ----------
def snaive_forecast(Y: np.ndarray, m: int, n_pred: int, eval_start: int, eval_end: int):
    """
    Y: (T, N), seasonal period m, predict n_pred steps ahead for each t in [eval_start, eval_end)
    Returns:
        y_true: (T_eval, n_pred, N)
        y_pred: (T_eval, n_pred, N)
    """
    T, N = Y.shape
    assert eval_start >= m, "Evaluation window must start after at least one full season."
    assert eval_end + n_pred <= T, "Evaluation end must allow n_pred steps."

    t_idx = np.arange(eval_start, eval_end)[:, None]                   # (T_eval, 1)
    h_idx = np.arange(1, n_pred + 1)[None, :]                          # (1, H)
    y_true = Y[t_idx + h_idx, :]                                       # (T_eval, H, N)
    y_pred = Y[t_idx + h_idx - m, :]                                   # (T_eval, H, N)
    return y_true, y_pred

# ---------- Evaluation wrapper ----------
def evaluate_snaive(
    data_path: str,
    day_slot: int = 288,                 # TPGNN default for 5-min data
    days_train_val_test=(34, 5, 5),      # TPGNN defaults in config.py
    n_pred: int = 12,                    # TPGNN default (12×5min = 1 hour)
    season: str = "daily",               # "daily" (m=288) or "weekly" (m=7*288)
    horizons_to_report=None,             # e.g., [3,6,9,12] like TPGNN mode=1
    compute_corr: bool = False
):
    if horizons_to_report is None:
        horizons_to_report = [3, 6, 9, 12]

    # pick seasonal period
    season = season.lower()
    if season == "daily":
        m = day_slot
    elif season == "weekly":
        m = 7 * day_slot
    else:
        try:
            m = int(season)  # allow custom integer period
        except ValueError:
            raise ValueError("season must be 'daily', 'weekly', or an integer period")

    # load data (T, N)
    Y = load_tpgnn_matrix(data_path)
    T, N = Y.shape

        # derive split indices with length check
    d_train, d_val, d_test = days_train_val_test
    n_train = int(d_train) * day_slot
    n_val   = int(d_val)   * day_slot
    requested_test = int(d_test) * day_slot

    remaining = T - (n_train + n_val)
    if remaining <= 0:
        raise ValueError("Not enough data after train+val for any testing.")
    n_test = min(requested_test, remaining)

    # evaluation over the entire test block
    eval_start = n_train + n_val
    eval_end   = n_train + n_val + n_test - n_pred  # last t s.t. t+h in test
    if eval_start < m:
        raise ValueError(f"Test start ({eval_start}) must be >= seasonal period m ({m}).")
    if eval_end <= eval_start:
        raise ValueError("Not enough test samples for the requested n_pred.")

    # Streaming evaluation: no (T_eval, H, N) allocations
    t_range = np.arange(eval_start, eval_end)
    results = {}
    for h in range(1, n_pred + 1):
        yt = Y[t_range + h, :]
        yp = Y[t_range + h - m, :]
        res = {
            "MAE":  mae(yt, yp),
            "RMSE": rmse(yt, yp),
            "MAPE": mape(yt, yp),
        }
        if compute_corr:
            res["CORR"] = corr_flat(yt, yp)
        results[h] = res

    # pretty print selected horizons + average
    print(f"\nS-Naïve (m={m}) on {data_path}")
    print(f"Sampling day_slot={day_slot}, splits(days)={days_train_val_test}, n_pred={n_pred}")
    print(f"Eval window: t in [{eval_start}, {eval_end})  (total eval steps: {eval_end - eval_start})")
    print("-" * 64)
    header = ["H", "MAE", "RMSE", "MAPE"] + (["CORR"] if compute_corr else [])
    print(("{:>3s}" + "  {:>10s}" * (len(header) - 1)).format(*header))


    def row(h):
        vals = results[h]
        base = f"{h:3d}  {vals['MAE']:10.4f}  {vals['RMSE']:10.4f}  {vals['MAPE']:10.2f}"
        if compute_corr:
            base += f"  {vals['CORR']:10.4f}"
        return base

    for h in horizons_to_report:
        if 1 <= h <= n_pred:
            print(row(h))

    # average over all horizons
    avg = {
        k: float(np.mean([results[h][k] for h in range(1, n_pred + 1)]))
        for k in results[1].keys()
    }
    base = f"AVG  {avg['MAE']:10.4f}  {avg['RMSE']:10.4f}  {avg['MAPE']:10.2f}"
    if compute_corr:
        base += f"  {avg['CORR']:10.4f}"
    print("-" * 64)
    print(base)
    print()
    
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="V_228.csv")
    p.add_argument("--day_slot", type=int, default=288)
    p.add_argument("--splits", type=int, nargs=3, default=[34, 5, 5], metavar=("TRAIN_D", "VAL_D", "TEST_D"))
    p.add_argument("--n_pred", type=int, default=12)
    p.add_argument("--season", type=str, default="daily", help="daily | weekly | <int>")
    p.add_argument("--report", type=int, nargs="*", default=[3, 6, 9, 12])
    p.add_argument("--corr", action="store_true")
    args = p.parse_args()

    evaluate_snaive(
        data_path=args.data_path,
        day_slot=args.day_slot,
        days_train_val_test=tuple(args.splits),
        n_pred=args.n_pred,
        season=args.season,
        horizons_to_report=args.report,
        compute_corr=args.corr,
    )
