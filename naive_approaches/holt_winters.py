"""
Holt–Winters / ETS(A,*,A) baselines for TPGNN CSV datasets.

- Loads V_*.csv shaped (T, N) (time x nodes), e.g., PeMS V_228.csv.
- Fits one ExponentialSmoothing (Holt–Winters) model PER NODE on TRAIN+VAL.
- Seasonality: weekly by default (2016 for 5-min PeMS) — ETS(A,N,A) or ETS(A,A,A).
- Evaluation:
  * single-origin (default): forecast whole TEST recursively from the TRAIN+VAL end
    (NOTE: optimistic for multi-step; prefer sliding-origin for fair horizon scoring);
  * sliding (optional): re-fit every `sliding_stride` samples and aggregate errors.

Outputs MAE / RMSE / MAPE per horizon (e.g., 3/6/9/12) and averages.

Run examples:
  python naive_approaches/holt_winters.py --data_path naive_approaches/V_228.csv --variant ANA --horizons 3 6 9 12
  python naive_approaches/holt_winters.py --data_path naive_approaches/V_228.csv --variant AAA --sliding_stride 72
"""

import argparse
import warnings
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # Holt–Winters (ETS(A,*,A))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

VERBOSE = True  # toggle console noise from per-node fits

# ---------- Metrics ----------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps=1e-6):
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)  # percent

# ---------- Data ----------
def load_tpgnn_matrix(data_path: str) -> np.ndarray:
    """
    Returns array shaped (T, N). If file is (N, T), it will be transposed.
    Assumes CSV has no header (as in PeMS V_228.csv).
    """
    df = pd.read_csv(data_path, header=None, low_memory=False)
    X = df.values.astype(np.float32)
    if X.shape[0] < X.shape[1]:
        X = X.T
    # Basic NaN handling
    X = pd.DataFrame(X).ffill().bfill().values
    return X  # (T, N)

# ---------- ETS core ----------
def fit_ets_series(y: np.ndarray, seasonal_periods: int, variant: str):
    """
    Fit a single univariate series with Holt–Winters / ETS(A,*,A).
    variant: 'ANA' (ETS(A,N,A)) or 'AAA' (ETS(A,A,A))
    Returns fitted statsmodels results object, or None if it fails.
    """
    trend = None if variant.upper() == "ANA" else "add"
    try:
        y = y.astype(np.float64, copy=False)
        model = ExponentialSmoothing(
            y,
            trend=trend,              # None for ANA, 'add' for AAA
            seasonal="add",           # ETS(A,*,A): additive seasonality
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        res = model.fit(optimized=True, use_boxcox=None, remove_bias=True)
        return res
    except Exception:
        return None

def forecast_single_origin(Y_trval: np.ndarray, n_test: int, seasonal_periods: int, variant: str) -> np.ndarray:
    """
    Fit once per node on TRAIN+VAL and forecast TEST of length n_test (recursive).
    Returns array (n_test, N).
    """
    N = Y_trval.shape[1]
    preds = np.zeros((n_test, N), dtype=np.float32)

    if VERBOSE:
        print(f"  Fitting ETS models for {N} nodes...")

    # Process nodes in batches for better progress tracking
    batch_size = max(1, min(50, N // 10))  # up to 50 nodes
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        if VERBOSE:
            print(f"    Processing nodes {batch_start+1}-{batch_end} of {N}")
        for j in range(batch_start, batch_end):
            y = Y_trval[:, j]
            try:
                res = fit_ets_series(y, seasonal_periods, variant)
                if res is None:
                    # Fallback: seasonal naive (safe & fast)
                    last_season = y[-seasonal_periods:]
                    reps = int(np.ceil(n_test / seasonal_periods))
                    preds[:, j] = np.tile(last_season, reps)[:n_test]
                else:
                    f = res.forecast(n_test)
                    preds[:, j] = np.asarray(f, dtype=np.float32)
            except Exception:
                last_season = y[-seasonal_periods:]
                reps = int(np.ceil(n_test / seasonal_periods))
                preds[:, j] = np.tile(last_season, reps)[:n_test]

    if VERBOSE:
        print(f"  Completed ETS forecasting for all {N} nodes")
    return preds

def forecast_sliding(Y: np.ndarray, eval_start: int, eval_end: int,
                     n_pred: int, seasonal_periods: int, variant: str,
                     sliding_stride: int) -> Dict[int, List[np.ndarray]]:
    """
    Sliding (periodic re-fit) forecasts:
    - At origins o = eval_start, eval_start+stride, ..., <= eval_end-1
    - Fit per node on Y[:o], forecast n_pred steps, collect horizon-h predictions for time o+h.

    Returns dict h -> list of arrays (N,), each array is forecast for horizon h at a given origin.
    """
    T, N = Y.shape
    out = {h: [] for h in range(1, n_pred + 1)}
    origins = list(range(eval_start, max(eval_end, eval_start), max(1, sliding_stride)))
    for o in origins:
        # ensure we never look ahead
        Y_hist = Y[:o, :]  # up to origin (exclusive)
        n_fore = min(n_pred, max(0, (T - 1) - o))  # last index is T-1; forecast to T-1
        if n_fore <= 0:
            continue
        preds = forecast_single_origin(Y_hist, n_fore, seasonal_periods, variant)  # (n_fore, N)
        for h in range(1, n_fore + 1):
            out[h].append(preds[h - 1, :])  # forecast for time t = o + h
    return out  # dict of lists

# ---------- Evaluation ----------
def evaluate_single_origin(Y: np.ndarray, day_slot: int, splits: Tuple[int, int, int],
                           n_pred: int, seasonal_periods: int, variant: str,
                           horizons: List[int]) -> Dict[int, Dict[str, float]]:
    """
    Fit on TRAIN+VAL and forecast whole TEST. Compute metrics for each horizon h
    using the forecast at offset h-1 from the single origin.
    NOTE: Each horizon is scored on ONE instance; prefer evaluate_sliding() for fair horizon scoring.
    """
    d_train, d_val, d_test = splits
    n_train, n_val, n_test = d_train * day_slot, d_val * day_slot, d_test * day_slot
    eval_start = n_train + n_val

    if eval_start < seasonal_periods:
        raise ValueError(
            f"Need at least one full season before evaluation: eval_start={eval_start} < seasonal_periods={seasonal_periods}"
        )

    Y_trval = Y[:eval_start, :]
    Y_test = Y[eval_start:eval_start + n_test, :]

    preds = forecast_single_origin(Y_trval, n_test, seasonal_periods, variant)  # (n_test, N)

    results = {}
    for h in horizons:
        if h - 1 >= n_test:
            continue
        y_true = Y_test[h - 1, :]       # one timestamp across all nodes
        y_pred = preds[h - 1, :]
        results[h] = {
            "MAE": mae(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "MAPE": mape(y_true, y_pred),
        }
    if results:
        avg = {k: float(np.mean([results[h][k] for h in results])) for k in ["MAE", "RMSE", "MAPE"]}
        results["AVG"] = avg
    return results

def evaluate_sliding(Y: np.ndarray, day_slot: int, splits: Tuple[int, int, int],
                     n_pred: int, seasonal_periods: int, variant: str,
                     horizons: List[int], sliding_stride: int) -> Dict[int, Dict[str, float]]:
    """
    Periodic re-fit. Aggregate metrics across many origins (and nodes) for each horizon h.
    This matches rolling-origin evaluation recommended in the literature.
    """
    d_train, d_val, d_test = splits
    n_train, n_val, n_test = d_train * day_slot, d_val * day_slot, d_test * day_slot
    eval_start = n_train + n_val
    eval_end   = n_train + n_val + n_test - n_pred  # last origin so that o+h stays in test

    if eval_start < seasonal_periods:
        raise ValueError(
            f"Need at least one full season before evaluation: eval_start={eval_start} < seasonal_periods={seasonal_periods}"
        )
    if eval_end <= eval_start:
        raise ValueError("Not enough test samples for the requested n_pred and stride.")

    # Collect forecasts per horizon across origins
    fc_dict = forecast_sliding(
        Y, eval_start=eval_start, eval_end=eval_end,
        n_pred=n_pred, seasonal_periods=seasonal_periods,
        variant=variant, sliding_stride=sliding_stride
    )

    # Compute metrics per horizon
    results = {}
    origins = list(range(eval_start, eval_end, max(1, sliding_stride)))
    for h in horizons:
        preds_list = fc_dict.get(h, [])
        if not preds_list:
            continue
        y_pred_all = []
        y_true_all = []
        for k, origin in enumerate(origins):
            if k >= len(preds_list):
                break
            t = origin + h
            if t >= Y.shape[0]:
                break
            y_pred_all.append(preds_list[k])          # (N,)
            y_true_all.append(Y[t, :])                # (N,)
        if not y_pred_all:
            continue
        y_pred_all = np.vstack(y_pred_all)
        y_true_all = np.vstack(y_true_all)
        results[h] = {
            "MAE": mae(y_true_all, y_pred_all),
            "RMSE": rmse(y_true_all, y_pred_all),
            "MAPE": mape(y_true_all, y_pred_all),
        }
    if results:
        avg = {k: float(np.mean([results[h][k] for h in results])) for k in ["MAE", "RMSE", "MAPE"]}
        results["AVG"] = avg
    return results

def print_table(title: str, results: Dict[int, Dict[str, float]], horizons: List[int]):
    print("\n" + title)
    print("-" * 64)
    header = ["H", "MAE", "RMSE", "MAPE"]
    print("{:>4s} {:>10s} {:>10s} {:>10s}".format(*header))
    for h in horizons:
        if h in results:
            r = results[h]
            print(f"{h:>4d} {r['MAE']:10.4f} {r['RMSE']:10.4f} {r['MAPE']:10.2f}")
    if "AVG" in results:
        r = results["AVG"]
        print("-" * 64)
        print(f"AVG  {r['MAE']:10.4f} {r['RMSE']:10.4f} {r['MAPE']:10.2f}")
    print()

# ---------- Evaluation wrapper ----------
def evaluate_ets(
    data_path: str,
    day_slot: int = 288,                 # 5-min data: 288 per day
    days_train_val_test=(34, 5, 5),      # mirrors common PeMSD7 splits
    n_pred: int = 12,                    # 12 steps (~60 min for 5-min data)
    season: str = "weekly",              # weekly or daily seasonality
    variant: str = "ANA",                # ETS variant: ANA or AAA
    horizons_to_report=None,
    sliding_stride: int = 0,             # 0 = single-origin, >0 = sliding window
    fast_mode: bool = False              # If True, use simplified approach for speed
):
    """
    Wrapper function to evaluate ETS/Holt-Winters forecasting.
    """
    if horizons_to_report is None:
        horizons_to_report = [3, 6, 9, 12]

    # Seasonal periods
    season = season.lower()
    if season == "weekly":
        seasonal_periods = 7 * day_slot       # 2016 for PeMS 5-min (7×288)
    elif season == "daily":
        seasonal_periods = day_slot           # 288
    else:
        seasonal_periods = int(season)

    # Load data
    Y = load_tpgnn_matrix(data_path)

    # Evaluate
    if fast_mode:
        results = evaluate_fast_ets(
            Y, day_slot=day_slot, splits=days_train_val_test,
            n_pred=n_pred, seasonal_periods=seasonal_periods,
            variant=variant, horizons=horizons_to_report
        )
    elif sliding_stride and sliding_stride > 0:
        results = evaluate_sliding(
            Y, day_slot=day_slot, splits=days_train_val_test,
            n_pred=n_pred, seasonal_periods=seasonal_periods,
            variant=variant, horizons=horizons_to_report,
            sliding_stride=sliding_stride
        )
    else:
        print("[note] Single-origin ETS scoring can be optimistic for multi-step horizons — "
              "consider --sliding_stride for fairer evaluation.")
        results = evaluate_single_origin(
            Y, day_slot=day_slot, splits=days_train_val_test,
            n_pred=n_pred, seasonal_periods=seasonal_periods,
            variant=variant, horizons=horizons_to_report
        )

    # Filter results to only requested horizons, but keep AVG for reporting
    filtered_results = {}
    for h in horizons_to_report:
        if h in results:
            filtered_results[h] = results[h]
    if "AVG" in results:
        filtered_results["AVG"] = results["AVG"]
    return filtered_results


def evaluate_fast_ets(Y: np.ndarray, day_slot: int, splits: Tuple[int, int, int],
                      n_pred: int, seasonal_periods: int, variant: str,
                      horizons: List[int]) -> Dict[int, Dict[str, float]]:
    """
    Fast ETS evaluation using simplified approach:
    seasonal naive (+ optional linear trend for AAA).
    """
    d_train, d_val, d_test = splits
    n_train, n_val, n_test = d_train * day_slot, d_val * day_slot, d_test * day_slot
    eval_start = n_train + n_val
    if eval_start < seasonal_periods:
        raise ValueError(
            f"Need at least one full season before evaluation: eval_start={eval_start} < seasonal_periods={seasonal_periods}"
        )
    Y_trval = Y[:eval_start, :]
    Y_test = Y[eval_start:eval_start + n_test, :]

    if VERBOSE:
        print(f"  Fast ETS mode: seasonal naive (+trend for AAA) for {Y.shape[1]} nodes")

    preds = np.zeros((n_test, Y.shape[1]), dtype=np.float32)
    for j in range(Y.shape[1]):
        y = Y_trval[:, j]
        last_season = y[-seasonal_periods:]
        reps = int(np.ceil(n_test / seasonal_periods))
        seasonal_comp = np.tile(last_season, reps)[:n_test]

        if variant.upper() == "AAA":
            # Linear trend from recent 2 seasons if available
            trend_window = min(seasonal_periods * 2, len(y))
            if trend_window > seasonal_periods:
                recent = y[-trend_window:]
                x = np.arange(len(recent))
                slope = np.polyfit(x, recent, 1)[0]
                trend_comp = slope * np.arange(n_test)
                preds[:, j] = seasonal_comp + trend_comp
            else:
                preds[:, j] = seasonal_comp
        else:
            preds[:, j] = seasonal_comp

    results = {}
    for h in horizons:
        if h - 1 >= n_test:
            continue
        y_true = Y_test[h - 1, :]
        y_pred = preds[h - 1, :]
        results[h] = {
            "MAE": mae(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "MAPE": mape(y_true, y_pred),
        }
    if results:
        avg = {k: float(np.mean([results[h][k] for h in results])) for k in ["MAE", "RMSE", "MAPE"]}
        results["AVG"] = avg
    return results


# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="V_228.csv")
    p.add_argument("--day_slot", type=int, default=288, help="samples per day; PeMS 5-min ⇒ 288")
    p.add_argument("--splits", type=int, nargs=3, default=[34, 5, 5], metavar=("TRAIN_D", "VAL_D", "TEST_D"))
    p.add_argument("--n_pred", type=int, default=12, help="max forecast horizon (steps)")
    p.add_argument("--variant", type=str, default="ANA", choices=["ANA", "AAA"],
                   help="ETS(A,N,A) = ANA (no trend) or ETS(A,A,A) = AAA (additive trend)")
    p.add_argument("--season", type=str, default="weekly", help="weekly|daily|<int> seasonal periods")
    p.add_argument("--horizons", type=int, nargs="*", default=[3, 6, 9, 12])
    p.add_argument("--sliding_stride", type=int, default=0,
                   help="0 = single-origin; >0 = re-fit every <stride> samples (e.g., 72 ≈ 6h)")
    args = p.parse_args()

    # Seasonal periods
    season = args.season.lower()
    if season == "weekly":
        seasonal_periods = 7 * args.day_slot       # 2016 for PeMS 5-min
    elif season == "daily":
        seasonal_periods = args.day_slot           # 288 for PeMS 5-min
    else:
        seasonal_periods = int(season)

    # Load data
    Y = load_tpgnn_matrix(args.data_path)

    # Evaluate
    if args.sliding_stride > 0:
        results = evaluate_sliding(
            Y, day_slot=args.day_slot, splits=tuple(args.splits),
            n_pred=args.n_pred, seasonal_periods=seasonal_periods,
            variant=args.variant, horizons=args.horizons,
            sliding_stride=args.sliding_stride
        )
        title = f"ETS({args.variant}) sliding (stride={args.sliding_stride}, season={seasonal_periods}) on {args.data_path}"
    else:
        results = evaluate_single_origin(
            Y, day_slot=args.day_slot, splits=tuple(args.splits),
            n_pred=args.n_pred, seasonal_periods=seasonal_periods,
            variant=args.variant, horizons=args.horizons
        )
        title = f"ETS({args.variant}) single-origin (season={seasonal_periods}) on {args.data_path}"

    print_table(title, results, args.horizons)

if __name__ == "__main__":
    main()
