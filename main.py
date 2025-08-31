import os
import json
import csv
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

# TOML parsing: use stdlib tomllib on Python 3.11+, fallback to tomli
try:
	import tomllib  # type: ignore[attr-defined]
	toml_load = tomllib.load
except Exception:  # pragma: no cover
	import tomli  # type: ignore
	toml_load = tomli.load

from naive_approaches.seasonal_naive import evaluate_snaive
from naive_approaches.hour_of_week import evaluate_ha
from naive_approaches.holt_winters import evaluate_ets, print_table


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.toml")


def _iso_now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: str) -> None:
	d = os.path.dirname(os.path.abspath(path))
	if d and not os.path.exists(d):
		os.makedirs(d, exist_ok=True)


def _load_config(path: str) -> Dict[str, Any]:
	with open(path, "rb") as f:
		return toml_load(f)


def _get_general(cfg: Dict[str, Any]) -> Dict[str, Any]:
	general = cfg.get("general", {})
	return {
		"data_path": general.get("data_path", "naive_approaches/V_228.csv"),
		"day_slot": int(general.get("day_slot", 288)),
		"splits": tuple(general.get("splits", [34, 5, 5])),
		"n_pred": int(general.get("n_pred", 12)),
		"horizons": list(general.get("horizons", [3, 6, 9, 12])),
		"season": str(general.get("season", "weekly")),
		"output_json": cfg.get("general", {}).get("output_json", cfg.get("general", {}).get("output_json", "results/naive_approaches_results.json")),
		"output_csv": cfg.get("general", {}).get("output_csv", cfg.get("general", {}).get("output_csv", "results/naive_approaches_wide.csv")),
	}


def _collect_results_entries(
	approach: str,
	variant: str,
	metrics_by_h: Dict[int, Dict[str, float]],
	horizons: List[int],
	base_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
	entries: List[Dict[str, Any]] = []
	for h in horizons:
		if h in metrics_by_h:
			m = metrics_by_h[h]
			entries.append({
				"approach": approach,
				"variant": variant,
				"horizon": int(h),
				"MAE": float(m.get("MAE", float("nan"))),
				"RMSE": float(m.get("RMSE", float("nan"))),
				"MAPE": float(m.get("MAPE", float("nan"))),
				"timestamp": _iso_now(),
				"params": base_params,
			})
	return entries


def _write_outputs(entries: List[Dict[str, Any]], json_path: str, csv_path: str) -> None:
	# JSON (programmatic)
	_ensure_dir(json_path)
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(entries, f, indent=2)

	# CSV (wide, human-readable)
	_ensure_dir(csv_path)
	csv_cols = ["approach", "variant", "horizon", "MAE", "MAPE", "RMSE"]  # note MAPE before RMSE
	with open(csv_path, "w", newline="", encoding="utf-8") as f:
		w = csv.DictWriter(f, fieldnames=csv_cols)
		w.writeheader()
		for e in entries:
			w.writerow({
				"approach": e["approach"],
				"variant": e["variant"],
				"horizon": e["horizon"],
				"MAE": e["MAE"],
				"MAPE": e["MAPE"],
				"RMSE": e["RMSE"],
			})


def main() -> None:
	cfg = _load_config(CONFIG_PATH)
	general = _get_general(cfg)
	toggles = cfg.get("toggles", {})

	entries: List[Dict[str, Any]] = []

	data_path: str = general["data_path"]
	day_slot: int = general["day_slot"]
	splits: Tuple[int, int, int] = general["splits"]
	n_pred: int = general["n_pred"]
	horizons: List[int] = general["horizons"]
	season_general: str = general["season"]

	# Seasonal Naive
	if toggles.get("run_seasonal_naive", True):
		sec = {**cfg.get("seasonal_naive", {})}
		season = str(sec.get("season", season_general))
		res = evaluate_snaive(
			data_path=data_path,
			day_slot=day_slot,
			days_train_val_test=splits,
			n_pred=n_pred,
			season=season,
			horizons_to_report=horizons,
			compute_corr=False,
		)
		variant = f"{season}"
		base_params = {
			"season": season,
			"day_slot": day_slot,
			"n_pred": n_pred,
		}
		entries.extend(_collect_results_entries(
			approach="Seasonal_Naive",
			variant=variant,
			metrics_by_h=res,
			horizons=horizons,
			base_params=base_params,
		))

	# Hour of Week (Historical Average)
	if toggles.get("run_hour_of_week", True):
		sec = {**cfg.get("hour_of_week", {})}
		season = str(sec.get("season", season_general))
		lambda_resid = float(sec.get("lambda_resid", 0.0))
		start_offset = int(sec.get("start_offset", 0))
		compute_corr = bool(sec.get("compute_corr", False))
		res = evaluate_ha(
			data_path=data_path,
			day_slot=day_slot,
			days_train_val_test=splits,
			n_pred=n_pred,
			season=season,
			lambda_resid=lambda_resid,
			horizons_to_report=horizons,
			compute_corr=compute_corr,
			start_offset=start_offset,
		)
		variant = f"{season}_lambda_{lambda_resid}"
		base_params = {
			"season": season,
			"day_slot": day_slot,
			"n_pred": n_pred,
			"lambda_resid": lambda_resid,
		}
		entries.extend(_collect_results_entries(
			approach="Hour_of_Week",
			variant=variant,
			metrics_by_h=res,
			horizons=horizons,
			base_params=base_params,
		))

	# Holt-Winters / ETS
	if toggles.get("run_holt_winters", True):
		sec = {**cfg.get("holt_winters", {})}
		season = str(sec.get("season", season_general))
		variant_hw = str(sec.get("variant", "ANA"))
		sliding_stride = int(sec.get("sliding_stride", 0))
		fast_mode = bool(sec.get("fast_mode", False))
		res = evaluate_ets(
			data_path=data_path,
			day_slot=day_slot,
			days_train_val_test=splits,
			n_pred=n_pred,
			season=season,
			variant=variant_hw,
			horizons_to_report=horizons,
			sliding_stride=sliding_stride,
			fast_mode=fast_mode,
		)
		suffix = "sliding" if sliding_stride and sliding_stride > 0 else "single"
		# Pretty-print ETS results for parity with other approaches
		title = f"ETS({variant_hw}) {suffix} on {data_path}"
		print_table(title, res, horizons)
		variant = f"{variant_hw}_{suffix}"
		base_params = {
			"season": season,
			"day_slot": day_slot,
			"n_pred": n_pred,
			"variant": variant_hw,
			"sliding_stride": sliding_stride if sliding_stride != 0 else None,
			"fast_mode": fast_mode,
		}
		entries.extend(_collect_results_entries(
			approach="Holt_Winters",
			variant=variant,
			metrics_by_h=res,
			horizons=horizons,
			base_params=base_params,
		))

	# Persist
	_write_outputs(entries, general["output_json"], general["output_csv"])
	print(f"Wrote {len(entries)} rows to {general['output_json']} and {general['output_csv']}")


if __name__ == "__main__":
	main()
