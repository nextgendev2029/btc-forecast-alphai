"""
Hyperparameter tuning for BTC prediction model.

Sweeps over range_scale, vol_window, and tail_df to find the
parameter combination with coverage closest to 0.95 and lowest Winkler score.
"""

import itertools
import time

import pandas as pd
from tqdm import tqdm

from src.data import load_btc_prices
from src.model import predict_next_range
from src.backtest import winkler_score, evaluate


# ── Parameter grid ──────────────────────────────────────────────────────────
RANGE_SCALES = [0.95, 1.0, 1.03, 1.05, 1.08, 1.1]
VOL_WINDOWS  = [30, 50, 80]
TAIL_DFS     = [4, 5, 7]

# ── Fixed backtest settings ─────────────────────────────────────────────────
TRAIN_WINDOW = 500
TEST_WINDOW  = 300
N_SIMS       = 2000
DRIFT_WINDOW = 200          # keep constant (not being tuned)


def run_single_backtest(
    prices: pd.Series,
    range_scale: float,
    vol_window: int,
    tail_df: int,
) -> dict:
    """Run a backtest for one parameter combination and return metrics."""
    start_i = len(prices) - TEST_WINDOW - 1
    end_i   = len(prices) - 1

    results = []

    for i in range(start_i, end_i):
        history      = prices.iloc[: i + 1]
        actual_next  = float(prices.iloc[i + 1])
        recent       = history.tail(TRAIN_WINDOW)

        low_95, high_95 = predict_next_range(
            recent,
            n_sims=N_SIMS,
            vol_window=vol_window,
            drift_window=DRIFT_WINDOW,
            tail_df=tail_df,
            range_scale=range_scale,
            seed=i,
        )

        low_95  = float(low_95)
        high_95 = float(high_95)
        width   = high_95 - low_95
        covered = int(low_95 <= actual_next <= high_95)
        winkler = winkler_score(actual_next, low_95, high_95)

        results.append({
            "width_95":   width,
            "covered_95": covered,
            "winkler_95": winkler,
        })

    df = pd.DataFrame(results)
    return {
        "coverage_95":      float(df["covered_95"].mean()),
        "average_width_95": float(df["width_95"].mean()),
        "mean_winkler_95":  float(df["winkler_95"].mean()),
        "n_predictions":    len(df),
    }


def main() -> None:
    print("=" * 70)
    print("  BTC Prediction Model — Hyperparameter Sweep")
    print("=" * 70)

    # ── Load data once ──────────────────────────────────────────────────
    print("\n📡 Fetching BTCUSDT hourly data …")
    prices = load_btc_prices(limit=1500)
    print(f"   Loaded {len(prices)} bars  "
          f"({prices.index[0]}  →  {prices.index[-1]})\n")

    # ── Build parameter grid ────────────────────────────────────────────
    grid = list(itertools.product(RANGE_SCALES, VOL_WINDOWS, TAIL_DFS))
    total = len(grid)
    print(f"🔬 Running {total} experiments  "
          f"(test_window={TEST_WINDOW}, n_sims={N_SIMS})\n")

    all_results = []

    for idx, (rs, vw, tdf) in enumerate(tqdm(grid, desc="Tuning"), 1):
        t0 = time.time()
        metrics = run_single_backtest(prices, rs, vw, tdf)
        elapsed = time.time() - t0

        row = {
            "range_scale":      rs,
            "vol_window":       vw,
            "tail_df":          tdf,
            "coverage_95":      metrics["coverage_95"],
            "average_width_95": metrics["average_width_95"],
            "mean_winkler_95":  metrics["mean_winkler_95"],
            "elapsed_s":        round(elapsed, 2),
        }
        all_results.append(row)

    # ── Results table ───────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("mean_winkler_95").reset_index(drop=True)

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENT RESULTS  (sorted by Winkler ↑)")
    print("=" * 70)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(results_df.to_string(index=False))

    # ── Save full results ───────────────────────────────────────────────
    results_df.to_csv("outputs/tuning_results.csv", index=False)
    print("\n💾 Full results saved → outputs/tuning_results.csv")

    # ── Select best ─────────────────────────────────────────────────────
    # 1. coverage distance from 0.95
    results_df["cov_distance"] = (results_df["coverage_95"] - 0.95).abs()
    min_cov_dist = results_df["cov_distance"].min()

    # 2. among those closest to 0.95, pick lowest Winkler
    candidates = results_df[results_df["cov_distance"] == min_cov_dist]
    best = candidates.sort_values("mean_winkler_95").iloc[0]

    print("\n" + "=" * 70)
    print("  🏆 BEST CONFIGURATION")
    print("=" * 70)
    print(f"  range_scale      : {best['range_scale']}")
    print(f"  vol_window       : {int(best['vol_window'])}")
    print(f"  tail_df          : {int(best['tail_df'])}")
    print("-" * 40)
    print(f"  coverage_95      : {best['coverage_95']:.4f}")
    print(f"  average_width_95 : {best['average_width_95']:.2f}")
    print(f"  mean_winkler_95  : {best['mean_winkler_95']:.2f}")
    print(f"  cov_distance     : {best['cov_distance']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
