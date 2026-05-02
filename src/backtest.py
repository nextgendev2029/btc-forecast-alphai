import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import load_btc_prices
from src.model import predict_next_range


def winkler_score(actual: float, low: float, high: float, alpha: float = 0.05) -> float:
    width = high - low

    if actual < low:
        return width + (2 / alpha) * (low - actual)

    if actual > high:
        return width + (2 / alpha) * (actual - high)

    return width


def run_backtest(
    prices: pd.Series,
    train_window: int = 500,
    test_window: int = 720,
    n_sims: int = 1000,
) -> pd.DataFrame:
    """
    No-peeking backtest.

    For every prediction:
    - use only prices up to current bar
    - predict next bar range
    - compare with actual next close
    """
    if len(prices) < train_window + test_window + 1:
        raise ValueError(
            f"Need at least {train_window + test_window + 1} bars, got {len(prices)}"
        )

    results = []

    start_i = len(prices) - test_window - 1
    end_i = len(prices) - 1

    for i in tqdm(range(start_i, end_i), desc="Backtesting"):
        history = prices.iloc[: i + 1]
        actual_next = float(prices.iloc[i + 1])

        recent_history = history.tail(train_window)

        low_95, high_95 = predict_next_range(
            recent_history,
            n_sims=n_sims,
            vol_window=80,
            drift_window=200,
            tail_df=4,
            range_scale=1.05,
            seed=i,
        )

        low_95 = float(low_95)
        high_95 = float(high_95)
        width_95 = high_95 - low_95
        covered = int(low_95 <= actual_next <= high_95)
        winkler = winkler_score(actual_next, low_95, high_95)

        results.append(
            {
                "prediction_time": history.index[-1].isoformat(),
                "target_time": prices.index[i + 1].isoformat(),
                "current_price": float(history.iloc[-1]),
                "actual_next_price": actual_next,
                "low_95": low_95,
                "high_95": high_95,
                "width_95": width_95,
                "covered_95": covered,
                "winkler_95": winkler,
            }
        )

    return pd.DataFrame(results)


def evaluate(results_df: pd.DataFrame) -> dict:
    return {
        "coverage_95": float(results_df["covered_95"].mean()),
        "average_width_95": float(results_df["width_95"].mean()),
        "mean_winkler_95": float(results_df["winkler_95"].mean()),
        "n_predictions": int(len(results_df)),
    }


def save_jsonl(results_df: pd.DataFrame, output_path: str = "outputs/backtest_results.jsonl") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in results_df.to_dict(orient="records"):
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    prices = load_btc_prices(limit=1500)

    results_df = run_backtest(
        prices=prices,
        train_window=500,
        test_window=720,
        n_sims=2000,
    )

    metrics = evaluate(results_df)
    save_jsonl(results_df)

    print("\nBacktest metrics:")
    print(f"Coverage 95%     : {metrics['coverage_95']:.4f}")
    print(f"Average width 95 : {metrics['average_width_95']:.2f}")
    print(f"Mean Winkler 95  : {metrics['mean_winkler_95']:.2f}")
    print(f"N predictions    : {metrics['n_predictions']}")
    print("\nSaved: outputs/backtest_results.jsonl")