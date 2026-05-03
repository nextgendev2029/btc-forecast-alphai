import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1)).dropna()


def estimate_recent_volatility(log_ret: pd.Series, vol_window: int = 80) -> float:
    recent = log_ret.tail(vol_window)
    sigma = recent.std()

    if pd.isna(sigma) or sigma <= 0:
        sigma = log_ret.std()

    return float(sigma)


def estimate_recent_drift(log_ret: pd.Series, drift_window: int = 200) -> float:
    recent = log_ret.tail(drift_window)
    mu = recent.mean()

    if pd.isna(mu):
        mu = log_ret.mean()

    return float(mu)


def simulate_next_prices(
    S0: float,
    mu: float,
    sigma: float,
    n_sims: int = 2000,
    tail_df: int = 4,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    z = rng.standard_t(df=tail_df, size=n_sims)

    # standardize Student-t so variance is ~1 when df > 2
    z = z * np.sqrt((tail_df - 2) / tail_df)

    next_prices = S0 * np.exp((mu - 0.5 * sigma**2) + sigma * z)

    return next_prices


def predict_next_range(
    prices: pd.Series,
    n_sims: int = 2000,
    vol_window: int = 80,
    drift_window: int = 200,
    tail_df: int = 4,
    range_scale: float = 1.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Predict 95% range for the next BTCUSDT hourly close.

    range_scale > 1.0 = wider range, higher coverage
    range_scale < 1.0 = tighter range, lower coverage
    """
    log_ret = compute_log_returns(prices)

    if len(log_ret) < max(10, vol_window):
        raise ValueError(f"Not enough returns to predict. Got {len(log_ret)}.")

    mu = estimate_recent_drift(log_ret, drift_window=drift_window)
    sigma = estimate_recent_volatility(log_ret, vol_window=vol_window)
    S0 = float(prices.iloc[-1])

    sims = simulate_next_prices(
        S0=S0,
        mu=mu,
        sigma=sigma,
        n_sims=n_sims,
        tail_df=tail_df,
        seed=seed,
    )

    low, high = np.percentile(sims, [2.5, 97.5])

    # widen/tighten around center
    center = S0
    low = center + range_scale * (low - center)
    high = center + range_scale * (high - center)

    return float(low), float(high)


if __name__ == "__main__":
    from src.data import load_btc_prices

    prices = load_btc_prices(limit=1500)

    low, high = predict_next_range(
    prices,
    n_sims=5000,
    vol_window=80,
    drift_window=200,
    tail_df=4,
    range_scale=1.05,
    seed=42,
    )

    print("Prediction (next hour):")
    print(f"Current price: {prices.iloc[-1]:.2f}")
    print(f"Lower 95%: {low:.2f}")
    print(f"Upper 95%: {high:.2f}")
    print(f"Width: {high - low:.2f}")