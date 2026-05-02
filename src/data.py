import requests
import pandas as pd


BINANCE_BASE_URL = "https://data-api.binance.vision/api/v3/klines"


def _raw_klines_to_df(raw: list) -> pd.DataFrame:
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]

    df = pd.DataFrame(raw, columns=columns)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["number_of_trades"] = pd.to_numeric(
        df["number_of_trades"],
        errors="coerce",
    ).astype("Int64")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df = df.sort_values("open_time").reset_index(drop=True)

    return df


def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 1500,
) -> pd.DataFrame:
    """
    Fetch Binance OHLCV candles.

    Binance gives limited rows per request, so this function paginates backward
    until we collect the requested number of bars.
    """
    all_raw = []
    remaining = limit
    end_time = None

    while remaining > 0:
        batch_limit = min(1000, remaining)

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": batch_limit,
        }

        if end_time is not None:
            params["endTime"] = end_time

        response = requests.get(BINANCE_BASE_URL, params=params, timeout=15)

        if response.status_code != 200:
            raise RuntimeError(f"Binance API error {response.status_code}: {response.text}")

        raw = response.json()

        if not raw:
            break

        all_raw = raw + all_raw

        earliest_open_time = raw[0][0]
        end_time = earliest_open_time - 1

        remaining -= len(raw)

        if len(raw) < batch_limit:
            break

    df = _raw_klines_to_df(all_raw)

    df = df.drop_duplicates(subset=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    df = df.tail(limit).reset_index(drop=True)

    return df


def get_close_prices(df: pd.DataFrame) -> pd.Series:
    prices = df.set_index("close_time")["close"].copy()
    prices.name = "close"
    return prices


def validate_hourly_data(prices: pd.Series) -> None:
    if prices.empty:
        raise ValueError("Price series is empty.")

    if prices.isna().any():
        raise ValueError("Price series contains NaN values.")

    if len(prices) < 100:
        raise ValueError(f"Too few price bars: {len(prices)}")

    time_diffs = prices.index.to_series().diff().dropna()
    most_common_gap = time_diffs.value_counts().index[0]

    if most_common_gap != pd.Timedelta(hours=1):
        raise ValueError(f"Expected hourly data, got most common gap: {most_common_gap}")


def load_btc_prices(limit: int = 1500) -> pd.Series:
    df = fetch_binance_klines(symbol="BTCUSDT", interval="1h", limit=limit)
    prices = get_close_prices(df)
    validate_hourly_data(prices)
    return prices


if __name__ == "__main__":
    prices = load_btc_prices(limit=1500)

    print("BTCUSDT hourly data loaded successfully.")
    print("Total bars:", len(prices))
    print("First timestamp:", prices.index[0])
    print("Last timestamp:", prices.index[-1])
    print("Latest close:", prices.iloc[-1])