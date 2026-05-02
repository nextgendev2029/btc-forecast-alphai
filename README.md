# BTCUSDT Next-Hour 95% Forecast

AlphaI × Polaris assignment submission.

This project predicts the **95% price range for BTCUSDT one hour ahead** using Binance public 1-hour candle data and a Student-t GBM simulation model.

---

## Live Dashboard

Dashboard URL: [Live Demo](https://btc-forecast-alphai.streamlit.app/)

---

## Repository Structure

```text
btc-forecast-alphai/
├── app/
│   └── app.py
├── notebook/
│   └── BTC_Forecast_AlphaI_Final.ipynb
├── outputs/
│   ├── backtest_results.jsonl
│   └── tuning_results.csv
├── src/
│   ├── data.py
│   ├── model.py
│   ├── backtest.py
│   └── tune.py
├── requirements.txt
└── README.md
```

## Data Source

Data is fetched from Binance public API:

- **Endpoint:** `https://data-api.binance.vision/api/v3/klines`
- **Symbol:** `BTCUSDT`
- **Interval:** `1 hour`
- **Data window:** latest 1500 hourly candles
- *No API key required*

## Model

This model focuses on probabilistic forecasting rather than point prediction, providing calibrated uncertainty estimates.

The forecasting model is a one-step simulation based on:

- Log returns of BTC price
- Recent drift (mean return)
- Recent volatility (rolling window)
- Student-t distribution for fat-tailed shocks

### Why Student-t?

Bitcoin hourly returns have fat tails, meaning large price moves happen more often than a normal distribution expects. Student-t shocks prevent overconfident narrow ranges.

### Final Parameters
```python
train_window = 500
vol_window = 80
drift_window = 200
tail_df = 4
range_scale = 1.05
n_sims = 2000
```

## No-Peeking Backtest

For each prediction:

1. Use only past data up to time *t*
2. Predict next-hour range
3. Compare with actual next-hour close
4. Record performance

This ensures no future data leakage.

### Backtest Results (720 Predictions)
```text
coverage_95      = 0.9486
average_width_95 = 1195.41
mean_winkler_95  = 1774.43
n_predictions    = 720
```

Backtest file: `outputs/backtest_results.jsonl`

## Hyperparameter Tuning

Grid search was performed over:

- `range_scale`
- `vol_window`
- `tail_df`

**Objective:**
- coverage ≈ 0.95
- minimize Winkler score

Results saved in: `outputs/tuning_results.csv`

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run backtest
```bash
export PYTHONPATH=.
python src/backtest.py
```

### Run dashboard
```bash
export PYTHONPATH=.
streamlit run app/app.py
```

## Dashboard Features
- Live BTCUSDT price
- Next-hour 95% prediction range
- Range width
- Backtest metrics
- Last 50 candles + forecast ribbon

## Files for Review
- **Notebook:** `notebook/BTC_Forecast_AlphaI_Final.ipynb`
- **Dashboard:** `app/app.py`
- **Model:** `src/model.py`
- **Backtest:** `src/backtest.py`
- **Results:** `outputs/backtest_results.jsonl`
