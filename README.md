# Baseline HAR and Feature-Engineered HAR-VIX Models for S&P 500 RV Forecasting

This repository implements two sophisticated experiments for forecasting realized volatility (RV) of the S&P 500 Index using Heterogeneous Autoregressive (HAR) models and regime-aware extensions.

## Experiments

### Experiment 1: Baseline HAR and HAR-VIX Models
Establishes baseline performance for RV forecasting using:
- Standard HAR model (daily, weekly, monthly RV lags)
- Feature-engineered HAR-VIX model (adds VIX, realized kurtosis, jump variation)
- Three forecasting methodologies: non-recursive, single-recursive, dual-recursive
- Evaluation across Pre-COVID, COVID, and Post-COVID periods

### Experiment 4: Coefficient-Based Soft Clustering HAR-VIX
Advanced regime-aware model featuring:
- Mood's Median Test for variance change point detection
- PCA dimensionality reduction of HAR-VIX coefficients
- Bayesian Gaussian Mixture Model (BGMM) for soft clustering
- XGBoost model for regime probability prediction
- Weighted ensemble forecasting

## Project Structure

```
.
├── src/
│   ├── data_loader.py              # Data fetching and preprocessing
│   ├── features.py                 # Feature engineering (RV, BV, JMP, KTS)
│   ├── exp_1_baseline_har.py       # Experiment 1 implementation
│   ├── exp_4_coefficient_clustering.py  # Experiment 4 implementation
│   └── utils.py                    # Utility functions
├── tests/
│   ├── test_data_loader.py
│   ├── test_features.py
│   ├── test_exp_1.py
│   └── test_exp_4.py
├── results/
│   └── RESULTS.md                  # Experiment results and metrics
├── run_experiments.py              # Main execution script
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Set the following environment variable:
```bash
export MASSIVE_TOKEN="your_api_key_here"
```

## Usage

Run all experiments:
```bash
python run_experiments.py
```

Run tests:
```bash
pytest tests/ -v
```

## Data Requirements

- **SPX Intraday Prices**: 5-minute close prices (2014-05-01 to 2025-05-27)
- **VIX Daily Levels**: Daily closing values for the same period
- Data is fetched using the Massive API

## Methodology Highlights

### Feature Construction
- **Realized Volatility (RV)**: `sqrt((N/n_t) * sum(r_{t,i}^2))`
- **Bipower Variation (BV)**: `(pi/2) * sum(|r_{t,i}| * |r_{t,i-1}|)`
- **Jump Variation (JMP)**: `max(0, RV_t - BV_t)`
- **Realized Kurtosis (KTS)**: `(n_t * sum(r_{t,i}^4)) / (sum(r_{t,i}^2))^2`

### Rolling Window Framework
- Training window: 441 observations
- Forecast horizons: 5 days (Pre/Post-COVID), 10 days (COVID)
- Z-score normalization per window
- OLS regression for model fitting

### Performance Metrics
- Mean Squared Error (MSE) scaled by 10^6
- Mean Absolute Percentage Error (MAPE)

## Expected Results

Results are saved in `results/RESULTS.md` with detailed metrics for:
- Each experiment
- Each market period (Pre-COVID, COVID, Post-COVID)
- Each forecasting methodology (non-recursive, single-recursive, dual-recursive)

## References

Implementation follows the methodology described in the research paper on regime-aware volatility forecasting.

## License

MIT License
