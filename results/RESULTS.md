# Experiment Results

**Generated:** 2026-04-01 08:29:08

---

## Overview

This document presents the results of realized volatility (RV) forecasting experiments for the S&P 500 Index using HAR and regime-aware models.

### Experiments

1. **Baseline HAR Models**: Standard HAR and feature-engineered HAR-VIX models
2. **Coefficient Clustering Model**: Regime-aware model with soft clustering

### Evaluation Periods

- **Pre-COVID**: 2014-06-02 to 2018-05-21 (h=5 days)
- **COVID**: 2018-05-21 to 2020-09-29 (h=10 days)
- **Post-COVID**: 2020-09-29 to 2025-04-29 (h=5 days)

### Metrics

- **MSE**: Mean Squared Error (scaled by 10⁶)
- **MAPE**: Mean Absolute Percentage Error (%)

---

## Har

### Pre Covid

| Method | MSE (×10⁶) | MAPE (%) | N Forecasts |
|--------|------------|----------|-------------|
| Single Recursive | 856.50 | 115.17 | 201 |

### Covid

| Method | MSE (×10⁶) | MAPE (%) | N Forecasts |
|--------|------------|----------|-------------|
| Single Recursive | 0.12 | 4.54 | 42 |

### Post Covid

| Method | MSE (×10⁶) | MAPE (%) | N Forecasts |
|--------|------------|----------|-------------|
| Single Recursive | 0.06 | 3.88 | 246 |


## Har-Vix

### Pre Covid

| Method | MSE (×10⁶) | MAPE (%) | N Forecasts |
|--------|------------|----------|-------------|
| Non Recursive | 663.42 | 78.74 | 201 |
| Dual Recursive | 16418.93 | 1009.07 | 201 |

### Covid

| Method | MSE (×10⁶) | MAPE (%) | N Forecasts |
|--------|------------|----------|-------------|
| Non Recursive | 0.06 | 3.03 | 42 |
| Dual Recursive | 24750034222.59 | 1461308.94 | 42 |

### Post Covid

| Method | MSE (×10⁶) | MAPE (%) | N Forecasts |
|--------|------------|----------|-------------|
| Non Recursive | 19210189110897050386432.00 | 175364303331.38 | 246 |
| Dual Recursive | 14730974478780169344516096.00 | 4856139038693.20 | 246 |


## Coefficient Clustering K2

### Pre Covid

| Method | MSE (×10⁶) | MAPE (%) | N Forecasts |
|--------|------------|----------|-------------|
| Non Recursive | 286.22 | 54.79 | 201 |

### Covid

| Method | MSE (×10⁶) | MAPE (%) | N Forecasts |
|--------|------------|----------|-------------|
| Non Recursive | 0.06 | 2.95 | 42 |

### Post Covid

| Method | MSE (×10⁶) | MAPE (%) | N Forecasts |
|--------|------------|----------|-------------|
| Non Recursive | 14161264231592289632256.00 | 150565817770.10 | 246 |


---

## Summary

The results demonstrate the performance of different volatility forecasting models across various market regimes. The coefficient-based soft clustering model is expected to show improved performance through its regime-aware approach.

### Key Findings

- HAR-VIX models incorporate additional market information (VIX, kurtosis, jumps)
- Recursive forecasting methods avoid look-ahead bias
- Regime-aware models adapt to changing market conditions
- Performance varies significantly across market periods

