"""
Tests for feature engineering module.
"""

import pytest
import numpy as np
import pandas as pd
from src.features import (
    compute_intraday_returns,
    compute_daily_rv,
    compute_daily_bv,
    compute_daily_jmp,
    compute_daily_kts,
    compute_lagged_averages
)


def test_compute_intraday_returns():
    """Test intraday return calculation."""
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=2, freq='D')
    timestamps = []
    prices = []
    
    for date in dates:
        for i in range(5):
            timestamps.append(pd.Timestamp(date) + pd.Timedelta(minutes=i*5))
            prices.append(100 + i)
    
    df = pd.DataFrame({
        'date': [t.date() for t in timestamps],
        'timestamp': timestamps,
        'close': prices
    })
    df['date'] = pd.to_datetime(df['date'])
    
    result = compute_intraday_returns(df)
    
    # Check that returns are computed
    assert 'return' in result.columns
    
    # Check that we have some non-NaN returns
    assert result['return'].notna().sum() > 0
    
    # Returns should be log returns
    # For prices going from 100 to 101, log return should be ln(101/100)
    expected_return = np.log(101/100)
    # Find a non-NaN return and check it's reasonable
    non_nan_returns = result['return'].dropna()
    if len(non_nan_returns) > 0:
        assert np.abs(non_nan_returns.iloc[0] - expected_return) < 0.01


def test_compute_daily_rv():
    """Test realized volatility calculation."""
    # Create sample data with known returns
    dates = pd.date_range('2020-01-01', periods=2, freq='D')
    data = []
    
    for date in dates:
        for i in range(78):  # Standard number of intervals
            data.append({
                'date': pd.Timestamp(date),
                'return': 0.001 if i > 0 else np.nan
            })
    
    df = pd.DataFrame(data)
    
    result = compute_daily_rv(df, N=78)
    
    # Check output structure
    assert len(result) == 2
    assert 'RV' in result.columns
    
    # RV should be positive
    assert all(result['RV'] > 0)
    
    # Test formula: RV = sqrt((N/n_t) * sum(r^2))
    # With n_t=77 (one NaN), r=0.001, N=78
    expected_rv = np.sqrt((78/77) * 77 * 0.001**2)
    assert np.isclose(result['RV'].iloc[0], expected_rv, rtol=0.01)


def test_compute_daily_bv():
    """Test bipower variation calculation."""
    dates = pd.date_range('2020-01-01', periods=1, freq='D')
    data = []
    
    for date in dates:
        for i in range(10):
            data.append({
                'date': pd.Timestamp(date),
                'return': 0.001 if i > 0 else np.nan
            })
    
    df = pd.DataFrame(data)
    
    result = compute_daily_bv(df)
    
    # Check output structure
    assert len(result) == 1
    assert 'BV' in result.columns
    
    # BV should be positive
    assert result['BV'].iloc[0] > 0
    
    # BV formula: (pi/2) * sum(|r_i| * |r_{i-1}|)
    expected_bv = (np.pi / 2) * 8 * 0.001 * 0.001
    assert np.isclose(result['BV'].iloc[0], expected_bv, rtol=0.01)


def test_compute_daily_jmp():
    """Test jump variation calculation."""
    rv_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=3, freq='D'),
        'RV': [0.5, 0.6, 0.4]
    })
    
    bv_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=3, freq='D'),
        'BV': [0.3, 0.7, 0.5]
    })
    
    result = compute_daily_jmp(rv_df, bv_df)
    
    # Check output structure
    assert len(result) == 3
    assert 'JMP' in result.columns
    
    # JMP = max(0, RV - BV)
    assert result['JMP'].iloc[0] == 0.2  # 0.5 - 0.3
    assert result['JMP'].iloc[1] == 0.0  # max(0, 0.6 - 0.7)
    assert result['JMP'].iloc[2] == 0.0  # max(0, 0.4 - 0.5)
    
    # JMP should never be negative
    assert all(result['JMP'] >= 0)


def test_compute_daily_kts():
    """Test realized kurtosis calculation."""
    dates = pd.date_range('2020-01-01', periods=1, freq='D')
    returns = [0.001, 0.002, -0.001, 0.003, -0.002]
    
    data = []
    for date in dates:
        for i, ret in enumerate([np.nan] + returns):
            data.append({
                'date': pd.Timestamp(date),
                'return': ret
            })
    
    df = pd.DataFrame(data)
    
    result = compute_daily_kts(df)
    
    # Check output structure
    assert len(result) == 1
    assert 'KTS' in result.columns
    
    # KTS should be positive
    assert result['KTS'].iloc[0] > 0
    
    # KTS formula: (n * sum(r^4)) / (sum(r^2))^2
    returns_arr = np.array(returns)
    n = len(returns_arr)
    expected_kts = (n * np.sum(returns_arr**4)) / (np.sum(returns_arr**2)**2)
    assert np.isclose(result['KTS'].iloc[0], expected_kts, rtol=0.01)


def test_compute_lagged_averages():
    """Test lagged moving average calculation."""
    df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=30, freq='D'),
        'RV': np.random.rand(30) * 0.5 + 0.2,
        'VIX': np.random.rand(30) * 10 + 15
    })
    
    result = compute_lagged_averages(df, ['RV', 'VIX'])
    
    # Check that lagged columns are created
    assert 'RV_w' in result.columns
    assert 'RV_m' in result.columns
    assert 'VIX_w' in result.columns
    assert 'VIX_m' in result.columns
    
    # Check 5-day average (weekly)
    rv_w_manual = result['RV'].rolling(window=5, min_periods=1).mean()
    assert np.allclose(result['RV_w'], rv_w_manual)
    
    # Check 22-day average (monthly)
    rv_m_manual = result['RV'].rolling(window=22, min_periods=1).mean()
    assert np.allclose(result['RV_m'], rv_m_manual)


def test_edge_case_empty_data():
    """Test handling of empty data."""
    df = pd.DataFrame(columns=['date', 'timestamp', 'close', 'return'])
    
    result = compute_daily_rv(df, N=78)
    assert len(result) == 0


def test_edge_case_single_observation():
    """Test handling of single observation per day."""
    df = pd.DataFrame({
        'date': [pd.Timestamp('2020-01-01')],
        'return': [0.001]
    })
    
    result = compute_daily_rv(df, N=78)
    
    # Should handle single observation
    assert len(result) == 1
    assert result['RV'].iloc[0] > 0


def test_rv_scaling_factor():
    """Test that RV scaling factor N is applied correctly."""
    # Create data with known number of observations
    n_obs = 50
    df = pd.DataFrame({
        'date': [pd.Timestamp('2020-01-01')] * (n_obs + 1),
        'return': [np.nan] + [0.001] * n_obs
    })
    
    N = 78
    result = compute_daily_rv(df, N=N)
    
    # RV should scale with sqrt(N/n_t)
    expected_rv = np.sqrt((N / n_obs) * n_obs * 0.001**2)
    assert np.isclose(result['RV'].iloc[0], expected_rv, rtol=0.01)


def test_kurtosis_properties():
    """Test that kurtosis has expected mathematical properties."""
    # Normal distribution should have kurtosis around 3
    np.random.seed(42)
    normal_returns = np.random.normal(0, 0.001, 1000)
    
    df = pd.DataFrame({
        'date': [pd.Timestamp('2020-01-01')] * 1001,
        'return': [np.nan] + list(normal_returns)
    })
    
    result = compute_daily_kts(df)
    
    # Kurtosis should be positive
    assert result['KTS'].iloc[0] > 0
    
    # For normal distribution, should be around 3
    # (allowing wide tolerance due to sample size)
    assert 2 < result['KTS'].iloc[0] < 4


def test_jmp_non_negative():
    """Test that jump variation is always non-negative."""
    # Create random RV and BV values
    np.random.seed(42)
    n = 100
    
    rv_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n, freq='D'),
        'RV': np.random.rand(n) * 0.5 + 0.2
    })
    
    bv_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n, freq='D'),
        'BV': np.random.rand(n) * 0.5 + 0.2
    })
    
    result = compute_daily_jmp(rv_df, bv_df)
    
    # All JMP values should be non-negative
    assert all(result['JMP'] >= 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
