"""
Tests for Experiment 1: Baseline HAR models.
"""

import pytest
import numpy as np
import pandas as pd
from src.exp_1_baseline_har import BaselineHARModel, VIXModel


def create_sample_data(n_days: int = 500) -> pd.DataFrame:
    """Create sample data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Generate synthetic RV and VIX data
    rv = np.random.rand(n_days) * 0.3 + 0.2
    vix = np.random.rand(n_days) * 10 + 15
    
    df = pd.DataFrame({
        'date': dates,
        'RV': rv,
        'VIX': vix,
        'KTS': np.random.rand(n_days) * 2 + 2,
        'JMP': np.random.rand(n_days) * 0.1
    })
    
    # Compute lagged averages
    df['RV_w'] = df['RV'].rolling(window=5, min_periods=1).mean()
    df['RV_m'] = df['RV'].rolling(window=22, min_periods=1).mean()
    df['VIX_w'] = df['VIX'].rolling(window=5, min_periods=1).mean()
    df['VIX_m'] = df['VIX'].rolling(window=22, min_periods=1).mean()
    
    return df


def test_baseline_har_model_initialization():
    """Test HAR model initialization."""
    # HAR model (RV only)
    model = BaselineHARModel(include_vix=False, include_kts_jmp=False)
    assert not model.include_vix
    assert not model.include_kts_jmp
    assert model._get_feature_columns() == ['RV', 'RV_w', 'RV_m']
    
    # HAR-VIX model
    model = BaselineHARModel(include_vix=True, include_kts_jmp=True)
    assert model.include_vix
    assert model.include_kts_jmp
    expected_features = ['RV', 'RV_w', 'RV_m', 'VIX', 'VIX_w', 'VIX_m', 'KTS', 'JMP']
    assert model._get_feature_columns() == expected_features


def test_baseline_har_model_fit():
    """Test HAR model fitting."""
    df = create_sample_data(n_days=500)
    
    model = BaselineHARModel(include_vix=True, include_kts_jmp=True)
    model.fit(df)
    
    # Check that scalers are fitted
    assert model.feature_scaler is not None
    assert model.target_scaler is not None
    
    # Check that model is fitted
    assert hasattr(model.model, 'coef_')
    assert hasattr(model.model, 'intercept_')
    
    # Check coefficient dimensions
    assert len(model.model.coef_) == 8  # 8 features


def test_baseline_har_model_predict():
    """Test HAR model prediction."""
    df = create_sample_data(n_days=500)
    
    model = BaselineHARModel(include_vix=True, include_kts_jmp=True)
    model.fit(df.iloc[:400])
    
    # Make predictions on test data
    X_test = df[model.feature_names].iloc[400:410].values
    predictions = model.predict(X_test)
    
    # Check prediction shape
    assert len(predictions) == 10
    
    # Predictions should be positive (RV is always positive)
    assert all(predictions > 0)
    
    # Predictions should be reasonable (not too extreme)
    assert all(predictions < 2.0)


def test_vix_model_fit_and_predict():
    """Test VIX model fitting and prediction."""
    df = create_sample_data(n_days=500)
    
    model = VIXModel()
    model.fit(df.iloc[:400])
    
    # Check that model is fitted
    assert hasattr(model.model, 'coef_')
    assert len(model.model.coef_) == 6  # 6 features (RV and VIX lags)
    
    # Test one-step forecast
    rv_history = df[['RV']].iloc[:400]
    vix_history = df[['VIX']].iloc[:400]
    
    vix_pred = model.forecast_one_step(rv_history, vix_history)
    
    # VIX prediction should be positive and reasonable
    assert vix_pred > 0
    assert vix_pred < 100


def test_non_recursive_forecasting():
    """Test non-recursive forecasting methodology."""
    df = create_sample_data(n_days=500)
    
    model = BaselineHARModel(include_vix=True, include_kts_jmp=True)
    model.fit(df.iloc[:400])
    
    # Test non-recursive forecast
    test_df = df.iloc[400:450]
    horizon = 5
    
    forecasts = model.forecast_non_recursive(test_df, horizon)
    
    # Should generate forecasts
    assert len(forecasts) > 0
    
    # All forecasts should be positive
    assert all(forecasts > 0)


def test_single_recursive_forecasting():
    """Test single-recursive forecasting methodology."""
    df = create_sample_data(n_days=500)
    
    model = BaselineHARModel(include_vix=False, include_kts_jmp=False)
    model.fit(df.iloc[:400])
    
    # Test single-recursive forecast
    train_df = df.iloc[:400]
    horizon = 5
    
    forecast = model.forecast_single_recursive(train_df, horizon)
    
    # Should return a single forecast value
    assert isinstance(forecast, (float, np.floating))
    
    # Forecast should be positive
    assert forecast > 0


def test_dual_recursive_forecasting():
    """Test dual-recursive forecasting methodology."""
    df = create_sample_data(n_days=500)
    
    # Fit HAR-VIX model
    har_model = BaselineHARModel(include_vix=True, include_kts_jmp=True)
    har_model.fit(df.iloc[:400])
    
    # Fit VIX model
    vix_model = VIXModel()
    vix_model.fit(df.iloc[:400])
    
    # Test dual-recursive forecast
    train_df = df.iloc[:400]
    horizon = 5
    
    forecast = har_model.forecast_dual_recursive(train_df, vix_model, horizon)
    
    # Should return a single forecast value
    assert isinstance(forecast, (float, np.floating))
    
    # Forecast should be positive
    assert forecast > 0


def test_feature_lagging():
    """Test that features are properly lagged to avoid look-ahead bias."""
    df = create_sample_data(n_days=100)
    
    model = BaselineHARModel(include_vix=True, include_kts_jmp=True)
    
    # Extract features as done in fit method
    X = df[model._get_feature_columns()].shift(1).values[1:]
    y = df['RV'].values[1:]
    
    # Check that X is lagged by 1
    assert len(X) == len(df) - 1
    assert len(y) == len(df) - 1
    
    # First row of X should match second row of original features
    assert np.allclose(X[0], df[model._get_feature_columns()].iloc[0].values)


def test_zscore_normalization():
    """Test that Z-score normalization is applied correctly."""
    df = create_sample_data(n_days=500)
    
    model = BaselineHARModel(include_vix=True, include_kts_jmp=True)
    model.fit(df.iloc[:400])
    
    # Check that scalers have correct statistics
    assert model.feature_scaler.mean_ is not None
    assert model.feature_scaler.scale_ is not None
    assert model.target_scaler.mean_ is not None
    assert model.target_scaler.scale_ is not None
    
    # Mean should be close to 0 after normalization
    X = df[model.feature_names].shift(1).values[1:401]
    X_norm = model.feature_scaler.transform(X)
    assert np.abs(X_norm.mean()) < 0.1


def test_model_coefficients_reasonable():
    """Test that fitted model coefficients are reasonable."""
    df = create_sample_data(n_days=500)
    
    model = BaselineHARModel(include_vix=True, include_kts_jmp=True)
    model.fit(df)
    
    # Coefficients should not be extreme
    assert all(np.abs(model.model.coef_) < 10)
    
    # Intercept should not be extreme
    assert np.abs(model.model.intercept_) < 10


def test_recursive_forecast_convergence():
    """Test that recursive forecasts don't diverge."""
    df = create_sample_data(n_days=500)
    
    model = BaselineHARModel(include_vix=False, include_kts_jmp=False)
    model.fit(df.iloc[:400])
    
    train_df = df.iloc[:400]
    
    # Test multiple horizons
    for horizon in [1, 5, 10, 20]:
        forecast = model.forecast_single_recursive(train_df, horizon)
        
        # Forecast should remain positive and reasonable
        assert forecast > 0
        assert forecast < 5.0  # Should not explode


def test_different_train_window_sizes():
    """Test model with different training window sizes."""
    df = create_sample_data(n_days=500)
    
    for window_size in [100, 200, 441]:
        if window_size > len(df):
            continue
        
        model = BaselineHARModel(include_vix=True, include_kts_jmp=True)
        model.fit(df.iloc[:window_size])
        
        # Model should fit successfully
        assert hasattr(model.model, 'coef_')
        
        # Should be able to make predictions
        X_test = df[model.feature_names].iloc[window_size:window_size+10].values
        predictions = model.predict(X_test)
        assert len(predictions) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
