"""
Experiment 1: Baseline HAR and Feature-Engineered HAR-VIX Models

This module implements standard HAR and HAR-VIX models with three forecasting methodologies:
1. Non-recursive forecasting
2. Single-recursive forecasting (RV only)
3. Dual-recursive forecasting (RV and VIX)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class BaselineHARModel:
    """
    Baseline HAR model using daily, weekly, and monthly RV lags.
    """
    
    def __init__(self, include_vix: bool = False, include_kts_jmp: bool = False):
        """
        Initialize HAR model.
        
        Parameters
        ----------
        include_vix : bool
            Whether to include VIX features (HAR-VIX model)
        include_kts_jmp : bool
            Whether to include KTS and JMP features
        """
        self.include_vix = include_vix
        self.include_kts_jmp = include_kts_jmp
        self.model = LinearRegression()
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_names = []
        
    def _get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        features = ['RV', 'RV_w', 'RV_m']
        
        if self.include_vix:
            features.extend(['VIX', 'VIX_w', 'VIX_m'])
        
        if self.include_kts_jmp:
            features.extend(['KTS', 'JMP'])
        
        return features
    
    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fit the HAR model on training data.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with all features
        """
        self.feature_names = self._get_feature_columns()
        
        # Prepare features (lagged by 1 day)
        X = train_df[self.feature_names].shift(1).values[1:]
        y = train_df['RV'].values[1:]
        
        # Z-score normalization
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        X_norm = self.feature_scaler.fit_transform(X)
        y_norm = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Fit OLS model
        self.model.fit(X_norm, y_norm)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on normalized features.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predictions in original scale
        """
        X_norm = self.feature_scaler.transform(X)
        y_pred_norm = self.model.predict(X_norm)
        y_pred = self.target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        
        return y_pred
    
    def forecast_non_recursive(
        self,
        test_df: pd.DataFrame,
        horizon: int
    ) -> np.ndarray:
        """
        Non-recursive h-step ahead forecasting.
        
        Assumes features at t+h-1 are available at time t.
        
        Parameters
        ----------
        test_df : pd.DataFrame
            Test data
        horizon : int
            Forecast horizon
            
        Returns
        -------
        np.ndarray
            Forecasts for each test point
        """
        forecasts = []
        
        for i in range(len(test_df)):
            if i < horizon:
                # Not enough history for h-step ahead
                continue
            
            # Use features from t+h-1 to predict t+h
            X = test_df[self.feature_names].iloc[i-1:i].values
            y_pred = self.predict(X)
            forecasts.append(y_pred[0])
        
        return np.array(forecasts)
    
    def forecast_single_recursive(
        self,
        train_df: pd.DataFrame,
        horizon: int
    ) -> float:
        """
        Single-recursive h-step ahead forecasting (RV only).
        
        Iteratively forecasts RV for h steps, using previous forecasts as features.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data (used to get initial values)
        horizon : int
            Forecast horizon
            
        Returns
        -------
        float
            Forecast at t+h
        """
        # Use only RV features for recursive forecasting
        rv_features = ['RV', 'RV_w', 'RV_m']
        
        # Initialize history with last values from training data
        history = train_df[['RV']].copy()
        
        for step in range(1, horizon + 1):
            # Compute lagged features from history
            rv_daily = history['RV'].iloc[-1]
            rv_weekly = history['RV'].iloc[-5:].mean() if len(history) >= 5 else rv_daily
            rv_monthly = history['RV'].iloc[-22:].mean() if len(history) >= 22 else rv_daily
            
            # Create feature vector
            X = np.array([[rv_daily, rv_weekly, rv_monthly]])
            
            # Normalize and predict
            X_norm = self.feature_scaler.transform(
                np.pad(X, ((0, 0), (0, len(self.feature_names) - 3)), mode='constant')
            )[:, :3]
            
            # Use only RV coefficients
            y_pred_norm = (X_norm @ self.model.coef_[:3]) + self.model.intercept_
            y_pred = self.target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1))[0, 0]
            
            # Add forecast to history
            new_row = pd.DataFrame({'RV': [y_pred]})
            history = pd.concat([history, new_row], ignore_index=True)
        
        return history['RV'].iloc[-1]
    
    def forecast_dual_recursive(
        self,
        train_df: pd.DataFrame,
        vix_model: 'VIXModel',
        horizon: int
    ) -> float:
        """
        Dual-recursive h-step ahead forecasting (RV and VIX).
        
        Iteratively forecasts both VIX and RV for h steps.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        vix_model : VIXModel
            Fitted VIX forecasting model
        horizon : int
            Forecast horizon
            
        Returns
        -------
        float
            RV forecast at t+h
        """
        # Initialize history
        rv_history = train_df[['RV']].copy()
        vix_history = train_df[['VIX']].copy()
        
        for step in range(1, horizon + 1):
            # Forecast VIX first
            vix_pred = vix_model.forecast_one_step(rv_history, vix_history)
            
            # Compute RV features
            rv_daily = rv_history['RV'].iloc[-1]
            rv_weekly = rv_history['RV'].iloc[-5:].mean() if len(rv_history) >= 5 else rv_daily
            rv_monthly = rv_history['RV'].iloc[-22:].mean() if len(rv_history) >= 22 else rv_daily
            
            # Compute VIX features
            vix_daily = vix_pred
            vix_weekly = vix_history['VIX'].iloc[-4:].mean() if len(vix_history) >= 4 else vix_pred
            vix_weekly = (vix_weekly * 4 + vix_pred) / 5  # Include current prediction
            vix_monthly = vix_history['VIX'].iloc[-21:].mean() if len(vix_history) >= 21 else vix_pred
            vix_monthly = (vix_monthly * 21 + vix_pred) / 22
            
            # Get KTS and JMP (use last known values)
            kts = train_df['KTS'].iloc[-1]
            jmp = train_df['JMP'].iloc[-1]
            
            # Create feature vector
            X = np.array([[rv_daily, rv_weekly, rv_monthly, 
                          vix_daily, vix_weekly, vix_monthly, kts, jmp]])
            
            # Predict RV
            rv_pred = self.predict(X)[0]
            
            # Update histories
            rv_history = pd.concat([rv_history, pd.DataFrame({'RV': [rv_pred]})], ignore_index=True)
            vix_history = pd.concat([vix_history, pd.DataFrame({'VIX': [vix_pred]})], ignore_index=True)
        
        return rv_history['RV'].iloc[-1]


class VIXModel:
    """
    Model for forecasting VIX using RV and VIX lags.
    """
    
    def __init__(self):
        """Initialize VIX model."""
        self.model = LinearRegression()
        self.feature_scaler = None
        self.target_scaler = None
    
    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fit VIX model on training data.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data with RV and VIX features
        """
        # Features: RV lags and VIX lags
        feature_cols = ['RV', 'RV_w', 'RV_m', 'VIX', 'VIX_w', 'VIX_m']
        
        X = train_df[feature_cols].shift(1).values[1:]
        y = train_df['VIX'].values[1:]
        
        # Z-score normalization
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        X_norm = self.feature_scaler.fit_transform(X)
        y_norm = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Fit OLS model
        self.model.fit(X_norm, y_norm)
    
    def forecast_one_step(
        self,
        rv_history: pd.DataFrame,
        vix_history: pd.DataFrame
    ) -> float:
        """
        Forecast VIX one step ahead.
        
        Parameters
        ----------
        rv_history : pd.DataFrame
            Historical RV values
        vix_history : pd.DataFrame
            Historical VIX values
            
        Returns
        -------
        float
            VIX forecast
        """
        # Compute features
        rv_daily = rv_history['RV'].iloc[-1]
        rv_weekly = rv_history['RV'].iloc[-5:].mean() if len(rv_history) >= 5 else rv_daily
        rv_monthly = rv_history['RV'].iloc[-22:].mean() if len(rv_history) >= 22 else rv_daily
        
        vix_daily = vix_history['VIX'].iloc[-1]
        vix_weekly = vix_history['VIX'].iloc[-5:].mean() if len(vix_history) >= 5 else vix_daily
        vix_monthly = vix_history['VIX'].iloc[-22:].mean() if len(vix_history) >= 22 else vix_daily
        
        X = np.array([[rv_daily, rv_weekly, rv_monthly, vix_daily, vix_weekly, vix_monthly]])
        
        # Normalize and predict
        X_norm = self.feature_scaler.transform(X)
        y_pred_norm = self.model.predict(X_norm)
        y_pred = self.target_scaler.inverse_transform(y_pred_norm.reshape(-1, 1))[0, 0]
        
        return y_pred


def run_rolling_backtest(
    df: pd.DataFrame,
    period: str,
    model_type: str = 'har_vix',
    forecast_method: str = 'non_recursive',
    train_window: int = 441
) -> Dict[str, float]:
    """
    Run rolling window backtest for a given period and model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    period : str
        One of 'pre_covid', 'covid', 'post_covid'
    model_type : str
        'har' or 'har_vix'
    forecast_method : str
        'non_recursive', 'single_recursive', or 'dual_recursive'
    train_window : int
        Size of training window
        
    Returns
    -------
    Dict[str, float]
        Dictionary with 'mse' and 'mape' metrics
    """
    from src.utils import get_period_dates, compute_mse, compute_mape
    
    # Get period parameters
    start_date, end_date, horizon = get_period_dates(period)
    
    # Filter data for period
    period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    period_df = period_df.reset_index(drop=True)
    
    print(f"\nRunning {model_type} - {forecast_method} for {period}")
    print(f"Period: {start_date} to {end_date}, Horizon: {horizon}")
    print(f"Total days: {len(period_df)}")
    
    # Determine model configuration
    include_vix = (model_type == 'har_vix')
    include_kts_jmp = (model_type == 'har_vix')
    
    # Storage for predictions
    all_predictions = []
    all_actuals = []
    
    # Rolling window
    step_size = horizon
    start_idx = 0
    
    while start_idx + train_window + horizon <= len(period_df):
        train_end_idx = start_idx + train_window
        test_idx = train_end_idx + horizon - 1  # Index for t+h
        
        train_df = period_df.iloc[start_idx:train_end_idx].copy()
        
        # Fit model
        model = BaselineHARModel(
            include_vix=include_vix,
            include_kts_jmp=include_kts_jmp
        )
        model.fit(train_df)
        
        # Make forecast based on method
        if forecast_method == 'non_recursive':
            # Use features at t+h-1 to predict t+h
            if test_idx < len(period_df):
                X = period_df[model.feature_names].iloc[test_idx:test_idx+1].values
                y_pred = model.predict(X)[0]
                y_actual = period_df['RV'].iloc[test_idx]
                
                all_predictions.append(y_pred)
                all_actuals.append(y_actual)
        
        elif forecast_method == 'single_recursive':
            # Recursive forecasting with RV only
            if not include_vix:
                y_pred = model.forecast_single_recursive(train_df, horizon)
                y_actual = period_df['RV'].iloc[test_idx]
                
                all_predictions.append(y_pred)
                all_actuals.append(y_actual)
        
        elif forecast_method == 'dual_recursive':
            # Recursive forecasting with RV and VIX
            if include_vix:
                # Fit VIX model
                vix_model = VIXModel()
                vix_model.fit(train_df)
                
                y_pred = model.forecast_dual_recursive(train_df, vix_model, horizon)
                y_actual = period_df['RV'].iloc[test_idx]
                
                all_predictions.append(y_pred)
                all_actuals.append(y_actual)
        
        start_idx += step_size
    
    # Compute metrics
    if len(all_predictions) > 0:
        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)
        
        mse = compute_mse(actuals, predictions)
        mape = compute_mape(actuals, predictions)
        
        print(f"Forecasts: {len(predictions)}")
        print(f"MSE (×10⁶): {mse:.2f}")
        print(f"MAPE (%): {mape:.2f}")
        
        return {'mse': mse, 'mape': mape, 'n_forecasts': len(predictions)}
    else:
        print("No forecasts generated")
        return {'mse': np.nan, 'mape': np.nan, 'n_forecasts': 0}


def run_experiment_1(df: pd.DataFrame) -> Dict:
    """
    Run complete Experiment 1 with all model configurations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataset
        
    Returns
    -------
    Dict
        Results for all periods and methods
    """
    results = {
        'HAR': {},
        'HAR-VIX': {}
    }
    
    periods = ['pre_covid', 'covid', 'post_covid']
    
    # HAR model (RV only)
    print("\n" + "="*80)
    print("BASELINE HAR MODEL (RV only)")
    print("="*80)
    
    for period in periods:
        results['HAR'][period] = {}
        
        # Non-recursive not applicable for pure HAR in recursive mode
        # Single-recursive
        metrics = run_rolling_backtest(
            df, period, model_type='har', 
            forecast_method='single_recursive'
        )
        results['HAR'][period]['single_recursive'] = metrics
    
    # HAR-VIX model
    print("\n" + "="*80)
    print("FEATURE-ENGINEERED HAR-VIX MODEL")
    print("="*80)
    
    for period in periods:
        results['HAR-VIX'][period] = {}
        
        # Non-recursive
        metrics = run_rolling_backtest(
            df, period, model_type='har_vix',
            forecast_method='non_recursive'
        )
        results['HAR-VIX'][period]['non_recursive'] = metrics
        
        # Dual-recursive
        metrics = run_rolling_backtest(
            df, period, model_type='har_vix',
            forecast_method='dual_recursive'
        )
        results['HAR-VIX'][period]['dual_recursive'] = metrics
    
    return results
