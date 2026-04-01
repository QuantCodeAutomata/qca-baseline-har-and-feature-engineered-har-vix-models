"""
Utility functions for model training and evaluation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler


def zscore_normalize(
    train_data: np.ndarray,
    test_data: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Apply Z-score normalization to training and test data.
    
    Parameters
    ----------
    train_data : np.ndarray
        Training data to fit the scaler
    test_data : np.ndarray, optional
        Test data to transform
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, StandardScaler]
        Normalized training data, normalized test data (or None), and fitted scaler
    """
    scaler = StandardScaler()
    train_normalized = scaler.fit_transform(train_data)
    
    if test_data is not None:
        test_normalized = scaler.transform(test_data)
    else:
        test_normalized = None
    
    return train_normalized, test_normalized, scaler


def inverse_zscore(
    data: np.ndarray,
    scaler: StandardScaler
) -> np.ndarray:
    """
    Apply inverse Z-score transformation.
    
    Parameters
    ----------
    data : np.ndarray
        Normalized data
    scaler : StandardScaler
        Fitted scaler
        
    Returns
    -------
    np.ndarray
        Original scale data
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    return scaler.inverse_transform(data).flatten()


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray, scale: float = 1e6) -> float:
    """
    Compute Mean Squared Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    scale : float
        Scaling factor for MSE (default: 1e6 as per paper)
        
    Returns
    -------
    float
        Scaled MSE
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return mse * scale


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Percentage Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        MAPE as percentage
    """
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    
    mape = 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return mape


def get_period_dates(period: str) -> Tuple[str, str, int]:
    """
    Get start date, end date, and forecast horizon for a given period.
    
    Parameters
    ----------
    period : str
        One of 'pre_covid', 'covid', 'post_covid'
        
    Returns
    -------
    Tuple[str, str, int]
        Start date, end date, forecast horizon
    """
    periods = {
        'pre_covid': ('2014-06-02', '2018-05-21', 5),
        'covid': ('2018-05-21', '2020-09-29', 10),
        'post_covid': ('2020-09-29', '2025-04-29', 5)
    }
    
    if period not in periods:
        raise ValueError(f"Period must be one of {list(periods.keys())}")
    
    return periods[period]


def create_rolling_windows(
    df: pd.DataFrame,
    train_window: int = 441,
    forecast_horizon: int = 5,
    step_size: int = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create rolling window splits for backtesting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    train_window : int
        Size of training window
    forecast_horizon : int
        Number of steps ahead to forecast
    step_size : int, optional
        Number of periods to roll forward (default: forecast_horizon)
        
    Returns
    -------
    List[Tuple[pd.DataFrame, pd.DataFrame]]
        List of (train, test) DataFrame tuples
    """
    if step_size is None:
        step_size = forecast_horizon
    
    windows = []
    n = len(df)
    
    start_idx = 0
    while start_idx + train_window + forecast_horizon <= n:
        train_end_idx = start_idx + train_window
        test_end_idx = min(train_end_idx + forecast_horizon, n)
        
        train_df = df.iloc[start_idx:train_end_idx].copy()
        test_df = df.iloc[train_end_idx:test_end_idx].copy()
        
        windows.append((train_df, test_df))
        
        start_idx += step_size
    
    return windows


def prepare_har_features(
    df: pd.DataFrame,
    include_vix: bool = True,
    include_kts_jmp: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix and target vector for HAR models.
    
    Parameters
    ----------
    df : pd.DataFrame
        Daily feature data
    include_vix : bool
        Whether to include VIX features
    include_kts_jmp : bool
        Whether to include KTS and JMP features
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        Feature matrix, target vector, feature names
    """
    feature_cols = ['RV', 'RV_w', 'RV_m']
    
    if include_vix:
        feature_cols.extend(['VIX', 'VIX_w', 'VIX_m'])
    
    if include_kts_jmp:
        feature_cols.extend(['KTS', 'JMP'])
    
    # Shift features by 1 to avoid look-ahead bias
    X = df[feature_cols].shift(1).values
    y = df['RV'].values
    
    # Remove first row with NaN
    X = X[1:]
    y = y[1:]
    
    return X, y, feature_cols


def save_results_to_markdown(
    results: Dict,
    output_path: str = "results/RESULTS.md"
) -> None:
    """
    Save experiment results to a markdown file.
    
    Parameters
    ----------
    results : dict
        Dictionary containing experiment results
    output_path : str
        Path to output markdown file
    """
    with open(output_path, 'w') as f:
        f.write("# Experiment Results\n\n")
        f.write("## Baseline HAR and Feature-Engineered HAR-VIX Models\n\n")
        
        for exp_name, exp_results in results.items():
            f.write(f"### {exp_name}\n\n")
            
            for period, period_results in exp_results.items():
                f.write(f"#### {period.replace('_', ' ').title()}\n\n")
                
                f.write("| Method | MSE (×10⁶) | MAPE (%) |\n")
                f.write("|--------|------------|----------|\n")
                
                for method, metrics in period_results.items():
                    mse = metrics.get('mse', np.nan)
                    mape = metrics.get('mape', np.nan)
                    f.write(f"| {method} | {mse:.2f} | {mape:.2f} |\n")
                
                f.write("\n")
            
            f.write("\n")
    
    print(f"Results saved to {output_path}")
