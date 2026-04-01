"""
Feature engineering module for realized volatility and related metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict


def compute_intraday_returns(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Compute 5-minute log returns for each trading day.
    
    Parameters
    ----------
    df : pd.DataFrame
        Intraday data with 'date' and price column
    price_col : str
        Name of the price column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'return' column
    """
    df = df.copy()
    df = df.sort_values(['date', 'timestamp'])
    
    # Compute log returns within each day
    df['return'] = df.groupby('date')[price_col].transform(
        lambda x: np.log(x / x.shift(1))
    )
    
    return df


def compute_daily_rv(
    df: pd.DataFrame,
    return_col: str = 'return',
    N: int = 78
) -> pd.DataFrame:
    """
    Compute daily Realized Volatility (RV).
    
    Formula: RV_t = sqrt((N/n_t) * sum(r_{t,i}^2))
    
    Parameters
    ----------
    df : pd.DataFrame
        Intraday data with returns
    return_col : str
        Name of the return column
    N : int
        Standard number of intraday intervals (default: 78)
        
    Returns
    -------
    pd.DataFrame
        Daily DataFrame with RV
    """
    def calc_rv(returns):
        returns = returns.dropna()
        n_t = len(returns)
        if n_t == 0:
            return np.nan
        sum_squared = (returns ** 2).sum()
        rv = np.sqrt((N / n_t) * sum_squared)
        return rv
    
    daily_rv = df.groupby('date')[return_col].apply(calc_rv).reset_index()
    daily_rv.columns = ['date', 'RV']
    
    return daily_rv


def compute_daily_bv(
    df: pd.DataFrame,
    return_col: str = 'return'
) -> pd.DataFrame:
    """
    Compute daily Bipower Variation (BV).
    
    Formula: BV_t = (pi/2) * sum(|r_{t,i}| * |r_{t,i-1}|)
    
    Parameters
    ----------
    df : pd.DataFrame
        Intraday data with returns
    return_col : str
        Name of the return column
        
    Returns
    -------
    pd.DataFrame
        Daily DataFrame with BV
    """
    def calc_bv(returns):
        returns = returns.dropna()
        if len(returns) < 2:
            return np.nan
        abs_returns = np.abs(returns.values)
        bv = (np.pi / 2) * np.sum(abs_returns[1:] * abs_returns[:-1])
        return bv
    
    daily_bv = df.groupby('date')[return_col].apply(calc_bv).reset_index()
    daily_bv.columns = ['date', 'BV']
    
    return daily_bv


def compute_daily_jmp(rv_df: pd.DataFrame, bv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily Jump Variation (JMP).
    
    Formula: JMP_t = max(0, RV_t - BV_t)
    
    Parameters
    ----------
    rv_df : pd.DataFrame
        Daily RV data
    bv_df : pd.DataFrame
        Daily BV data
        
    Returns
    -------
    pd.DataFrame
        Daily DataFrame with JMP
    """
    merged = rv_df.merge(bv_df, on='date', how='inner')
    merged['JMP'] = np.maximum(0, merged['RV'] - merged['BV'])
    
    return merged[['date', 'JMP']]


def compute_daily_kts(
    df: pd.DataFrame,
    return_col: str = 'return'
) -> pd.DataFrame:
    """
    Compute daily Realized Kurtosis (KTS).
    
    Formula: KTS_t = (n_t * sum(r_{t,i}^4)) / (sum(r_{t,i}^2))^2
    
    Parameters
    ----------
    df : pd.DataFrame
        Intraday data with returns
    return_col : str
        Name of the return column
        
    Returns
    -------
    pd.DataFrame
        Daily DataFrame with KTS
    """
    def calc_kts(returns):
        returns = returns.dropna()
        n_t = len(returns)
        if n_t == 0:
            return np.nan
        sum_fourth = (returns ** 4).sum()
        sum_squared = (returns ** 2).sum()
        if sum_squared == 0:
            return np.nan
        kts = (n_t * sum_fourth) / (sum_squared ** 2)
        return kts
    
    daily_kts = df.groupby('date')[return_col].apply(calc_kts).reset_index()
    daily_kts.columns = ['date', 'KTS']
    
    return daily_kts


def compute_lagged_averages(
    df: pd.DataFrame,
    columns: list,
    windows: Dict[str, int] = None
) -> pd.DataFrame:
    """
    Compute lagged moving averages for specified columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Daily data
    columns : list
        List of column names to compute averages for
    windows : dict, optional
        Dictionary mapping suffix to window size (default: {'_w': 5, '_m': 22})
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added lagged average columns
    """
    if windows is None:
        windows = {'_w': 5, '_m': 22}
    
    df = df.copy()
    df = df.sort_values('date')
    
    for col in columns:
        for suffix, window in windows.items():
            new_col = f"{col}{suffix}"
            df[new_col] = df[col].rolling(window=window, min_periods=1).mean()
    
    return df


def create_feature_dataframe(
    spx_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    N: int = 78
) -> pd.DataFrame:
    """
    Create complete feature DataFrame with all required features.
    
    Parameters
    ----------
    spx_df : pd.DataFrame
        SPX intraday data
    vix_df : pd.DataFrame
        VIX daily data
    N : int
        Standard number of intraday intervals
        
    Returns
    -------
    pd.DataFrame
        Daily DataFrame with all features: RV, BV, JMP, KTS, VIX, and their lags
    """
    # Compute intraday returns
    spx_df = compute_intraday_returns(spx_df)
    
    # Compute daily metrics
    rv_df = compute_daily_rv(spx_df, N=N)
    bv_df = compute_daily_bv(spx_df)
    jmp_df = compute_daily_jmp(rv_df, bv_df)
    kts_df = compute_daily_kts(spx_df)
    
    # Merge all daily metrics
    daily_df = rv_df.copy()
    for df in [bv_df, jmp_df, kts_df]:
        daily_df = daily_df.merge(df, on='date', how='left')
    
    # Merge VIX data
    vix_df = vix_df.rename(columns={'vix_close': 'VIX'})
    daily_df = daily_df.merge(vix_df[['date', 'VIX']], on='date', how='left')
    
    # Interpolate any missing values
    daily_df = daily_df.sort_values('date')
    for col in ['RV', 'BV', 'JMP', 'KTS', 'VIX']:
        daily_df[col] = daily_df[col].interpolate(method='linear')
    
    # Compute lagged averages for RV and VIX
    daily_df = compute_lagged_averages(daily_df, ['RV', 'VIX'])
    
    # Drop any remaining NaN rows
    daily_df = daily_df.dropna().reset_index(drop=True)
    
    print(f"Created feature DataFrame with {len(daily_df)} days")
    print(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print(f"Features: {list(daily_df.columns)}")
    
    return daily_df


def create_lagged_features(
    df: pd.DataFrame,
    target_col: str = 'RV',
    feature_cols: list = None,
    lag: int = 1
) -> pd.DataFrame:
    """
    Create lagged feature vectors for forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Daily feature data
    target_col : str
        Target variable column name
    feature_cols : list, optional
        List of feature columns to lag (if None, uses standard HAR-VIX features)
    lag : int
        Number of periods to lag features
        
    Returns
    -------
    pd.DataFrame
        DataFrame with lagged features and target
    """
    if feature_cols is None:
        feature_cols = ['RV', 'RV_w', 'RV_m', 'VIX', 'VIX_w', 'VIX_m', 'KTS', 'JMP']
    
    df = df.copy()
    df = df.sort_values('date')
    
    # Create lagged features
    for col in feature_cols:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    # Keep target at current time
    df['target'] = df[target_col]
    
    # Drop rows with NaN in lagged features
    df = df.dropna().reset_index(drop=True)
    
    return df
