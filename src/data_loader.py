"""
Data loading and preprocessing module for SPX and VIX data.
"""

import os
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from massive import RESTClient
from datetime import datetime, timedelta


def fetch_spx_intraday_data(
    start_date: str = "2014-05-01",
    end_date: str = "2025-05-27",
    timespan: str = "minute",
    multiplier: int = 5,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch 5-minute intraday data for S&P 500 Index (SPX).
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    timespan : str
        Timespan for aggregates (default: "minute")
    multiplier : int
        Multiplier for timespan (default: 5 for 5-minute bars)
    api_key : str, optional
        Massive API key. If None, reads from MASSIVE_TOKEN environment variable
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if api_key is None:
        api_key = os.getenv("MASSIVE_TOKEN")
    
    if not api_key:
        raise ValueError("API key must be provided or set in MASSIVE_TOKEN environment variable")
    
    client = RESTClient(api_key=api_key)
    ticker = "SPX"
    
    print(f"Fetching {ticker} intraday data from {start_date} to {end_date}...")
    
    aggs = []
    try:
        for agg in client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_date,
            to=end_date,
            limit=50000
        ):
            aggs.append(agg)
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise
    
    if not aggs:
        raise ValueError(f"No data returned for {ticker}")
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'timestamp': pd.to_datetime(a.get('t', a.get('timestamp')), unit='ms'),
        'open': a.get('o', a.get('open')),
        'high': a.get('h', a.get('high')),
        'low': a.get('l', a.get('low')),
        'close': a.get('c', a.get('close')),
        'volume': a.get('v', a.get('volume'))
    } for a in aggs])
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Fetched {len(df)} intraday bars for {ticker}")
    
    return df


def fetch_vix_daily_data(
    start_date: str = "2014-05-01",
    end_date: str = "2025-05-27",
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch daily VIX data.
    
    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    api_key : str, optional
        Massive API key. If None, reads from MASSIVE_TOKEN environment variable
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, vix_close
    """
    if api_key is None:
        api_key = os.getenv("MASSIVE_TOKEN")
    
    if not api_key:
        raise ValueError("API key must be provided or set in MASSIVE_TOKEN environment variable")
    
    client = RESTClient(api_key=api_key)
    ticker = "VIX"
    
    print(f"Fetching {ticker} daily data from {start_date} to {end_date}...")
    
    aggs = []
    try:
        for agg in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date,
            limit=50000
        ):
            aggs.append(agg)
    except Exception as e:
        print(f"Error fetching VIX data: {e}")
        raise
    
    if not aggs:
        raise ValueError(f"No data returned for {ticker}")
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'date': pd.to_datetime(a.get('t', a.get('timestamp')), unit='ms').date(),
        'vix_close': a.get('c', a.get('close'))
    } for a in aggs])
    
    df = df.sort_values('date').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Fetched {len(df)} daily bars for {ticker}")
    
    return df


def load_or_fetch_data(
    cache_dir: str = "data",
    force_refresh: bool = False,
    api_key: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from cache or fetch from API if not available.
    
    Parameters
    ----------
    cache_dir : str
        Directory to cache data files
    force_refresh : bool
        If True, fetch data even if cache exists
    api_key : str, optional
        Massive API key
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        SPX intraday data and VIX daily data
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    spx_cache = os.path.join(cache_dir, "spx_intraday.parquet")
    vix_cache = os.path.join(cache_dir, "vix_daily.parquet")
    
    # Load or fetch SPX data
    if os.path.exists(spx_cache) and not force_refresh:
        print(f"Loading SPX data from cache: {spx_cache}")
        spx_df = pd.read_parquet(spx_cache)
    else:
        spx_df = fetch_spx_intraday_data(api_key=api_key)
        spx_df.to_parquet(spx_cache, index=False)
        print(f"Cached SPX data to: {spx_cache}")
    
    # Load or fetch VIX data
    if os.path.exists(vix_cache) and not force_refresh:
        print(f"Loading VIX data from cache: {vix_cache}")
        vix_df = pd.read_parquet(vix_cache)
    else:
        vix_df = fetch_vix_daily_data(api_key=api_key)
        vix_df.to_parquet(vix_cache, index=False)
        print(f"Cached VIX data to: {vix_cache}")
    
    return spx_df, vix_df


def clean_and_align_data(
    spx_df: pd.DataFrame,
    vix_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and align SPX and VIX data to common trading days.
    
    Parameters
    ----------
    spx_df : pd.DataFrame
        SPX intraday data
    vix_df : pd.DataFrame
        VIX daily data
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Cleaned and aligned SPX and VIX data
    """
    # Extract date from SPX timestamps
    spx_df = spx_df.copy()
    spx_df['date'] = pd.to_datetime(spx_df['timestamp']).dt.date
    spx_df['date'] = pd.to_datetime(spx_df['date'])
    
    # Get unique trading days from SPX
    spx_trading_days = spx_df['date'].unique()
    
    # Filter VIX to only include SPX trading days
    vix_df = vix_df[vix_df['date'].isin(spx_trading_days)].copy()
    
    # Interpolate missing VIX values
    vix_df = vix_df.set_index('date').sort_index()
    vix_df['vix_close'] = vix_df['vix_close'].interpolate(method='linear')
    vix_df = vix_df.reset_index()
    
    print(f"Aligned data: {len(spx_trading_days)} trading days")
    
    return spx_df, vix_df
