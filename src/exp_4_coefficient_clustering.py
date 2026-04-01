"""
Experiment 4: Coefficient-Based Soft Clustering HAR-VIX Model

This module implements a sophisticated regime-aware model that:
1. Segments time series using Mood's Median Test
2. Fits HAR-VIX models on each segment and extracts coefficients
3. Clusters coefficients using PCA + Bayesian GMM
4. Trains XGBoost to predict regime probabilities
5. Makes weighted ensemble forecasts
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def moods_median_test_segmentation(
    series: np.ndarray,
    half_window: int = 40,
    alpha: float = 0.05
) -> List[int]:
    """
    Apply Mood's Median Test to detect variance change points.
    
    Parameters
    ----------
    series : np.ndarray
        Time series data
    half_window : int
        Half-window size for the test
    alpha : float
        Significance level
        
    Returns
    -------
    List[int]
        List of change point indices
    """
    n = len(series)
    change_points = [0]  # Start with first index
    
    for t in range(half_window, n - half_window):
        # Get windows before and after time t
        before = series[t - half_window:t]
        after = series[t:t + half_window]
        
        # Compute squared deviations from median
        combined = np.concatenate([before, after])
        median = np.median(combined)
        
        before_sq = (before - median) ** 2
        after_sq = (after - median) ** 2
        
        # Perform Mann-Whitney U test on squared deviations
        try:
            statistic, p_value = stats.mannwhitneyu(before_sq, after_sq, alternative='two-sided')
            
            if p_value < alpha:
                # Significant change detected
                if len(change_points) == 0 or t - change_points[-1] > half_window:
                    change_points.append(t)
        except:
            continue
    
    change_points.append(n)  # End with last index
    
    return change_points


def create_segments_from_changepoints(
    df: pd.DataFrame,
    change_points: List[int],
    min_segment_size: int = 30
) -> List[pd.DataFrame]:
    """
    Create data segments from change points.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    change_points : List[int]
        List of change point indices
    min_segment_size : int
        Minimum size for a segment
        
    Returns
    -------
    List[pd.DataFrame]
        List of segment DataFrames
    """
    segments = []
    
    for i in range(len(change_points) - 1):
        start_idx = change_points[i]
        end_idx = change_points[i + 1]
        
        if end_idx - start_idx >= min_segment_size:
            segment = df.iloc[start_idx:end_idx].copy()
            segments.append(segment)
    
    return segments


def fit_har_vix_on_segment(
    segment: pd.DataFrame,
    feature_cols: List[str]
) -> np.ndarray:
    """
    Fit HAR-VIX model on a segment and return coefficient vector.
    
    Parameters
    ----------
    segment : pd.DataFrame
        Segment data
    feature_cols : List[str]
        List of feature column names
        
    Returns
    -------
    np.ndarray
        Coefficient vector (intercept + slopes)
    """
    # Prepare features and target
    X = segment[feature_cols].shift(1).values[1:]
    y = segment['RV'].values[1:]
    
    if len(X) < len(feature_cols) + 1:
        # Not enough data, use Ridge with small penalty
        model = Ridge(alpha=1e-4)
    else:
        model = LinearRegression()
    
    try:
        model.fit(X, y)
        # Return intercept + coefficients
        coef_vector = np.concatenate([[model.intercept_], model.coef_])
    except:
        # Fallback: return zeros
        coef_vector = np.zeros(len(feature_cols) + 1)
    
    return coef_vector


def cluster_coefficients_bgmm(
    coefficient_matrix: np.ndarray,
    n_components: int = 2,
    variance_threshold: float = 0.95
) -> Tuple[np.ndarray, PCA, BayesianGaussianMixture]:
    """
    Cluster coefficient vectors using PCA + Bayesian GMM.
    
    Parameters
    ----------
    coefficient_matrix : np.ndarray
        Matrix of coefficient vectors (n_segments × n_features)
    n_components : int
        Number of mixture components
    variance_threshold : float
        Variance explained threshold for PCA
        
    Returns
    -------
    Tuple[np.ndarray, PCA, BayesianGaussianMixture]
        Soft membership probabilities, fitted PCA, fitted BGMM
    """
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=variance_threshold, svd_solver='full')
    coef_reduced = pca.fit_transform(coefficient_matrix)
    
    print(f"PCA: {coefficient_matrix.shape[1]} → {coef_reduced.shape[1]} dimensions")
    
    # Fit Bayesian GMM
    bgmm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type='full',
        max_iter=200,
        random_state=42
    )
    bgmm.fit(coef_reduced)
    
    # Get soft membership probabilities
    soft_memberships = bgmm.predict_proba(coef_reduced)
    
    return soft_memberships, pca, bgmm


def train_xgboost_regime_classifier(
    segment_features: np.ndarray,
    soft_memberships: np.ndarray,
    n_regimes: int = 2
) -> xgb.XGBClassifier:
    """
    Train XGBoost model to predict regime probabilities.
    
    Parameters
    ----------
    segment_features : np.ndarray
        Mean feature vectors for each segment
    soft_memberships : np.ndarray
        Soft membership probabilities from BGMM
    n_regimes : int
        Number of regimes
        
    Returns
    -------
    xgb.XGBClassifier
        Trained XGBoost classifier
    """
    # Convert soft memberships to hard labels for training
    hard_labels = np.argmax(soft_memberships, axis=1)
    
    # Train XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=n_regimes,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(segment_features, hard_labels)
    
    return xgb_model


class CoefficientClusteringModel:
    """
    Coefficient-based soft clustering HAR-VIX model.
    """
    
    def __init__(
        self,
        n_regimes: int = 2,
        half_window: int = 40,
        alpha: float = 0.05
    ):
        """
        Initialize coefficient clustering model.
        
        Parameters
        ----------
        n_regimes : int
            Number of regimes
        half_window : int
            Half-window for Mood's test
        alpha : float
            Significance level for Mood's test
        """
        self.n_regimes = n_regimes
        self.half_window = half_window
        self.alpha = alpha
        
        self.feature_cols = ['RV', 'RV_w', 'RV_m', 'VIX', 'VIX_w', 'VIX_m', 'KTS', 'JMP']
        self.regime_models = []
        self.xgb_model = None
        self.feature_scaler = None
        self.target_scaler = None
        
    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fit the coefficient clustering model.
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        """
        print(f"Fitting coefficient clustering model on {len(train_df)} days...")
        
        # Step 1: Segmentation using Mood's test
        rv_series = train_df['RV'].values
        change_points = moods_median_test_segmentation(
            rv_series,
            half_window=self.half_window,
            alpha=self.alpha
        )
        
        segments = create_segments_from_changepoints(train_df, change_points)
        print(f"Created {len(segments)} segments")
        
        if len(segments) < self.n_regimes:
            print(f"Warning: Only {len(segments)} segments, less than {self.n_regimes} regimes")
            # Fallback: use simple time-based segmentation
            segment_size = len(train_df) // self.n_regimes
            segments = [
                train_df.iloc[i*segment_size:(i+1)*segment_size]
                for i in range(self.n_regimes)
            ]
        
        # Step 2: Fit HAR-VIX on each segment and extract coefficients
        coefficient_matrix = []
        segment_mean_features = []
        
        for segment in segments:
            coef_vector = fit_har_vix_on_segment(segment, self.feature_cols)
            coefficient_matrix.append(coef_vector)
            
            # Compute mean features for this segment
            mean_features = segment[self.feature_cols].mean().values
            segment_mean_features.append(mean_features)
        
        coefficient_matrix = np.array(coefficient_matrix)
        segment_mean_features = np.array(segment_mean_features)
        
        # Step 3: Cluster coefficients using PCA + BGMM
        soft_memberships, pca, bgmm = cluster_coefficients_bgmm(
            coefficient_matrix,
            n_components=self.n_regimes
        )
        
        print(f"Soft memberships shape: {soft_memberships.shape}")
        
        # Step 4: Assign point-level weights
        point_weights = np.zeros((len(train_df), self.n_regimes))
        
        for seg_idx, segment in enumerate(segments):
            start_idx = segment.index[0]
            end_idx = segment.index[-1] + 1
            
            # Find corresponding indices in train_df
            mask = (train_df.index >= start_idx) & (train_df.index < end_idx)
            point_weights[mask] = soft_memberships[seg_idx]
        
        # Step 5: Fit regime-specific WLS models
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        X = train_df[self.feature_cols].shift(1).values[1:]
        y = train_df['RV'].values[1:]
        
        X_norm = self.feature_scaler.fit_transform(X)
        y_norm = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Adjust weights for shifted data
        point_weights = point_weights[1:]
        
        self.regime_models = []
        for k in range(self.n_regimes):
            model = LinearRegression()
            
            # Weighted least squares
            weights = point_weights[:, k]
            weights = weights / (weights.sum() + 1e-10)  # Normalize
            
            # Fit with sample weights
            model.fit(X_norm, y_norm, sample_weight=weights)
            self.regime_models.append(model)
        
        # Step 6: Train XGBoost for regime probability prediction
        self.xgb_model = train_xgboost_regime_classifier(
            segment_mean_features,
            soft_memberships,
            n_regimes=self.n_regimes
        )
        
        print("Model fitting complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make weighted ensemble predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        # Get regime probabilities from XGBoost
        regime_probs = self.xgb_model.predict_proba(X)
        
        # Normalize features
        X_norm = self.feature_scaler.transform(X)
        
        # Get predictions from each regime model
        predictions = np.zeros(len(X))
        
        for i in range(len(X)):
            weighted_pred = 0.0
            
            for k in range(self.n_regimes):
                regime_pred_norm = self.regime_models[k].predict(X_norm[i:i+1])[0]
                regime_pred = self.target_scaler.inverse_transform([[regime_pred_norm]])[0, 0]
                weighted_pred += regime_probs[i, k] * regime_pred
            
            predictions[i] = weighted_pred
        
        return predictions
    
    def forecast_non_recursive(
        self,
        test_df: pd.DataFrame,
        horizon: int
    ) -> np.ndarray:
        """
        Non-recursive h-step ahead forecasting.
        
        Parameters
        ----------
        test_df : pd.DataFrame
            Test data
        horizon : int
            Forecast horizon
            
        Returns
        -------
        np.ndarray
            Forecasts
        """
        forecasts = []
        
        for i in range(len(test_df)):
            if i < horizon:
                continue
            
            X = test_df[self.feature_cols].iloc[i-1:i].values
            y_pred = self.predict(X)
            forecasts.append(y_pred[0])
        
        return np.array(forecasts)


def run_rolling_backtest_exp4(
    df: pd.DataFrame,
    period: str,
    n_regimes: int = 2,
    train_window: int = 441
) -> Dict[str, float]:
    """
    Run rolling window backtest for Experiment 4.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    period : str
        One of 'pre_covid', 'covid', 'post_covid'
    n_regimes : int
        Number of regimes
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
    
    print(f"\nRunning Coefficient Clustering (K={n_regimes}) for {period}")
    print(f"Period: {start_date} to {end_date}, Horizon: {horizon}")
    print(f"Total days: {len(period_df)}")
    
    # Storage for predictions
    all_predictions = []
    all_actuals = []
    
    # Rolling window
    step_size = horizon
    start_idx = 0
    
    while start_idx + train_window + horizon <= len(period_df):
        train_end_idx = start_idx + train_window
        test_idx = train_end_idx + horizon - 1
        
        train_df = period_df.iloc[start_idx:train_end_idx].copy()
        
        # Fit model
        model = CoefficientClusteringModel(n_regimes=n_regimes)
        
        try:
            model.fit(train_df)
            
            # Make forecast
            if test_idx < len(period_df):
                X = period_df[model.feature_cols].iloc[test_idx:test_idx+1].values
                y_pred = model.predict(X)[0]
                y_actual = period_df['RV'].iloc[test_idx]
                
                all_predictions.append(y_pred)
                all_actuals.append(y_actual)
        except Exception as e:
            print(f"Error in window {start_idx}: {e}")
        
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


def run_experiment_4(df: pd.DataFrame) -> Dict:
    """
    Run complete Experiment 4.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataset
        
    Returns
    -------
    Dict
        Results for all periods
    """
    results = {'Coefficient_Clustering_K2': {}}
    
    periods = ['pre_covid', 'covid', 'post_covid']
    
    print("\n" + "="*80)
    print("COEFFICIENT-BASED SOFT CLUSTERING MODEL (K=2)")
    print("="*80)
    
    for period in periods:
        metrics = run_rolling_backtest_exp4(df, period, n_regimes=2)
        results['Coefficient_Clustering_K2'][period] = {'non_recursive': metrics}
    
    return results
