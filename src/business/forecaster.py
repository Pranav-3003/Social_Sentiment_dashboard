import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger("Forecaster")

# Conditional import of statsmodels
STATSMODELS_AVAILABLE = False
try:
    from statsmodels.tsa.api import Holt, ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    logger.warning("statsmodels library is not available. Using simple regression/moving average fallback for forecasting.")

class SentimentForecaster:
    def __init__(self):
        pass

    def forecast_sentiment(self, history_df: pd.DataFrame, 
                           target_col: str = 'average_polarity', 
                           steps: int = 7) -> Tuple[pd.DataFrame, str]:
        """
        Forecasts future values of a target column (average polarity or volume) over next N steps.
        Returns: (forecast_df, model_description_string).
        """
        if len(history_df) < 3:
            return self._generate_flat_fallback(history_df, target_col, steps, "Insufficient data for forecasting (< 3 periods). flat baseline used.")
            
        # Ensure index is sorted by date
        df_sorted = history_df.sort_values(by='timestamp').reset_index(drop=True)
        y = df_sorted[target_col].values
        dates = pd.to_datetime(df_sorted['timestamp'])
        
        last_date = dates.iloc[-1]
        freq_diff = dates.iloc[-1] - dates.iloc[-2] if len(dates) > 1 else timedelta(days=1)
        
        future_dates = [last_date + (freq_diff * (i + 1)) for i in range(steps)]
        
        # 1. Statsmodels Holt-Winters / ARIMA
        if STATSMODELS_AVAILABLE:
            try:
                # Use Holt's Linear Trend method (great for short timelines)
                # We fit on values
                model = Holt(y, initialization_method="estimated").fit(smoothing_level=0.6, smoothing_trend=0.2)
                forecast_vals = model.forecast(steps)
                
                # Simple confidence interval heuristic (standard error propagation)
                residuals = y - model.fittedvalues
                std_err = np.std(residuals) if len(residuals) > 0 else 0.1
                
                lower_bounds = []
                upper_bounds = []
                for i in range(steps):
                    # Error margin grows with step index
                    margin = 1.96 * std_err * np.sqrt(i + 1)
                    lower_bounds.append(forecast_vals[i] - margin)
                    upper_bounds.append(forecast_vals[i] + margin)
                    
                forecast_df = pd.DataFrame({
                    "timestamp": future_dates,
                    "forecast": forecast_vals,
                    "lower_bound": lower_bounds,
                    "upper_bound": upper_bounds
                })
                
                # Clip bounds for polarity between -1 and 1
                if target_col == 'average_polarity':
                    forecast_df['forecast'] = forecast_df['forecast'].clip(-1.0, 1.0)
                    forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(-1.0, 1.0)
                    forecast_df['upper_bound'] = forecast_df['upper_bound'].clip(-1.0, 1.0)
                    
                return forecast_df, f"Holt's Linear Exponential Smoothing (Statsmodels)"
            except Exception as e:
                logger.error(f"Statsmodels forecasting failed: {e}. Falling back to linear trend.")
                
        # 2. Linear Regression / Moving Average Fallback
        return self._generate_linear_fallback(df_sorted, target_col, future_dates, steps)

    def _generate_flat_fallback(self, history_df: pd.DataFrame, target_col: str, steps: int, desc: str) -> Tuple[pd.DataFrame, str]:
        # Simple flat baseline forecast based on last value
        val = history_df[target_col].iloc[-1] if len(history_df) > 0 else 0.0
        last_date = pd.to_datetime(history_df['timestamp'].iloc[-1]) if len(history_df) > 0 else datetime.utcnow()
        future_dates = [last_date + timedelta(days=i+1) for i in range(steps)]
        
        forecast_df = pd.DataFrame({
            "timestamp": future_dates,
            "forecast": [val] * steps,
            "lower_bound": [val - 0.2] * steps,
            "upper_bound": [val + 0.2] * steps
        })
        return forecast_df, desc

    def _generate_linear_fallback(self, df_sorted: pd.DataFrame, target_col: str, future_dates: list, steps: int) -> Tuple[pd.DataFrame, str]:
        y = df_sorted[target_col].values
        x = np.arange(len(y))
        
        # Fit linear trend line: y = m*x + c
        m, c = np.polyfit(x, y, 1)
        
        forecast_vals = []
        lower_bounds = []
        upper_bounds = []
        
        residuals = y - (m * x + c)
        std_err = np.std(residuals) if len(residuals) > 0 else 0.1
        
        for i in range(steps):
            next_x = len(y) + i
            pred = (m * next_x) + c
            margin = 1.96 * std_err * np.sqrt(i + 1)
            
            forecast_vals.append(pred)
            lower_bounds.append(pred - margin)
            upper_bounds.append(pred + margin)
            
        forecast_df = pd.DataFrame({
            "timestamp": future_dates,
            "forecast": forecast_vals,
            "lower_bound": lower_bounds,
            "upper_bound": upper_bounds
        })
        
        if target_col == 'average_polarity':
            forecast_df['forecast'] = forecast_df['forecast'].clip(-1.0, 1.0)
            forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(-1.0, 1.0)
            forecast_df['upper_bound'] = forecast_df['upper_bound'].clip(-1.0, 1.0)
            
        return forecast_df, "Linear Trend Regression"
