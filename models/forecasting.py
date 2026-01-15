"""
Forecasting models for demand prediction.
Supports multiple model types with automatic selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing
from scipy import stats


class ModelType(Enum):
    AUTOMATIC = "automatic"
    SEASONAL_TREND = "seasonal_trend"
    LINEAR_TREND = "linear_trend"
    CONSTANT_LEVEL = "constant_level"
    INTERMITTENT = "intermittent"
    CROSTON = "croston"


@dataclass
class ForecastResult:
    """Container for forecast results and metadata."""
    item_code: str
    model_type: ModelType
    forecast: pd.Series
    fitted_values: pd.Series
    confidence_lower: pd.Series
    confidence_upper: pd.Series
    metrics: Dict[str, float]
    model_explanation: str


class DemandClassifier:
    """Classifies demand patterns to recommend appropriate forecast models."""

    @staticmethod
    def classify_demand(series: pd.Series) -> Dict:
        """
        Analyze demand pattern characteristics.
        Returns classification and recommended model.
        """
        # Remove any non-numeric values
        series = pd.to_numeric(series, errors='coerce').fillna(0)

        non_zero = series[series > 0]
        total_periods = len(series)
        non_zero_periods = len(non_zero)

        # Calculate metrics
        adi = total_periods / non_zero_periods if non_zero_periods > 0 else np.inf
        cv2 = (non_zero.std() / non_zero.mean()) ** 2 if non_zero_periods > 0 and non_zero.mean() > 0 else 0

        # Detect seasonality using autocorrelation
        has_seasonality = False
        seasonal_period = None
        if len(series) >= 24:
            try:
                acf_12 = series.autocorr(lag=12)
                if acf_12 > 0.2:  # Lowered threshold
                    has_seasonality = True
                    seasonal_period = 12
            except:
                pass

        # Also check monthly patterns for seasonality
        if not has_seasonality and len(series) >= 24:
            try:
                # Check if certain months consistently have higher demand
                if hasattr(series.index, 'month'):
                    monthly_avg = series.groupby(series.index.month).mean()
                    if monthly_avg.std() > monthly_avg.mean() * 0.3:
                        has_seasonality = True
                        seasonal_period = 12
            except:
                pass

        # Detect trend
        has_trend = False
        if len(series) >= 6:
            try:
                x = np.arange(len(series))
                slope, _, r_value, p_value, _ = stats.linregress(x, series.values)
                if p_value < 0.05 and abs(r_value) > 0.3:
                    has_trend = True
            except:
                pass

        # Classification based on ADI and CV2 (Syntetos-Boylan classification)
        if adi > 1.32:
            if cv2 > 0.49:
                pattern = "lumpy"
                recommended_model = ModelType.CROSTON
            else:
                pattern = "intermittent"
                recommended_model = ModelType.INTERMITTENT
        else:
            if cv2 > 0.49:
                pattern = "erratic"
                recommended_model = ModelType.CONSTANT_LEVEL
            else:
                pattern = "smooth"
                if has_seasonality and has_trend:
                    recommended_model = ModelType.SEASONAL_TREND
                elif has_trend:
                    recommended_model = ModelType.LINEAR_TREND
                elif has_seasonality:
                    recommended_model = ModelType.SEASONAL_TREND
                else:
                    recommended_model = ModelType.CONSTANT_LEVEL

        return {
            'pattern': pattern,
            'adi': adi,
            'cv2': cv2,
            'has_seasonality': has_seasonality,
            'seasonal_period': seasonal_period,
            'has_trend': has_trend,
            'non_zero_pct': non_zero_periods / total_periods * 100 if total_periods > 0 else 0,
            'recommended_model': recommended_model
        }

    @staticmethod
    def get_model_explanation(classification: Dict, model_type: ModelType) -> str:
        """Generate human-readable explanation of model selection."""
        explanations = {
            ModelType.SEASONAL_TREND: (
                "**Seasonal & Trend Model (Holt-Winters)**\n\n"
                "Best for: Items with consistent demand that shows both seasonal patterns "
                "(higher in certain months) and an overall trend (growing or declining).\n\n"
                "Why selected: This item shows {seasonality} and {trend}. "
                "The model captures both patterns to provide accurate forecasts."
            ),
            ModelType.LINEAR_TREND: (
                "**Linear Trend Model (Holt's Method)**\n\n"
                "Best for: Items with steady growth or decline but no strong seasonal pattern.\n\n"
                "Why selected: This item shows {trend} but no significant seasonal variation. "
                "The model focuses on capturing the direction of demand change."
            ),
            ModelType.CONSTANT_LEVEL: (
                "**Constant Level Model (Simple Exponential Smoothing)**\n\n"
                "Best for: Items with stable demand that fluctuates around a consistent average.\n\n"
                "Why selected: This item has relatively stable demand without strong trends "
                "or seasonal patterns. The model smooths out random variations."
            ),
            ModelType.INTERMITTENT: (
                "**Intermittent Demand Model (Seasonal Croston)**\n\n"
                "Best for: Slow-moving items with many zero-demand periods but consistent "
                "quantities when sales do occur.\n\n"
                "Why selected: This item has {non_zero_pct:.0f}% of periods with sales. "
                "The model handles sparse data while capturing seasonal patterns."
            ),
            ModelType.CROSTON: (
                "**Croston's Method with Seasonality**\n\n"
                "Best for: Slow-moving items with irregular demand timing AND variable quantities.\n\n"
                "Why selected: This item has sporadic sales ({non_zero_pct:.0f}% of periods) "
                "with variability in order sizes. This method forecasts both when demand "
                "occurs and how much, while accounting for seasonal patterns."
            )
        }

        template = explanations.get(model_type, "Model selected based on demand characteristics.")

        seasonality = "clear seasonal patterns" if classification['has_seasonality'] else "no seasonal pattern"
        trend = "an upward/downward trend" if classification['has_trend'] else "no significant trend"

        return template.format(
            seasonality=seasonality,
            trend=trend,
            non_zero_pct=classification['non_zero_pct']
        )


class Forecaster:
    """Main forecasting engine supporting multiple model types."""

    def __init__(self, forecast_periods: int = 36):
        self.forecast_periods = forecast_periods
        self.classifier = DemandClassifier()

    def forecast(
        self,
        series: pd.Series,
        item_code: str,
        model_type: ModelType = ModelType.AUTOMATIC,
        confidence_level: float = 0.95
    ) -> ForecastResult:
        """Generate forecast for a time series."""
        series = pd.to_numeric(series, errors='coerce').fillna(0)
        classification = self.classifier.classify_demand(series)

        if model_type == ModelType.AUTOMATIC:
            model_type = classification['recommended_model']

        # Generate forecast based on model type
        if model_type == ModelType.SEASONAL_TREND:
            result = self._seasonal_trend_forecast(series, classification)
        elif model_type == ModelType.LINEAR_TREND:
            result = self._linear_trend_forecast(series)
        elif model_type == ModelType.CONSTANT_LEVEL:
            result = self._constant_level_forecast(series)
        elif model_type in [ModelType.INTERMITTENT, ModelType.CROSTON]:
            result = self._seasonal_croston_forecast(series, classification)
        else:
            result = self._constant_level_forecast(series)

        forecast, fitted, lower, upper, metrics = result

        explanation = self.classifier.get_model_explanation(classification, model_type)

        return ForecastResult(
            item_code=item_code,
            model_type=model_type,
            forecast=forecast,
            fitted_values=fitted,
            confidence_lower=lower,
            confidence_upper=upper,
            metrics=metrics,
            model_explanation=explanation
        )

    def _calculate_seasonal_indices(self, series: pd.Series) -> np.ndarray:
        """Calculate monthly seasonal indices from historical data."""
        if not hasattr(series.index, 'month'):
            return np.ones(12)

        # Group by month and calculate average
        monthly_avg = series.groupby(series.index.month).mean()

        # Fill any missing months with overall average
        overall_avg = series.mean()
        if overall_avg == 0:
            return np.ones(12)

        indices = np.ones(12)
        for month in range(1, 13):
            if month in monthly_avg.index and monthly_avg[month] > 0:
                indices[month - 1] = monthly_avg[month] / overall_avg
            else:
                indices[month - 1] = 1.0

        # Normalize so average = 1
        indices = indices / indices.mean()

        return indices

    def _seasonal_trend_forecast(
        self,
        series: pd.Series,
        classification: Dict
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, Dict]:
        """Holt-Winters exponential smoothing with seasonality."""
        try:
            seasonal_period = classification.get('seasonal_period', 12) or 12

            if len(series) < seasonal_period * 2:
                return self._linear_trend_forecast(series)

            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_period,
                damped_trend=True
            ).fit(optimized=True)

            forecast = model.forecast(self.forecast_periods)
            fitted = model.fittedvalues

            residuals = series - fitted
            std_err = residuals.std()
            z = 1.96

            last_date = series.index[-1]
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=self.forecast_periods,
                freq='MS'
            )
            forecast.index = forecast_index

            # Ensure non-negative
            forecast = forecast.clip(lower=0)

            lower = (forecast - z * std_err).clip(lower=0)
            upper = forecast + z * std_err

            metrics = self._calculate_metrics(series, fitted)

            return forecast, fitted, lower, upper, metrics

        except Exception as e:
            return self._linear_trend_forecast(series)

    def _linear_trend_forecast(
        self,
        series: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, Dict]:
        """Holt's linear trend method."""
        try:
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=None,
                damped_trend=True
            ).fit(optimized=True)

            forecast = model.forecast(self.forecast_periods)
            fitted = model.fittedvalues

            residuals = series - fitted
            std_err = residuals.std()
            z = 1.96

            last_date = series.index[-1]
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=self.forecast_periods,
                freq='MS'
            )
            forecast.index = forecast_index

            forecast = forecast.clip(lower=0)
            lower = (forecast - z * std_err).clip(lower=0)
            upper = forecast + z * std_err

            metrics = self._calculate_metrics(series, fitted)

            return forecast, fitted, lower, upper, metrics

        except Exception:
            return self._constant_level_forecast(series)

    def _constant_level_forecast(
        self,
        series: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, Dict]:
        """Simple exponential smoothing for stable demand."""
        try:
            model = SimpleExpSmoothing(series).fit(optimized=True)

            forecast = model.forecast(self.forecast_periods)
            fitted = model.fittedvalues

            residuals = series - fitted
            std_err = residuals.std()
            z = 1.96

            last_date = series.index[-1]
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=self.forecast_periods,
                freq='MS'
            )
            forecast.index = forecast_index

            forecast = forecast.clip(lower=0)
            lower = (forecast - z * std_err).clip(lower=0)
            upper = forecast + z * std_err

            metrics = self._calculate_metrics(series, fitted)

            return forecast, fitted, lower, upper, metrics

        except Exception:
            mean_val = series.mean()
            last_date = series.index[-1]
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=self.forecast_periods,
                freq='MS'
            )

            forecast = pd.Series([mean_val] * self.forecast_periods, index=forecast_index)
            fitted = pd.Series([mean_val] * len(series), index=series.index)

            std_err = series.std()
            lower = (forecast - 1.96 * std_err).clip(lower=0)
            upper = forecast + 1.96 * std_err

            metrics = {'mae': 0, 'mape': 0, 'rmse': 0, 'bias': 0}

            return forecast, fitted, lower, upper, metrics

    def _seasonal_croston_forecast(
        self,
        series: pd.Series,
        classification: Dict
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, Dict]:
        """
        Croston's method with seasonal adjustment.
        Uses RECENT trend (last 12 months) as base, with seasonal indices.
        """
        demand = series.values
        n = len(demand)

        # Use recent 12 months as the base (matches competitor approach)
        recent_12 = series.tail(12)
        recent_avg = recent_12.mean()

        if recent_avg == 0:
            # No recent sales, use overall average
            recent_avg = series.mean()

        if recent_avg == 0:
            last_date = series.index[-1]
            forecast_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=self.forecast_periods,
                freq='MS'
            )
            forecast = pd.Series([0.0] * self.forecast_periods, index=forecast_index)
            fitted = pd.Series([0.0] * n, index=series.index)
            lower = forecast.copy()
            upper = forecast + 1
            return forecast, fitted, lower, upper, {'mae': 0, 'mape': 0, 'rmse': 0, 'bias': 0}

        # Calculate seasonal indices from recent data (last 24-36 months for stability)
        recent_period = series.tail(min(36, len(series)))
        seasonal_indices = self._calculate_seasonal_indices_recent(recent_period, recent_avg)

        # Create forecast
        last_date = series.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=self.forecast_periods,
            freq='MS'
        )

        # Apply seasonal indices to recent average
        forecast_values = []
        for date in forecast_index:
            month_idx = date.month - 1
            seasonal_factor = seasonal_indices[month_idx]
            # Base is recent average, adjusted by seasonal factor
            adjusted_forecast = recent_avg * seasonal_factor
            forecast_values.append(max(0, adjusted_forecast))

        forecast = pd.Series(forecast_values, index=forecast_index)

        # Fitted values
        fitted_values = []
        for date in series.index:
            if hasattr(date, 'month'):
                month_idx = date.month - 1
                seasonal_factor = seasonal_indices[month_idx]
                fitted_values.append(recent_avg * seasonal_factor)
            else:
                fitted_values.append(recent_avg)

        fitted = pd.Series(fitted_values, index=series.index)

        # Confidence intervals
        std_factor = 1.5
        lower = (forecast * 0.5).clip(lower=0)
        upper = forecast * 2.0

        metrics = self._calculate_metrics(series, fitted)

        return forecast, fitted, lower, upper, metrics

    def _calculate_seasonal_indices_recent(self, series: pd.Series, base_avg: float) -> np.ndarray:
        """Calculate seasonal indices from recent data, normalized to base average."""
        if not hasattr(series.index, 'month') or base_avg == 0:
            return np.ones(12)

        # Group by month
        monthly_avg = series.groupby(series.index.month).mean()

        indices = np.zeros(12)
        for month in range(1, 13):
            if month in monthly_avg.index:
                # Ratio of month's average to overall average
                indices[month - 1] = monthly_avg[month] / base_avg
            else:
                indices[month - 1] = 1.0

        # Normalize so the average index = 1
        if indices.mean() > 0:
            indices = indices / indices.mean()

        return indices

    def _calculate_metrics(self, actual: pd.Series, fitted: pd.Series) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        common_idx = actual.index.intersection(fitted.index)
        actual = actual.loc[common_idx]
        fitted = fitted.loc[common_idx]

        errors = actual - fitted
        abs_errors = errors.abs()

        mae = abs_errors.mean()
        rmse = np.sqrt((errors ** 2).mean())

        non_zero_actual = actual[actual > 0]
        if len(non_zero_actual) > 0:
            mape = (abs_errors[actual > 0] / non_zero_actual).mean() * 100
        else:
            mape = 0

        return {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'bias': errors.mean()
        }


def batch_forecast(
    monthly_data: pd.DataFrame,
    item_codes: List[str],
    forecast_periods: int = 36,
    model_type: ModelType = ModelType.AUTOMATIC
) -> Dict[str, ForecastResult]:
    """Generate forecasts for multiple items."""
    forecaster = Forecaster(forecast_periods=forecast_periods)
    results = {}

    for item_code in item_codes:
        item_data = monthly_data[monthly_data['item_code'] == item_code].copy()

        if len(item_data) < 3:
            continue

        item_data = item_data.sort_values('date')
        series = pd.Series(
            item_data['quantity_sold'].values,
            index=pd.DatetimeIndex(item_data['date'])
        )

        try:
            results[item_code] = forecaster.forecast(
                series=series,
                item_code=item_code,
                model_type=model_type
            )
        except Exception:
            continue

    return results
