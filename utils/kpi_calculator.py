"""
KPI calculation module for inventory management metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class InventoryKPIs:
    """Container for all inventory KPIs."""
    # Value metrics
    inventory_value: float
    overstock_value: float
    shortage_value: float
    non_moving_value: float

    # Turnover metrics
    turnover_ratio: float  # turns per year
    days_to_sell: float

    # Revenue metrics
    annual_revenue: float
    projected_revenue: float
    gross_margin: float
    gross_margin_pct: float

    # Sales metrics
    annual_sales_qty: float
    projected_sales_qty: float

    # Efficiency metrics
    fill_rate: float
    stockout_rate: float


class KPICalculator:
    """Calculate inventory and demand KPIs."""

    def __init__(self, forecast_months: int = 12):
        self.forecast_months = forecast_months

    def calculate_item_kpis(
        self,
        item_data: Dict,
        historical_sales: pd.DataFrame,
        forecast: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate KPIs for a single item.

        Args:
            item_data: Dictionary with item info (on_hand, cost, price, lead_time)
            historical_sales: DataFrame with date, quantity_sold, revenue
            forecast: Forecasted demand series

        Returns:
            Dictionary of KPI values
        """
        on_hand = item_data.get('last_on_hand', 0)
        unit_cost = item_data.get('inventory_value_unit', 0)
        lead_time = item_data.get('lead_time', 7)

        # Historical metrics (last 12 months)
        recent_sales = historical_sales[
            historical_sales['date'] >= historical_sales['date'].max() - pd.DateOffset(months=12)
        ]

        annual_qty = recent_sales['quantity_sold'].sum()
        annual_revenue = recent_sales['transaction_revenue'].sum()
        avg_monthly_qty = annual_qty / 12 if len(recent_sales) > 0 else 0
        avg_monthly_revenue = annual_revenue / 12 if len(recent_sales) > 0 else 0

        # Inventory value
        inventory_value = on_hand * unit_cost

        # Turnover calculations
        cogs = annual_qty * unit_cost
        avg_inventory_value = inventory_value  # Simplified - could use average

        if avg_inventory_value > 0:
            turnover_ratio = cogs / avg_inventory_value
            days_to_sell = 365 / turnover_ratio if turnover_ratio > 0 else 999
        else:
            turnover_ratio = 0
            days_to_sell = 999

        # Days of supply
        daily_demand = avg_monthly_qty / 30 if avg_monthly_qty > 0 else 0
        days_of_supply = on_hand / daily_demand if daily_demand > 0 else 999

        # Forecast-based metrics
        if forecast is not None and len(forecast) > 0:
            projected_annual_qty = forecast.sum() * (12 / len(forecast))
            projected_annual_revenue = projected_annual_qty * (
                annual_revenue / annual_qty if annual_qty > 0 else 0
            )
        else:
            projected_annual_qty = annual_qty
            projected_annual_revenue = annual_revenue

        # Gross margin (estimated)
        avg_price = annual_revenue / annual_qty if annual_qty > 0 else 0
        gross_margin_per_unit = avg_price - unit_cost
        gross_margin_pct = (gross_margin_per_unit / avg_price * 100) if avg_price > 0 else 0

        # Safety stock and reorder point
        # Using simple method: safety stock = lead_time_demand * safety_factor
        safety_factor = 1.5  # Covers ~93% service level
        lead_time_demand = daily_demand * lead_time
        safety_stock = lead_time_demand * safety_factor
        reorder_point = lead_time_demand + safety_stock

        # Overstock/Shortage
        max_stock = avg_monthly_qty * 3  # 3 months of stock as max
        overstock_qty = max(0, on_hand - max_stock)
        overstock_value = overstock_qty * unit_cost

        shortage_qty = max(0, reorder_point - on_hand)
        shortage_value = shortage_qty * unit_cost

        # Non-moving check (no sales in last 6 months)
        last_6_months = historical_sales[
            historical_sales['date'] >= historical_sales['date'].max() - pd.DateOffset(months=6)
        ]
        is_non_moving = last_6_months['quantity_sold'].sum() == 0
        non_moving_value = inventory_value if is_non_moving else 0

        return {
            'inventory_value': inventory_value,
            'overstock_value': overstock_value,
            'shortage_value': shortage_value,
            'non_moving_value': non_moving_value,
            'turnover_ratio': turnover_ratio,
            'days_to_sell': days_to_sell,
            'days_of_supply': days_of_supply,
            'annual_revenue': annual_revenue,
            'projected_revenue': projected_annual_revenue,
            'gross_margin': gross_margin_per_unit * annual_qty,
            'gross_margin_pct': gross_margin_pct,
            'annual_sales_qty': annual_qty,
            'projected_sales_qty': projected_annual_qty,
            'avg_monthly_qty': avg_monthly_qty,
            'avg_monthly_revenue': avg_monthly_revenue,
            'safety_stock': safety_stock,
            'reorder_point': reorder_point,
            'max_stock': max_stock,
            'overstock_qty': overstock_qty,
            'shortage_qty': shortage_qty,
            'is_non_moving': is_non_moving,
            'on_hand': on_hand,
            'unit_cost': unit_cost,
            'lead_time': lead_time
        }

    def calculate_aggregate_kpis(
        self,
        item_kpis: Dict[str, Dict],
        filter_abc: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate aggregate KPIs across all items or filtered subset.

        Args:
            item_kpis: Dictionary mapping item_code to KPI dict
            filter_abc: Optional ABC class filter ('A', 'B', 'C')

        Returns:
            Aggregated KPIs
        """
        if not item_kpis:
            return self._empty_kpis()

        # Aggregate sums
        total_inventory_value = sum(k['inventory_value'] for k in item_kpis.values())
        total_overstock_value = sum(k['overstock_value'] for k in item_kpis.values())
        total_shortage_value = sum(k['shortage_value'] for k in item_kpis.values())
        total_non_moving_value = sum(k['non_moving_value'] for k in item_kpis.values())
        total_annual_revenue = sum(k['annual_revenue'] for k in item_kpis.values())
        total_projected_revenue = sum(k['projected_revenue'] for k in item_kpis.values())
        total_gross_margin = sum(k['gross_margin'] for k in item_kpis.values())
        total_annual_qty = sum(k['annual_sales_qty'] for k in item_kpis.values())
        total_projected_qty = sum(k['projected_sales_qty'] for k in item_kpis.values())

        # Weighted averages
        total_cogs = sum(k['annual_sales_qty'] * k['unit_cost'] for k in item_kpis.values())

        # Aggregate turnover
        if total_inventory_value > 0:
            agg_turnover = total_cogs / total_inventory_value
            agg_days_to_sell = 365 / agg_turnover if agg_turnover > 0 else 999
        else:
            agg_turnover = 0
            agg_days_to_sell = 999

        # Gross margin percentage
        gross_margin_pct = (total_gross_margin / total_annual_revenue * 100) if total_annual_revenue > 0 else 0

        # Item counts
        total_items = len(item_kpis)
        overstock_items = sum(1 for k in item_kpis.values() if k['overstock_qty'] > 0)
        shortage_items = sum(1 for k in item_kpis.values() if k['shortage_qty'] > 0)
        non_moving_items = sum(1 for k in item_kpis.values() if k['is_non_moving'])

        return {
            'inventory_value': total_inventory_value,
            'overstock_value': total_overstock_value,
            'shortage_value': total_shortage_value,
            'non_moving_value': total_non_moving_value,
            'turnover_ratio': agg_turnover,
            'days_to_sell': agg_days_to_sell,
            'annual_revenue': total_annual_revenue,
            'projected_revenue': total_projected_revenue,
            'gross_margin': total_gross_margin,
            'gross_margin_pct': gross_margin_pct,
            'annual_sales_qty': total_annual_qty,
            'projected_sales_qty': total_projected_qty,
            'total_items': total_items,
            'overstock_items': overstock_items,
            'shortage_items': shortage_items,
            'non_moving_items': non_moving_items
        }

    def _empty_kpis(self) -> Dict[str, float]:
        """Return empty KPI dictionary."""
        return {
            'inventory_value': 0,
            'overstock_value': 0,
            'shortage_value': 0,
            'non_moving_value': 0,
            'turnover_ratio': 0,
            'days_to_sell': 0,
            'annual_revenue': 0,
            'projected_revenue': 0,
            'gross_margin': 0,
            'gross_margin_pct': 0,
            'annual_sales_qty': 0,
            'projected_sales_qty': 0,
            'total_items': 0,
            'overstock_items': 0,
            'shortage_items': 0,
            'non_moving_items': 0
        }


def format_currency(value: float) -> str:
    """Format number as currency."""
    if abs(value) >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.1f}K"
    else:
        return f"${value:.2f}"


def format_number(value: float) -> str:
    """Format large numbers with K/M suffixes."""
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"{value/1_000:.1f}K"
    else:
        return f"{value:.0f}"
