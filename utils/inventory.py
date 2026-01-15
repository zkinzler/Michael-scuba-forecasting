"""
Inventory management calculations module.
Handles safety stock, reorder points, and inventory projections.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ReplenishmentStrategy(Enum):
    PERIODIC = "periodic"  # Order at fixed intervals
    MIN_MAX = "min_max"    # Order when stock hits min, order up to max


@dataclass
class SafetyStockMatrix:
    """Safety stock multipliers based on ABC-XYZ classification."""
    # ABC = revenue importance, XYZ = demand variability
    matrix: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        if self.matrix is None:
            # Default safety stock days by ABC-XYZ
            self.matrix = {
                'A': {'X': 1.0, 'Y': 1.5, 'Z': 2.0},  # High revenue items
                'B': {'X': 1.2, 'Y': 1.8, 'Z': 2.5},  # Medium revenue items
                'C': {'X': 1.5, 'Y': 2.0, 'Z': 3.0},  # Low revenue items
            }

    def get_multiplier(self, abc_class: str, xyz_class: str) -> float:
        """Get safety stock multiplier for given classification."""
        return self.matrix.get(abc_class, {}).get(xyz_class, 1.5)


class InventoryCalculator:
    """Calculate inventory levels, projections, and ordering recommendations."""

    def __init__(
        self,
        default_lead_time: int = 7,
        service_level: float = 0.95,
        review_period: int = 30,  # days between orders
        safety_stock_matrix: Optional[SafetyStockMatrix] = None
    ):
        self.default_lead_time = default_lead_time
        self.service_level = service_level
        self.review_period = review_period
        self.safety_stock_matrix = safety_stock_matrix or SafetyStockMatrix()

        # Z-score for service level
        from scipy import stats
        self.z_score = stats.norm.ppf(service_level)

    def calculate_xyz_class(self, cv: float) -> str:
        """
        Classify item by demand variability (coefficient of variation).
        X: CV < 0.5 (stable demand)
        Y: 0.5 <= CV < 1.0 (moderate variability)
        Z: CV >= 1.0 (high variability)
        """
        if cv < 0.5:
            return 'X'
        elif cv < 1.0:
            return 'Y'
        else:
            return 'Z'

    def calculate_safety_stock(
        self,
        avg_daily_demand: float,
        demand_std: float,
        lead_time: int,
        abc_class: str = 'B',
        xyz_class: str = 'Y'
    ) -> float:
        """
        Calculate safety stock based on demand variability and service level.

        Formula: SS = Z * σ_LT
        where σ_LT = σ_daily * sqrt(lead_time)
        """
        if avg_daily_demand <= 0:
            return 0

        # Lead time demand standard deviation
        lead_time_std = demand_std * np.sqrt(lead_time)

        # Base safety stock
        base_ss = self.z_score * lead_time_std

        # Apply ABC-XYZ multiplier
        multiplier = self.safety_stock_matrix.get_multiplier(abc_class, xyz_class)

        return base_ss * multiplier

    def calculate_reorder_point(
        self,
        avg_daily_demand: float,
        lead_time: int,
        safety_stock: float
    ) -> float:
        """
        Calculate reorder point.

        Formula: ROP = (avg_daily_demand * lead_time) + safety_stock
        """
        lead_time_demand = avg_daily_demand * lead_time
        return lead_time_demand + safety_stock

    def calculate_order_quantity(
        self,
        avg_daily_demand: float,
        review_period: int,
        lead_time: int,
        safety_stock: float,
        on_hand: float,
        on_order: float = 0,
        min_lot: float = 1
    ) -> float:
        """
        Calculate order quantity using periodic review model.

        Order up to level = demand during (review_period + lead_time) + safety_stock
        """
        # Target inventory level
        protection_period = review_period + lead_time
        target_level = (avg_daily_demand * protection_period) + safety_stock

        # Net inventory position
        inventory_position = on_hand + on_order

        # Order quantity
        order_qty = max(0, target_level - inventory_position)

        # Round up to minimum lot size
        if order_qty > 0 and min_lot > 1:
            order_qty = np.ceil(order_qty / min_lot) * min_lot

        return order_qty

    def project_inventory(
        self,
        on_hand: float,
        forecast: pd.Series,
        orders_schedule: Optional[pd.DataFrame] = None,
        lead_time: int = 7
    ) -> pd.DataFrame:
        """
        Project inventory levels based on forecast and orders.

        Args:
            on_hand: Current inventory
            forecast: Forecasted demand (monthly)
            orders_schedule: DataFrame with [date, quantity] of planned orders
            lead_time: Days for orders to arrive

        Returns:
            DataFrame with projected inventory by month
        """
        projections = []
        current_inventory = on_hand

        for date, demand in forecast.items():
            # Add incoming orders
            incoming = 0
            if orders_schedule is not None:
                # Orders placed lead_time days ago arrive now
                arrival_window = date - pd.Timedelta(days=lead_time)
                incoming = orders_schedule[
                    orders_schedule['date'] <= arrival_window
                ]['quantity'].sum()

            # Calculate end of period inventory
            end_inventory = current_inventory + incoming - demand

            # Track shortages
            shortage = max(0, -end_inventory)
            end_inventory = max(0, end_inventory)

            projections.append({
                'date': date,
                'beginning_inventory': current_inventory,
                'demand': demand,
                'incoming_orders': incoming,
                'ending_inventory': end_inventory,
                'shortage': shortage
            })

            current_inventory = end_inventory

        return pd.DataFrame(projections)

    def calculate_days_of_supply(
        self,
        on_hand: float,
        avg_daily_demand: float
    ) -> float:
        """Calculate how many days current inventory will last."""
        if avg_daily_demand <= 0:
            return 999 if on_hand > 0 else 0
        return on_hand / avg_daily_demand

    def get_inventory_status(
        self,
        on_hand: float,
        reorder_point: float,
        max_stock: float,
        safety_stock: float
    ) -> str:
        """
        Determine inventory status.

        Returns: 'shortage', 'low', 'optimal', 'overstock'
        """
        if on_hand <= 0:
            return 'stockout'
        elif on_hand < safety_stock:
            return 'critical'
        elif on_hand < reorder_point:
            return 'low'
        elif on_hand > max_stock:
            return 'overstock'
        else:
            return 'optimal'

    def generate_ordering_plan(
        self,
        items_df: pd.DataFrame,
        forecasts: Dict[str, pd.Series],
        months_ahead: int = 3
    ) -> pd.DataFrame:
        """
        Generate ordering plan for multiple items.

        Args:
            items_df: DataFrame with item info
            forecasts: Dictionary of forecasts per item
            months_ahead: How many months to plan

        Returns:
            DataFrame with ordering recommendations
        """
        orders = []

        for _, item in items_df.iterrows():
            item_code = item['item_code']
            on_hand = item.get('last_on_hand', 0)
            lead_time = item.get('lead_time', self.default_lead_time)
            min_lot = item.get('min_lot', 1) or 1
            unit_cost = item.get('inventory_value_unit', 0)

            forecast = forecasts.get(item_code)
            if forecast is None:
                continue

            # Get forecast for planning horizon
            forecast_horizon = forecast.head(months_ahead)
            total_demand = forecast_horizon.sum()
            avg_monthly_demand = total_demand / months_ahead if months_ahead > 0 else 0
            avg_daily_demand = avg_monthly_demand / 30

            # Calculate safety stock
            demand_std = forecast_horizon.std() / 30  # daily std
            safety_stock = self.calculate_safety_stock(
                avg_daily_demand, demand_std, lead_time
            )

            # Calculate order quantity
            order_qty = self.calculate_order_quantity(
                avg_daily_demand=avg_daily_demand,
                review_period=self.review_period,
                lead_time=lead_time,
                safety_stock=safety_stock,
                on_hand=on_hand,
                min_lot=min_lot
            )

            # Project when to order
            days_of_supply = self.calculate_days_of_supply(on_hand, avg_daily_demand)
            reorder_point = self.calculate_reorder_point(
                avg_daily_demand, lead_time, safety_stock
            )

            orders.append({
                'item_code': item_code,
                'on_hand': on_hand,
                'avg_daily_demand': avg_daily_demand,
                'safety_stock': safety_stock,
                'reorder_point': reorder_point,
                'days_of_supply': days_of_supply,
                'recommended_order_qty': order_qty,
                'order_value': order_qty * unit_cost,
                'lead_time': lead_time,
                'needs_order': on_hand <= reorder_point
            })

        return pd.DataFrame(orders)


def get_status_color(status: str) -> str:
    """Get color for inventory status display."""
    colors = {
        'stockout': '#FF0000',    # Red
        'critical': '#FF6600',    # Orange
        'low': '#FFCC00',         # Yellow
        'optimal': '#00CC00',     # Green
        'overstock': '#0066FF'    # Blue
    }
    return colors.get(status, '#808080')


def get_status_emoji(status: str) -> str:
    """Get emoji for inventory status."""
    emojis = {
        'stockout': '🔴',
        'critical': '🟠',
        'low': '🟡',
        'optimal': '🟢',
        'overstock': '🔵'
    }
    return emojis.get(status, '⚪')
