"""
Data loading and preprocessing module for inventory forecasting.
Handles loading transaction and item data from CSV/Excel files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from functools import lru_cache

# Try to import streamlit, but make it optional for testing
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None


def cache_data(func):
    """Decorator that uses streamlit cache if available, otherwise lru_cache."""
    if HAS_STREAMLIT and st is not None:
        try:
            return st.cache_data(ttl=3600)(func)
        except Exception:
            return lru_cache(maxsize=32)(func)
    else:
        return lru_cache(maxsize=32)(func)


class DataLoader:
    """Handles data loading and preprocessing for the forecasting system."""

    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            # Default to parent of the inventory_forecast folder
            self.data_dir = Path(__file__).parent.parent.parent
        else:
            self.data_dir = Path(data_dir)
        self._transactions_df: Optional[pd.DataFrame] = None
        self._items_df: Optional[pd.DataFrame] = None
        self._monthly_sales: Optional[pd.DataFrame] = None

    def load_transactions(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load and preprocess transaction data."""
        if file_path is None:
            file_path = self.data_dir / "UPLOAD Charle.xlsx - Transactions.csv"

        df = pd.read_csv(file_path)

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Parse dates
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['year_month'] = df['date'].dt.to_period('M')

        # Clean item codes
        df['item_code'] = df['item_code'].astype(str).str.strip()

        # Ensure numeric columns
        df['quantity_sold'] = pd.to_numeric(df['quantity_sold'], errors='coerce').fillna(0)
        df['transaction_revenue'] = pd.to_numeric(df['transaction_revenue'], errors='coerce').fillna(0)

        return df

    def load_items(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load and preprocess item info data."""
        if file_path is None:
            file_path = self.data_dir / "UPLOAD Charle.xlsx - Item info.csv"

        df = pd.read_csv(file_path)

        # Standardize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')

        # Clean item codes
        df['item_code'] = df['item_code'].astype(str).str.strip()

        # Clean numeric columns
        numeric_cols = ['last_on_hand', 'inventory_value_unit', 'lead_time', 'min_lot']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Clean supplier codes
        if 'supplier_code' in df.columns:
            df['supplier_code'] = df['supplier_code'].astype(str).str.strip()

        # Remove duplicates - keep first occurrence
        df = df.drop_duplicates(subset=['item_code'], keep='first')

        return df

    def get_monthly_sales(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transactions to monthly level per SKU."""
        monthly = transactions_df.groupby(['item_code', 'year_month']).agg({
            'quantity_sold': 'sum',
            'transaction_revenue': 'sum',
            'date': 'first'
        }).reset_index()

        monthly['date'] = monthly['year_month'].dt.to_timestamp()

        return monthly

    def create_complete_time_series(self, monthly_sales: pd.DataFrame) -> pd.DataFrame:
        """Create complete time series with zeros for missing months."""
        # Get date range
        min_date = monthly_sales['date'].min()
        max_date = monthly_sales['date'].max()

        # Create all months
        all_months = pd.date_range(start=min_date, end=max_date, freq='MS')

        # Get all unique items
        all_items = monthly_sales['item_code'].unique()

        # Create complete index
        complete_index = pd.MultiIndex.from_product(
            [all_items, all_months],
            names=['item_code', 'date']
        )

        # Reindex
        monthly_sales = monthly_sales.set_index(['item_code', 'date'])
        complete_df = monthly_sales.reindex(complete_index, fill_value=0).reset_index()

        complete_df['year_month'] = complete_df['date'].dt.to_period('M')

        return complete_df

    def get_item_summary(self, transactions_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics per item for filtering and ABC analysis."""
        # Aggregate transactions
        item_stats = transactions_df.groupby('item_code').agg({
            'quantity_sold': 'sum',
            'transaction_revenue': 'sum',
            'date': ['min', 'max', 'count']
        }).reset_index()

        # Flatten column names
        item_stats.columns = ['item_code', 'total_qty', 'total_revenue',
                             'first_sale', 'last_sale', 'num_transactions']

        # Merge with item info
        summary = item_stats.merge(items_df, on='item_code', how='left')

        # Calculate average monthly sales
        summary['months_active'] = (
            (summary['last_sale'] - summary['first_sale']).dt.days / 30
        ).clip(lower=1)
        summary['avg_monthly_qty'] = summary['total_qty'] / summary['months_active']
        summary['avg_monthly_revenue'] = summary['total_revenue'] / summary['months_active']

        # Calculate inventory value
        summary['total_inventory_value'] = (
            summary['last_on_hand'] * summary['inventory_value_unit']
        )

        return summary

    def perform_abc_analysis(self, item_summary: pd.DataFrame) -> pd.DataFrame:
        """
        Perform ABC analysis based on revenue contribution.
        A: Top 80% of revenue (typically ~20% of items)
        B: Next 15% of revenue (typically ~30% of items)
        C: Bottom 5% of revenue (typically ~50% of items)
        """
        df = item_summary.copy()

        # Sort by revenue descending
        df = df.sort_values('total_revenue', ascending=False)

        # Calculate cumulative revenue percentage
        total_revenue = df['total_revenue'].sum()
        df['cumulative_revenue'] = df['total_revenue'].cumsum()
        df['revenue_pct'] = df['cumulative_revenue'] / total_revenue * 100

        # Assign ABC categories
        def assign_abc(pct):
            if pct <= 80:
                return 'A'
            elif pct <= 95:
                return 'B'
            else:
                return 'C'

        df['abc_class'] = df['revenue_pct'].apply(assign_abc)

        # Also calculate XYZ analysis based on demand variability
        # (will be computed per-item in forecasting)

        return df


def load_all_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience function to load all data at once."""
    loader = DataLoader(data_dir)

    transactions = loader.load_transactions()
    items = loader.load_items()
    monthly_sales = loader.get_monthly_sales(transactions)

    return transactions, items, monthly_sales
