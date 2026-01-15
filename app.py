"""
Inventory Forecasting Tool - Main Streamlit Application
Built for scalability with eventual web app migration in mind.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import DataLoader
from utils.kpi_calculator import KPICalculator, format_currency, format_number
from utils.inventory import InventoryCalculator, get_status_color, get_status_emoji
from models.forecasting import Forecaster, ModelType, batch_forecast, DemandClassifier

# Page config
st.set_page_config(
    page_title="Inventory Forecasting Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main layout */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Data table styling */
    .monthly-table {
        font-size: 12px;
    }

    .editable-cell {
        background-color: #e8f5e9 !important;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding-left: 16px;
        padding-right: 16px;
        font-size: 14px;
    }

    /* Metric cards */
    .kpi-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #eee;
    }
    .kpi-label {
        color: #666;
        font-size: 13px;
    }
    .kpi-value {
        font-weight: 500;
        font-size: 13px;
    }

    /* Settings panel */
    .settings-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #f0f0f0;
    }

    /* Item list */
    .item-list-container {
        max-height: 400px;
        overflow-y: auto;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'selected_item' not in st.session_state:
    st.session_state.selected_item = None
if 'forecast_overrides' not in st.session_state:
    st.session_state.forecast_overrides = {}
if 'price_overrides' not in st.session_state:
    st.session_state.price_overrides = {}


@st.cache_data(ttl=3600)
def load_data():
    """Load and cache all data."""
    data_dir = Path(__file__).parent.parent
    loader = DataLoader(data_dir)

    transactions = loader.load_transactions()
    items = loader.load_items()
    monthly_sales = loader.get_monthly_sales(transactions)

    # Get item summary with ABC analysis
    item_summary = loader.get_item_summary(transactions, items)
    item_summary = loader.perform_abc_analysis(item_summary)

    return transactions, items, monthly_sales, item_summary


def get_item_time_series(monthly_sales: pd.DataFrame, item_code: str) -> pd.Series:
    """Get time series for a specific item with zeros for missing months."""
    item_data = monthly_sales[monthly_sales['item_code'] == item_code].copy()
    item_data = item_data.sort_values('date')

    if len(item_data) == 0:
        return pd.Series(dtype=float)

    # Get the full date range from the dataset
    min_date = monthly_sales['date'].min()
    max_date = monthly_sales['date'].max()

    # Create complete monthly index
    all_months = pd.date_range(start=min_date, end=max_date, freq='MS')

    # Create series with zeros for all months
    series = pd.Series(0.0, index=all_months)

    # Fill in actual sales data
    for _, row in item_data.iterrows():
        if row['date'] in series.index:
            series[row['date']] = row['quantity_sold']

    return series


def get_item_revenue_series(monthly_sales: pd.DataFrame, item_code: str) -> pd.Series:
    """Get revenue time series for a specific item."""
    item_data = monthly_sales[monthly_sales['item_code'] == item_code].copy()
    item_data = item_data.sort_values('date')

    if len(item_data) == 0:
        return pd.Series(dtype=float)

    series = pd.Series(
        item_data['transaction_revenue'].values,
        index=pd.DatetimeIndex(item_data['date'])
    )
    return series


def calculate_item_kpis(item_info: pd.Series, historical: pd.Series, forecast_result) -> dict:
    """Calculate KPIs for a single item including reorder point and reorder quantity."""
    on_hand = item_info.get('last_on_hand', 0)
    unit_cost = item_info.get('inventory_value_unit', 0)
    total_revenue = item_info.get('total_revenue', 0)
    total_qty = item_info.get('total_qty', 0)
    lead_time = item_info.get('lead_time', 7)  # Default 7 days if not specified
    min_lot = item_info.get('min_lot', 1) or 1

    # Calculate average price
    avg_price = total_revenue / total_qty if total_qty > 0 else 0

    # Inventory value
    inventory_value = on_hand * unit_cost

    # Annual sales (last 12 months)
    if len(historical) >= 12:
        annual_sales = historical.tail(12).sum()
        annual_revenue = annual_sales * avg_price
    else:
        annual_sales = historical.sum()
        annual_revenue = total_revenue

    # Next year projections from forecast
    if forecast_result and hasattr(forecast_result, 'forecast'):
        next_year_sales = forecast_result.forecast.head(12).sum()
        next_year_revenue = next_year_sales * avg_price
        # Use forecast for demand planning
        forecast_3mo = forecast_result.forecast.head(3).sum()
        avg_monthly_forecast = forecast_3mo / 3 if forecast_3mo > 0 else 0
    else:
        next_year_sales = annual_sales
        next_year_revenue = annual_revenue
        avg_monthly_forecast = 0

    # Turnover
    avg_inventory = on_hand  # Simplified
    cogs = annual_sales * unit_cost
    turnover_ratio = cogs / inventory_value if inventory_value > 0 else 0
    days_to_sell = 365 / turnover_ratio if turnover_ratio > 0 else 999

    # Gross margin
    gross_margin = (avg_price - unit_cost) / avg_price * 100 if avg_price > 0 else 0

    # Average monthly demand (use forecast if available, else historical)
    avg_monthly_demand = avg_monthly_forecast if avg_monthly_forecast > 0 else (annual_sales / 12 if annual_sales > 0 else 0)
    avg_daily_demand = avg_monthly_demand / 30

    # Months of supply
    months_of_supply = on_hand / avg_monthly_demand if avg_monthly_demand > 0 else 999

    # Overstock calculation (inventory > 6 months supply)
    overstock_qty = max(0, on_hand - (avg_monthly_demand * 6))
    overstock_value = overstock_qty * unit_cost

    # Shortage (if on_hand < 1 month supply)
    shortage_qty = max(0, avg_monthly_demand - on_hand) if months_of_supply < 1 else 0
    shortage_value = shortage_qty * unit_cost

    # Service level (simplified - 100% if in stock)
    service_level = 100 if on_hand > 0 else 0

    # Non-moving (no sales in last 6 months)
    if len(historical) >= 6:
        recent_sales = historical.tail(6).sum()
        is_non_moving = recent_sales == 0
    else:
        is_non_moving = False
    non_moving_value = inventory_value if is_non_moving else 0

    # Turn-earn index
    turn_earn_index = turnover_ratio * gross_margin / 100 if gross_margin != 0 else 0

    # === REORDER POINT AND REORDER QUANTITY ===
    # Safety stock: ~2 weeks of average demand as buffer
    demand_std = historical.tail(12).std() / 30 if len(historical) >= 12 else avg_daily_demand * 0.5
    safety_stock = max(avg_daily_demand * 14, 1.65 * demand_std * np.sqrt(lead_time))  # ~95% service level

    # Reorder Point = Lead time demand + Safety stock
    lead_time_demand = avg_daily_demand * lead_time
    reorder_point = lead_time_demand + safety_stock

    # Reorder Quantity = Order up to 3 months supply
    target_months_supply = 3
    target_inventory = avg_monthly_demand * target_months_supply + safety_stock
    reorder_quantity = max(0, target_inventory - on_hand)

    # Round up to min lot size
    if reorder_quantity > 0 and min_lot > 1:
        reorder_quantity = np.ceil(reorder_quantity / min_lot) * min_lot

    # Reorder value
    reorder_value = reorder_quantity * unit_cost

    # Check if needs reorder
    needs_reorder = on_hand <= reorder_point

    return {
        'inventory_value': inventory_value,
        'overstock_value': overstock_value,
        'shortage_value': shortage_value,
        'service_level': service_level,
        'non_moving_value': non_moving_value,
        'turnover_ratio': turnover_ratio,
        'days_to_sell': days_to_sell,
        'gross_margin': gross_margin,
        'turn_earn_index': turn_earn_index,
        'annual_revenue': annual_revenue,
        'next_year_revenue': next_year_revenue,
        'annual_sales': annual_sales,
        'next_year_sales': next_year_sales,
        'avg_price': avg_price,
        'months_of_supply': months_of_supply,
        'reorder_point': reorder_point,
        'reorder_quantity': reorder_quantity,
        'reorder_value': reorder_value,
        'safety_stock': safety_stock,
        'needs_reorder': needs_reorder,
        'lead_time': lead_time
    }


def create_monthly_table(historical: pd.Series, forecast_result, item_info: pd.Series, avg_price: float):
    """Create the monthly data table."""
    on_hand = item_info.get('last_on_hand', 0)

    # Get forecast months
    if forecast_result and hasattr(forecast_result, 'forecast'):
        forecast = forecast_result.forecast
        forecast_months = forecast.index[:9]  # Show 9 months to fit better
    else:
        return None, []

    month_cols = [m.strftime('%b %Y') for m in forecast_months]

    # Build table data
    actual_row = {'Metric': 'Actual sales'}
    stat_forecast_row = {'Metric': 'Statistical forecast'}
    final_forecast_row = {'Metric': 'Final forecast'}
    avg_price_row = {'Metric': 'Average sales price'}
    revenue_row = {'Metric': 'Revenue'}
    on_hand_row = {'Metric': 'On hand'}
    shortage_row = {'Metric': 'Shortage days'}

    current_inventory = on_hand

    for month in forecast_months:
        month_str = month.strftime('%b %Y')

        # Check if we have historical data for this month
        if month in historical.index:
            actual_val = int(historical[month])
        else:
            actual_val = None

        forecast_val = forecast[month] if month in forecast.index else 0
        forecast_int = int(round(forecast_val))

        actual_row[month_str] = actual_val
        stat_forecast_row[month_str] = forecast_int
        final_forecast_row[month_str] = forecast_int
        avg_price_row[month_str] = int(round(avg_price))
        revenue_row[month_str] = int(round(forecast_val * avg_price))

        # Project inventory
        current_inventory = current_inventory - forecast_val
        on_hand_row[month_str] = int(round(max(0, current_inventory)))

        # Shortage days
        if current_inventory < 0:
            shortage_days = min(30, abs(current_inventory) / (forecast_val / 30) if forecast_val > 0 else 30)
            shortage_row[month_str] = int(round(shortage_days))
        else:
            shortage_row[month_str] = 0

    df = pd.DataFrame([
        actual_row,
        stat_forecast_row,
        final_forecast_row,
        avg_price_row,
        revenue_row,
        on_hand_row,
        shortage_row
    ])

    return df, month_cols


def render_monthly_table_html(df: pd.DataFrame, month_cols: list):
    """Render monthly table as styled HTML."""
    if df is None:
        return ""

    # Build HTML table
    html = """
    <style>
        .forecast-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .forecast-table th {
            background-color: #1e1e1e;
            color: #fff;
            padding: 10px 8px;
            text-align: center;
            font-weight: 500;
            border-bottom: 1px solid #333;
        }
        .forecast-table th:first-child {
            text-align: left;
            width: 160px;
        }
        .forecast-table td {
            padding: 8px;
            text-align: center;
            border-bottom: 1px solid #333;
            color: #e0e0e0;
            background-color: #2d2d2d;
        }
        .forecast-table td:first-child {
            text-align: left;
            font-weight: 500;
            color: #fff;
            background-color: #1e1e1e;
        }
        .forecast-table tr.editable td {
            background-color: #1a3d1a;
            color: #4ade80;
        }
        .forecast-table tr.editable td:first-child {
            background-color: #1e1e1e;
            color: #4ade80;
        }
        .forecast-table tr:hover td {
            background-color: #3d3d3d;
        }
        .forecast-table tr.editable:hover td {
            background-color: #2a4d2a;
        }
    </style>
    <table class="forecast-table">
        <thead>
            <tr>
                <th>Metric</th>
    """

    for col in month_cols:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"

    editable_rows = ['Final forecast', 'Revenue']

    for _, row in df.iterrows():
        metric = row['Metric']
        row_class = 'editable' if metric in editable_rows else ''
        html += f'<tr class="{row_class}">'
        html += f'<td>{metric}</td>'

        for col in month_cols:
            val = row.get(col, '')
            if val is None or (isinstance(val, float) and pd.isna(val)):
                display_val = ''
            elif metric == 'Average sales price':
                display_val = f"{val}"
            else:
                display_val = f"{int(val)}" if val != '' else ''
            html += f'<td>{display_val}</td>'

        html += '</tr>'

    html += "</tbody></table>"
    return html


def render_forecast_chart(historical: pd.Series, forecast_result, on_hand: float, forecast_overrides: dict = None):
    """Render the forecast chart with multiple layers. Shows only 1 year of forecast."""
    fig = go.Figure()

    # Actual sales (historical)
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical.values,
        mode='lines+markers',
        name='Actual sales',
        line=dict(color='#2196F3', width=2),
        marker=dict(size=4),
        fill='tozeroy',
        fillcolor='rgba(33, 150, 243, 0.1)'
    ))

    if forecast_result and hasattr(forecast_result, 'forecast'):
        # Limit forecast to 12 months (1 year)
        forecast = forecast_result.forecast.head(12)
        fitted = forecast_result.fitted_values
        lower = forecast_result.confidence_lower.head(12)
        upper = forecast_result.confidence_upper.head(12)

        # Apply any user overrides to the forecast
        if forecast_overrides:
            forecast_values = forecast.values.copy()
            for i, date in enumerate(forecast.index):
                month_key = date.strftime('%Y-%m')
                if month_key in forecast_overrides:
                    forecast_values[i] = forecast_overrides[month_key]
            forecast = pd.Series(forecast_values, index=forecast.index)

        # Fitted values (on historical period)
        fig.add_trace(go.Scatter(
            x=fitted.index,
            y=fitted.values,
            mode='lines',
            name='Statistical forecast',
            line=dict(color='#FF9800', width=1, dash='dot')
        ))

        # Confidence interval (only for 1 year)
        fig.add_trace(go.Scatter(
            x=list(forecast.index) + list(forecast.index[::-1]),
            y=list(upper.values) + list(lower.values[::-1]),
            fill='toself',
            fillcolor='rgba(76, 175, 80, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence interval',
            showlegend=True
        ))

        # Final forecast line (1 year only)
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines+markers',
            name='Final forecast',
            line=dict(color='#4CAF50', width=2),
            marker=dict(size=4)
        ))

        # Projected inventory (1 year only)
        inventory_projection = []
        current_inv = on_hand
        for val in forecast.values:
            current_inv = max(0, current_inv - val)
            inventory_projection.append(current_inv)

        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=inventory_projection,
            mode='lines',
            name='Projected inventory',
            line=dict(color='#9C27B0', width=2, dash='dash'),
            yaxis='y2'
        ))

    # Layout
    fig.update_layout(
        height=350,
        margin=dict(l=50, r=50, t=30, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10)
        ),
        xaxis=dict(
            title='',
            showgrid=True,
            gridcolor='#f0f0f0'
        ),
        yaxis=dict(
            title='Quantity',
            showgrid=True,
            gridcolor='#f0f0f0'
        ),
        yaxis2=dict(
            title='Inventory',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        hovermode='x unified',
        plot_bgcolor='white'
    )

    return fig


def render_kpis_panel(kpis: dict):
    """Render the KPIs panel."""
    st.markdown("##### KPIs")

    # Reorder section - highlighted at top
    needs_reorder = kpis.get('needs_reorder', False)
    reorder_status = ":red[ORDER NOW]" if needs_reorder else ":green[OK]"

    st.markdown(f"**Reorder Status:** {reorder_status}")
    st.markdown("---")

    # Reorder metrics (prominent display)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Reorder Point", f"{kpis.get('reorder_point', 0):.0f} units")
    with col2:
        st.metric("Reorder Qty", f"{kpis.get('reorder_quantity', 0):.0f} units")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Reorder Value", f"${kpis.get('reorder_value', 0):,.2f}")
    with col2:
        st.metric("Safety Stock", f"{kpis.get('safety_stock', 0):.0f} units")

    st.markdown("---")

    # Other KPIs
    kpi_items = [
        ('Inventory value', f"${kpis['inventory_value']:,.2f}"),
        ('Overstock value', f"${kpis['overstock_value']:,.2f}"),
        ('Shortage value', f"${kpis['shortage_value']:,.2f}"),
        ('Annual service level', f"{kpis['service_level']:.0f}%"),
        ('Non-moving inventory value', f"${kpis['non_moving_value']:,.2f}"),
        ('Turnover | Turns/year', f"{kpis['turnover_ratio']:.1f}"),
        ('Turnover | Days to sell', f"{kpis['days_to_sell']:.0f}"),
        ('Gross margin', f"{kpis['gross_margin']:.2f}%"),
        ('Turn-earn index', f"{kpis['turn_earn_index']:.1f}"),
        ('Annual revenue', f"${kpis['annual_revenue']:,.2f}"),
        ('Next year revenue', f"${kpis['next_year_revenue']:,.2f}"),
        ('Annual sales', f"{kpis['annual_sales']:.0f}"),
        ('Next year sales', f"{kpis['next_year_sales']:.0f}"),
    ]

    for label, value in kpi_items:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"<span style='color:#666; font-size:13px;'>{label}</span>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<span style='font-weight:500; font-size:13px;'>{value}</span>", unsafe_allow_html=True)


def render_forecasting_settings():
    """Render the forecasting settings panel."""
    st.markdown("##### Forecasting Settings")

    settings = [
        ('Active', ['Inherit (Yes)', 'Yes', 'No']),
        ('Forecast approach', ['Inherit (Bottom-up)', 'Bottom-up', 'Top-down', 'Top-down with children']),
        ('Model type', ['Automatic', 'Seasonal & trend', 'Linear trend', 'Constant level', 'Intermittent', 'Croston']),
        ('Ignore zero sales', ['Inherit (No)', 'Yes', 'No']),
        ('Ignore shortage days', ['Inherit (No)', 'Yes', 'No']),
        ('Ignore trend before', ['Inherit (Jan 2022)', 'Jan 2022', 'Jan 2023', 'Jan 2024', 'None']),
        ('Automatic window', ['Inherit (Yes)', 'Yes', 'No']),
        ('Automatic outliers', ['Inherit (Yes)', 'Yes', 'No']),
        ('Use price elasticity', ['Inherit (No)', 'Yes', 'No']),
        ('Use holidays', ['Inherit (None)', 'None', 'US Holidays', 'Custom']),
        ('Use promotions', ['Inherit (Yes)', 'Yes', 'No']),
        ('Seasonality pattern', ['Inherit (Auto)', 'Auto', 'Monthly', 'Quarterly', 'None']),
        ('Weekly pattern', ['Inherit (None)', 'None', 'Auto', 'Custom']),
    ]

    for label, options in settings:
        st.selectbox(label, options, key=f"setting_{label.replace(' ', '_')}", label_visibility="visible")


def render_demand_tab(monthly_sales, items_df, item_summary):
    """Render the main Demand Forecasting tab with new layout."""

    # === LEFT SIDEBAR ===
    with st.sidebar:
        st.markdown("### Items")

        # Search box
        search_query = st.text_input("🔍 Search", placeholder="Search items...", key="item_search")

        # Configuration options
        with st.expander("Configuration", expanded=True):
            group_by_abc = st.checkbox("Group by ABC analysis", value=False)
            hide_overstock = st.checkbox("Hide overstock / shortage", value=False)
            hide_inactive = st.checkbox("Hide inactive items", value=False)

        # Filters
        st.markdown("#### Filters")
        abc_filter = st.selectbox("ABC Class", ['All', 'A', 'B', 'C'], key="abc_filter")

        # Get unique suppliers (limit for performance)
        suppliers = ['All'] + sorted([s for s in item_summary['supplier_code'].dropna().unique() if s.strip()][:30])
        supplier_filter = st.selectbox("Supplier", suppliers, key="supplier_filter")

        # Filter items
        filtered_items = item_summary.copy()

        if search_query:
            # Normalize spaces for better matching (item codes have multiple spaces)
            # Split search into terms and match all of them
            search_terms = search_query.lower().split()

            def matches_all_terms(row):
                # Combine item_code and description, normalize spaces
                searchable = f"{row['item_code']} {row.get('item_description', '')}".lower()
                searchable = ' '.join(searchable.split())  # Normalize multiple spaces
                return all(term in searchable for term in search_terms)

            mask = filtered_items.apply(matches_all_terms, axis=1)
            filtered_items = filtered_items[mask]

        if abc_filter != 'All':
            filtered_items = filtered_items[filtered_items['abc_class'] == abc_filter]

        if supplier_filter != 'All':
            filtered_items = filtered_items[filtered_items['supplier_code'] == supplier_filter]

        if hide_inactive:
            # Hide items with no recent sales
            filtered_items = filtered_items[filtered_items['avg_monthly_qty'] > 0]

        # Sort by revenue
        filtered_items = filtered_items.sort_values('total_revenue', ascending=False)

        # Item count
        st.markdown(f"**{len(filtered_items):,} items**")

        # Item list
        if len(filtered_items) == 0:
            st.warning("No items match filters")
            return

        item_options = filtered_items['item_code'].tolist()[:500]  # Limit for performance

        # Create display labels
        def format_item(code):
            row = filtered_items[filtered_items['item_code'] == code].iloc[0]
            desc = str(row.get('item_description', ''))[:30]
            abc = row.get('abc_class', '')
            return f"{code} — {desc}"

        selected_item = st.selectbox(
            "Select Item",
            item_options,
            format_func=format_item,
            key="selected_item_dropdown",
            label_visibility="collapsed"
        )

    # === MAIN CONTENT AREA ===
    if not selected_item:
        st.info("Select an item from the sidebar")
        return

    # Get item info
    item_info = filtered_items[filtered_items['item_code'] == selected_item].iloc[0]
    item_desc = str(item_info.get('item_description', ''))[:60]

    # Header with item name
    st.markdown(f"### {selected_item} — {item_desc}")

    # Get time series
    historical = get_item_time_series(monthly_sales, selected_item)
    revenue_series = get_item_revenue_series(monthly_sales, selected_item)

    if len(historical) < 3:
        st.warning("Insufficient data for forecasting (need at least 3 months)")
        return

    # Generate forecast
    forecaster = Forecaster(forecast_periods=36)

    # Get model type from settings (default to automatic)
    # "Inherit" settings should use AUTOMATIC to let the system pick the best model
    model_setting = st.session_state.get('setting_Model_type', 'Automatic')
    if model_setting.startswith('Inherit') or 'Automatic' in model_setting:
        model_type = ModelType.AUTOMATIC
    elif 'Croston' in model_setting or 'Intermittent' in model_setting:
        model_type = ModelType.CROSTON
    elif 'Seasonal' in model_setting:
        model_type = ModelType.SEASONAL_TREND
    elif 'Linear' in model_setting:
        model_type = ModelType.LINEAR_TREND
    elif 'Constant' in model_setting:
        model_type = ModelType.CONSTANT_LEVEL
    else:
        model_type = ModelType.AUTOMATIC

    with st.spinner("Generating forecast..."):
        forecast_result = forecaster.forecast(
            series=historical,
            item_code=selected_item,
            model_type=model_type
        )

    # Calculate KPIs
    kpis = calculate_item_kpis(item_info, historical, forecast_result)
    avg_price = kpis['avg_price']

    # === TOP SECTION: Monthly Data Table with Editable Forecast ===
    if forecast_result and hasattr(forecast_result, 'forecast'):
        st.markdown("#### Monthly Breakdown")

        forecast = forecast_result.forecast
        forecast_months = forecast.index[:12]  # Show 12 months
        on_hand = item_info.get('last_on_hand', 0)

        # Initialize forecast overrides for this item if not exists
        if selected_item not in st.session_state.forecast_overrides:
            st.session_state.forecast_overrides[selected_item] = {}

        # Create columns for each month
        cols = st.columns(len(forecast_months) + 1)

        # Header row with month names
        cols[0].markdown("**Metric**")
        for i, month in enumerate(forecast_months):
            cols[i + 1].markdown(f"**{month.strftime('%b %Y')}**")

        # Actual sales row
        cols = st.columns(len(forecast_months) + 1)
        cols[0].markdown("Actual sales")
        for i, month in enumerate(forecast_months):
            if month in historical.index:
                cols[i + 1].markdown(f"{int(historical[month])}")
            else:
                cols[i + 1].markdown("-")

        # Statistical forecast row
        cols = st.columns(len(forecast_months) + 1)
        cols[0].markdown("Statistical forecast")
        for i, month in enumerate(forecast_months):
            forecast_val = forecast[month] if month in forecast.index else 0
            cols[i + 1].markdown(f"{int(round(forecast_val))}")

        # Final forecast row (EDITABLE)
        cols = st.columns(len(forecast_months) + 1)
        cols[0].markdown("**:green[Final forecast]**")
        final_forecasts = {}
        for i, month in enumerate(forecast_months):
            month_key = month.strftime('%Y-%m')
            stat_val = int(round(forecast[month])) if month in forecast.index else 0
            # Get override or use statistical forecast
            current_val = st.session_state.forecast_overrides[selected_item].get(month_key, stat_val)
            new_val = cols[i + 1].number_input(
                f"forecast_{month_key}",
                min_value=0,
                value=int(current_val),
                step=1,
                key=f"final_forecast_{selected_item}_{month_key}",
                label_visibility="collapsed"
            )
            final_forecasts[month] = new_val
            if new_val != stat_val:
                st.session_state.forecast_overrides[selected_item][month_key] = new_val

        # Average price row
        cols = st.columns(len(forecast_months) + 1)
        cols[0].markdown("Avg sales price")
        for i, month in enumerate(forecast_months):
            cols[i + 1].markdown(f"${avg_price:.0f}")

        # Revenue row (calculated from final forecast)
        cols = st.columns(len(forecast_months) + 1)
        cols[0].markdown("**:green[Revenue]**")
        for i, month in enumerate(forecast_months):
            revenue = final_forecasts[month] * avg_price
            cols[i + 1].markdown(f"${revenue:.0f}")

        # On hand row (projected inventory)
        cols = st.columns(len(forecast_months) + 1)
        cols[0].markdown("On hand")
        current_inventory = on_hand
        for i, month in enumerate(forecast_months):
            current_inventory = max(0, current_inventory - final_forecasts[month])
            cols[i + 1].markdown(f"{int(current_inventory)}")

        # Shortage days row
        cols = st.columns(len(forecast_months) + 1)
        cols[0].markdown("Shortage days")
        current_inventory = on_hand
        for i, month in enumerate(forecast_months):
            forecast_val = final_forecasts[month]
            current_inventory = current_inventory - forecast_val
            if current_inventory < 0 and forecast_val > 0:
                shortage_days = min(30, abs(current_inventory) / (forecast_val / 30))
                cols[i + 1].markdown(f"{int(shortage_days)}")
            else:
                cols[i + 1].markdown("0")

    # === MIDDLE SECTION: Chart and Settings Side by Side ===
    col_chart, col_settings = st.columns([2, 1])

    with col_chart:
        # View toggle
        view_col1, view_col2 = st.columns([1, 4])
        with view_col1:
            chart_view = st.radio("View", ["Quantity", "Revenue"], horizontal=True, label_visibility="collapsed")

        # Chart
        on_hand = item_info.get('last_on_hand', 0)
        item_overrides = st.session_state.forecast_overrides.get(selected_item, {})
        fig = render_forecast_chart(historical, forecast_result, on_hand, item_overrides)
        st.plotly_chart(fig, use_container_width=True)

    with col_settings:
        # Tabs for settings panels
        settings_tabs = st.tabs(["Forecasting", "Model", "Inventory", "KPIs"])

        with settings_tabs[0]:  # Forecasting
            render_forecasting_settings()

        with settings_tabs[1]:  # Model
            st.markdown("##### Model")
            st.info("Not available at this level")
            st.markdown(f"**Current Model:** {forecast_result.model_type.value}")
            st.markdown("---")
            st.markdown(forecast_result.model_explanation)

        with settings_tabs[2]:  # Inventory
            st.markdown("##### Inventory")
            st.info("Not available at this level")
            st.markdown(f"**On Hand:** {item_info.get('last_on_hand', 0):.0f}")
            st.markdown(f"**Lead Time:** {item_info.get('lead_time', 7):.0f} days")
            st.markdown(f"**Unit Cost:** ${item_info.get('inventory_value_unit', 0):.2f}")

        with settings_tabs[3]:  # KPIs
            render_kpis_panel(kpis)


def render_inventory_tab(monthly_sales, items_df, item_summary):
    """Render the Inventory Management tab."""
    st.header("Inventory Overview")

    # Calculate aggregate stats
    total_inventory_value = (item_summary['last_on_hand'] * item_summary['inventory_value_unit']).sum()
    total_items = len(item_summary)

    inv_calc = InventoryCalculator()

    # Calculate days of supply and reorder metrics
    item_summary = item_summary.copy()

    def calc_inventory_metrics(row):
        avg_daily = row['avg_monthly_qty'] / 30 if row['avg_monthly_qty'] > 0 else 0.001
        lead_time = row.get('lead_time', 7) or 7
        on_hand = row['last_on_hand']
        unit_cost = row.get('inventory_value_unit', 0)

        # Days of supply
        days_of_supply = inv_calc.calculate_days_of_supply(on_hand, avg_daily)

        # Safety stock (~2 weeks buffer)
        safety_stock = avg_daily * 14

        # Reorder point = lead time demand + safety stock
        reorder_point = (avg_daily * lead_time) + safety_stock

        # Reorder quantity = order up to 3 months supply
        target_inventory = (row['avg_monthly_qty'] * 3) + safety_stock
        reorder_qty = max(0, target_inventory - on_hand)

        # Round to min lot if available
        min_lot = row.get('min_lot', 1) or 1
        if reorder_qty > 0 and min_lot > 1:
            reorder_qty = np.ceil(reorder_qty / min_lot) * min_lot

        reorder_value = reorder_qty * unit_cost
        needs_reorder = on_hand <= reorder_point

        return pd.Series({
            'days_of_supply': days_of_supply,
            'reorder_point': reorder_point,
            'reorder_qty': reorder_qty,
            'reorder_value': reorder_value,
            'needs_reorder': needs_reorder
        })

    metrics = item_summary.apply(calc_inventory_metrics, axis=1)
    item_summary = pd.concat([item_summary, metrics], axis=1)

    low_stock = item_summary[item_summary['days_of_supply'] < 14]
    overstock = item_summary[item_summary['days_of_supply'] > 180]
    stockout = item_summary[item_summary['last_on_hand'] == 0]
    needs_reorder = item_summary[item_summary['needs_reorder'] == True]

    # KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Inventory Value", format_currency(total_inventory_value))
    with col2:
        st.metric("Total SKUs", format_number(total_items))
    with col3:
        st.metric("Low Stock Items", len(low_stock), delta=f"-{len(stockout)} stockouts")
    with col4:
        st.metric("Overstock Items", len(overstock))
    with col5:
        total_reorder_value = needs_reorder['reorder_value'].sum()
        st.metric("Needs Reorder", len(needs_reorder), delta=f"${total_reorder_value:,.0f}")

    # Inventory table with filters
    st.subheader("Inventory Details")

    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Status Filter",
            ['All', 'Needs Reorder', 'Low Stock', 'Stockout', 'Overstock', 'Optimal']
        )
    with col2:
        abc_filter = st.selectbox("ABC Class", ['All', 'A', 'B', 'C'], key='inv_abc')
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ['Days of Supply', 'Reorder Value', 'Inventory Value', 'Total Revenue']
        )

    # Apply filters
    display_df = item_summary.copy()

    if status_filter == 'Needs Reorder':
        display_df = display_df[display_df['needs_reorder'] == True]
    elif status_filter == 'Low Stock':
        display_df = display_df[(display_df['days_of_supply'] < 14) & (display_df['last_on_hand'] > 0)]
    elif status_filter == 'Stockout':
        display_df = display_df[display_df['last_on_hand'] == 0]
    elif status_filter == 'Overstock':
        display_df = display_df[display_df['days_of_supply'] > 180]
    elif status_filter == 'Optimal':
        display_df = display_df[(display_df['days_of_supply'] >= 14) & (display_df['days_of_supply'] <= 180)]

    if abc_filter != 'All':
        display_df = display_df[display_df['abc_class'] == abc_filter]

    # Sort
    sort_map = {
        'Days of Supply': 'days_of_supply',
        'Reorder Value': 'reorder_value',
        'Inventory Value': 'total_inventory_value',
        'Total Revenue': 'total_revenue'
    }
    ascending = sort_by in ['Days of Supply']
    display_df = display_df.sort_values(sort_map[sort_by], ascending=ascending)

    # Select columns to display - now including reorder columns
    display_cols = [
        'item_code', 'item_description', 'abc_class', 'supplier_code',
        'last_on_hand', 'reorder_point', 'reorder_qty', 'reorder_value',
        'days_of_supply', 'total_inventory_value', 'avg_monthly_qty'
    ]
    display_cols = [c for c in display_cols if c in display_df.columns]

    # Rename columns for display
    column_rename = {
        'item_code': 'Item Code',
        'item_description': 'Description',
        'abc_class': 'ABC',
        'supplier_code': 'Supplier',
        'last_on_hand': 'On Hand',
        'reorder_point': 'Reorder Point',
        'reorder_qty': 'Reorder Qty',
        'reorder_value': 'Reorder Value',
        'days_of_supply': 'Days Supply',
        'total_inventory_value': 'Inv Value',
        'avg_monthly_qty': 'Avg Monthly'
    }

    display_data = display_df[display_cols].head(500).copy()
    display_data = display_data.rename(columns=column_rename)

    st.dataframe(
        display_data.style.format({
            'On Hand': '{:.0f}',
            'Reorder Point': '{:.0f}',
            'Reorder Qty': '{:.0f}',
            'Reorder Value': '${:,.2f}',
            'Days Supply': '{:.0f}',
            'Inv Value': '${:,.2f}',
            'Avg Monthly': '{:.1f}'
        }),
        use_container_width=True,
        height=500
    )

    # Export option
    if st.button("Export to Excel"):
        output_path = Path(__file__).parent / "exports" / "inventory_export.xlsx"
        output_path.parent.mkdir(exist_ok=True)
        display_df.to_excel(output_path, index=False)
        st.success(f"Exported to {output_path}")


def render_reports_tab(monthly_sales, items_df, item_summary):
    """Render the Reports tab."""
    st.header("Reports")

    report_type = st.selectbox(
        "Select Report",
        [
            "ABC Analysis",
            "Demand Forecast Summary",
            "Inventory Health",
            "Revenue Projections"
        ]
    )

    if report_type == "ABC Analysis":
        st.subheader("ABC Analysis - Revenue Distribution")

        # Calculate ABC stats
        abc_stats = item_summary.groupby('abc_class').agg({
            'item_code': 'count',
            'total_revenue': 'sum',
            'total_inventory_value': 'sum'
        }).reset_index()
        abc_stats.columns = ['ABC Class', 'Item Count', 'Total Revenue', 'Inventory Value']

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                abc_stats,
                values='Total Revenue',
                names='ABC Class',
                title='Revenue by ABC Class',
                color='ABC Class',
                color_discrete_map={'A': '#2196F3', 'B': '#FF9800', 'C': '#F44336'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                abc_stats,
                x='ABC Class',
                y='Item Count',
                title='Item Count by ABC Class',
                color='ABC Class',
                color_discrete_map={'A': '#2196F3', 'B': '#FF9800', 'C': '#F44336'}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **ABC Classification:**
        - **A Items**: Top 80% of revenue - High priority, tight inventory control
        - **B Items**: Next 15% of revenue - Medium priority
        - **C Items**: Bottom 5% of revenue - Lower priority
        """)

        st.dataframe(abc_stats.style.format({
            'Total Revenue': '${:,.2f}',
            'Inventory Value': '${:,.2f}'
        }), use_container_width=True)

    elif report_type == "Demand Forecast Summary":
        st.subheader("Demand Forecast Summary")
        top_items = item_summary.nlargest(20, 'total_revenue')

        forecast_data = []
        for _, item in top_items.iterrows():
            series = get_item_time_series(monthly_sales, item['item_code'])
            if len(series) >= 3:
                classifier = DemandClassifier()
                classification = classifier.classify_demand(series)
                forecast_data.append({
                    'Item Code': item['item_code'],
                    'Description': str(item.get('item_description', ''))[:40],
                    'ABC': item['abc_class'],
                    'Annual Revenue': item['total_revenue'],
                    'Pattern': classification['pattern'],
                    'Recommended Model': classification['recommended_model'].value
                })

        if forecast_data:
            st.dataframe(
                pd.DataFrame(forecast_data).style.format({'Annual Revenue': '${:,.2f}'}),
                use_container_width=True
            )

    elif report_type == "Inventory Health":
        st.subheader("Inventory Health Dashboard")

        inv_calc = InventoryCalculator()
        item_summary = item_summary.copy()
        item_summary['days_of_supply'] = item_summary.apply(
            lambda x: inv_calc.calculate_days_of_supply(
                x['last_on_hand'],
                x['avg_monthly_qty'] / 30 if x['avg_monthly_qty'] > 0 else 0.001
            ),
            axis=1
        )

        def categorize_health(dos):
            if dos == 0:
                return 'Stockout'
            elif dos < 14:
                return 'Critical'
            elif dos < 30:
                return 'Low'
            elif dos <= 90:
                return 'Optimal'
            else:
                return 'Overstock'

        item_summary['health_status'] = item_summary['days_of_supply'].apply(categorize_health)

        health_stats = item_summary.groupby('health_status').agg({
            'item_code': 'count',
            'total_inventory_value': 'sum'
        }).reset_index()
        health_stats.columns = ['Status', 'Item Count', 'Inventory Value']

        status_order = ['Stockout', 'Critical', 'Low', 'Optimal', 'Overstock']
        health_stats['Status'] = pd.Categorical(health_stats['Status'], categories=status_order, ordered=True)
        health_stats = health_stats.sort_values('Status')

        col1, col2 = st.columns(2)

        with col1:
            colors = ['#F44336', '#FF9800', '#FFC107', '#4CAF50', '#2196F3']
            fig = px.pie(
                health_stats,
                values='Item Count',
                names='Status',
                title='Items by Health Status',
                color='Status',
                color_discrete_sequence=colors
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                health_stats,
                x='Status',
                y='Inventory Value',
                title='Inventory Value by Health Status',
                color='Status',
                color_discrete_sequence=colors
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(health_stats.style.format({'Inventory Value': '${:,.2f}'}), use_container_width=True)

    elif report_type == "Revenue Projections":
        st.subheader("Revenue Projections")
        st.info("Revenue projection report coming soon.")


def main():
    """Main application entry point."""
    st.title("📊 Inventory Forecasting Tool")

    # Load data
    with st.spinner("Loading data..."):
        try:
            transactions, items, monthly_sales, item_summary = load_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Please ensure the CSV files are in the parent directory.")
            return

    # Show data summary in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Data Summary**")
        st.write(f"{len(item_summary):,} SKUs")
        st.write(f"{len(transactions):,} transactions")
        date_range = f"{transactions['date'].min().strftime('%Y-%m')} to {transactions['date'].max().strftime('%Y-%m')}"
        st.write(f"{date_range}")

        # Cache clear button
        if st.button("Reload Data", help="Clear cached data and reload from files"):
            st.cache_data.clear()
            st.rerun()

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📈 Demand", "📦 Inventory", "📊 Reports"])

    with tab1:
        render_demand_tab(monthly_sales, items, item_summary)

    with tab2:
        render_inventory_tab(monthly_sales, items, item_summary)

    with tab3:
        render_reports_tab(monthly_sales, items, item_summary)


if __name__ == "__main__":
    main()
