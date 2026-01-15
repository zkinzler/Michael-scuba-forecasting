# Inventory Forecasting Tool - Project Report

## Project Overview

This project is an inventory forecasting and demand planning tool built as a replacement for expensive commercial software (Streamline). The tool provides demand forecasting, inventory management, and reorder planning capabilities for a scuba/marine equipment business.

**Status:** MVP Complete - Core Features Functional
**Date:** January 14, 2026
**Tech Stack:** Python, Streamlit, Pandas, Statsmodels, Plotly

---

## Data Summary

- **Total SKUs:** ~21,000 items with sales history
- **Transaction Records:** ~89,700 transactions
- **Date Range:** January 2022 - December 2025 (4 years)
- **Data Sources:**
  - `UPLOAD Charle.xlsx - Transactions.csv` - Sales transaction data
  - `UPLOAD Charle.xlsx - Item info.csv` - Item master data with costs, lead times, suppliers

---

## Features Implemented

### 1. Demand Forecasting (Primary Feature)

**Location:** Demand Tab

- **Multiple Forecasting Models:**
  - Automatic model selection based on demand pattern classification
  - Seasonal & Trend (Holt-Winters) - for items with clear seasonality
  - Linear Trend (Holt's Method) - for trending items without seasonality
  - Constant Level (Simple Exponential Smoothing) - for stable demand
  - Intermittent/Croston - for slow-moving items with sporadic sales

- **Demand Classification:**
  - Syntetos-Boylan classification (ADI, CV2 metrics)
  - Pattern detection: smooth, erratic, intermittent, lumpy
  - Automatic model recommendation based on pattern

- **Forecast Display:**
  - 12-month forecast horizon on chart
  - Monthly breakdown table with editable final forecast
  - Confidence intervals
  - Projected inventory levels

- **Editable Forecasts:**
  - Users can override statistical forecasts with manual values
  - Revenue automatically recalculates based on edits
  - Changes persist in session state

### 2. ABC Analysis

- Revenue-based classification:
  - A Items: Top 80% of revenue
  - B Items: Next 15% of revenue
  - C Items: Bottom 5% of revenue
- Filter items by ABC class
- Visual indicators in item list

### 3. Inventory Management

**Location:** Inventory Tab

- **Reorder Point Calculation:**
  - Formula: (Average Daily Demand × Lead Time) + Safety Stock
  - Safety Stock: ~2 weeks buffer based on demand variability
  - Uses lead time from item master data (default 7 days)

- **Reorder Quantity Calculation:**
  - Order up to 3 months supply + safety stock
  - Respects minimum lot sizes from item data
  - Calculates reorder value (quantity × unit cost)

- **Inventory Metrics:**
  - Days of Supply
  - Inventory Value
  - Overstock/Shortage detection
  - Non-moving inventory identification

- **Filters:**
  - Needs Reorder - items below reorder point
  - Low Stock, Stockout, Overstock, Optimal
  - ABC Class filter
  - Sort by Days of Supply, Reorder Value, etc.

### 4. KPIs Panel

**Location:** Right panel when viewing individual items

- Reorder Status (ORDER NOW / OK)
- Reorder Point, Reorder Qty, Reorder Value, Safety Stock
- Inventory Value
- Overstock/Shortage Value
- Turnover (Turns/year, Days to sell)
- Gross Margin
- Turn-Earn Index
- Annual/Next Year Revenue & Sales

### 5. Reports Tab

- ABC Analysis Report with pie/bar charts
- Demand Forecast Summary
- Inventory Health Dashboard
- Revenue Projections (placeholder)

### 6. Search & Navigation

- Multi-term search (handles spaces in item codes)
- Filter by ABC class, Supplier
- Hide inactive items option
- Item list limited to 500 for performance

---

## Technical Architecture

```
inventory_forecast/
├── app.py                 # Main Streamlit application
├── run.sh                 # Startup script
├── requirements.txt       # Python dependencies
├── models/
│   └── forecasting.py     # Forecasting models and demand classification
└── utils/
    ├── data_loader.py     # Data loading and preprocessing
    ├── kpi_calculator.py  # KPI calculation utilities
    └── inventory.py       # Inventory calculations (safety stock, reorder)
```

### Key Classes

- **`Forecaster`** - Main forecasting engine with multiple model support
- **`DemandClassifier`** - Classifies demand patterns for model selection
- **`DataLoader`** - Handles CSV loading and data preprocessing
- **`InventoryCalculator`** - Calculates safety stock, reorder points, projections

---

## Known Issues & Limitations

1. **Performance:** Large item lists (21K+ SKUs) can be slow to filter. Limited to 500 items in dropdown for performance.

2. **Streamlit Deprecation Warnings:** `use_container_width` parameter will be deprecated after 2025-12-31. Should migrate to `width='stretch'`.

3. **Data Caching:** Raw data loading is cached for 1 hour. Use "Reload Data" button to refresh.

4. **Forecast Accuracy:** Croston/Intermittent model works well for slow-moving items. May need tuning for specific product categories.

---

## Future Enhancements (Not Yet Implemented)

1. **Supplier-Level Ordering:**
   - Aggregate reorder recommendations by supplier
   - Generate purchase orders

2. **Promotion/Event Handling:**
   - Adjust forecasts for known promotions
   - Holiday demand patterns

3. **Multi-Location Support:**
   - Inventory by warehouse/location
   - Transfer recommendations

4. **Export Capabilities:**
   - Excel export of forecasts
   - Reorder reports for purchasing

5. **Price Elasticity:**
   - Adjust forecasts based on pricing changes

6. **Historical Accuracy Tracking:**
   - Compare forecasts to actuals
   - Model performance metrics

---

## How to Run

```bash
cd inventory_forecast
source venv/bin/activate
streamlit run app.py --server.port 8501
```

Access at: http://localhost:8501

---

## Configuration

Default settings in the code:
- Forecast horizon: 36 months (12 shown in UI)
- Safety stock: ~2 weeks of average demand
- Reorder target: 3 months supply
- Lead time default: 7 days
- Service level: 95%

---

## Dependencies

```
streamlit
pandas
numpy
plotly
statsmodels
scipy
openpyxl
```

---

## Session Notes

### Key Fixes Made:
1. **Forecast Accuracy:** Fixed model selection logic - was defaulting to Seasonal Trend instead of Automatic. Now correctly uses Croston for intermittent demand items.

2. **Time Series Zeros:** Fixed `get_item_time_series()` to fill zeros for months with no sales, not just return sparse data.

3. **Search Functionality:** Fixed search to handle multiple spaces in item codes (e.g., "ACC1149   BLUE").

4. **Caching Issues:** Removed `@st.cache_data` from time series functions to prevent stale data display.

### Validation:
- Tested against competitor software (Streamline) for item ACC1077
- Forecast values now match: 0,1,1,0,0,1,1,2,1,1,0 for 2026
- Croston/Intermittent model correctly selected for slow-moving items

---

*Report generated: January 14, 2026*
