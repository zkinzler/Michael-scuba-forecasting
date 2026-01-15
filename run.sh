#!/bin/bash
# Run the Inventory Forecasting Tool

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run the app
streamlit run app.py --server.port 8501
