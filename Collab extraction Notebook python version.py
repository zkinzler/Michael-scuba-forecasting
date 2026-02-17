# %% [markdown]
# <a href="https://colab.research.google.com/github/zkinzler/Michael-scuba-forecasting/blob/main/UpdatedScraper.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ##Toby start here Fetch 10-k & 10-Q

# %%
from requests.exceptions import HTTPError
import yfinance as yf
from yahooquery import Ticker
from yahooquery import search  # Unofficial Yahoo Finance search
from google.colab import drive
from IPython.display import display, clear_output
import ipywidgets as widgets
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.colab import auth
from gspread_dataframe import set_with_dataframe
import gspread
from google.auth import default
from google.colab import drive, auth
from tqdm import tqdm
import time
import requests
import pandas as pd
from datetime import datetime
import os

# Function to download the SEC Edgar bulk master.idx for a given year and quarter


def download_bulk_data(year, quarter):
    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"
    headers = {
        'User-Agent': 'Zach Kinzler (zkinzler@sandiego.edu)',
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'www.sec.gov'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        file_path = f"master_{year}_QTR{quarter}.idx"
        with open(file_path, 'wb') as file:
            file.write(response.content)
        return file_path
    else:
        print(f"Failed to download {url}: HTTP status {response.status_code}")
        return None


# Function to parse the downloaded .idx file and extract only 10-Q filings
def parse_idx_file(file_path):
    filings = []
    with open(file_path, 'r', encoding='latin-1') as file:
        for line in file:
            parts = line.strip().split('|')
            # We check for '10-Q' in the 3rd field (index 2)
            if len(parts) == 5 and parts[2].strip() == '10-Q':
                filings.append({
                    'CIK': parts[0].strip(),
                    'Company Name': parts[1].strip(),
                    'Form Type': parts[2].strip(),
                    'Date Filed': parts[3].strip(),
                    'File Name': parts[4].strip(),

                })
    return filings


# Function to download and parse multiple quarters/years
def fetch_bulk_data(years, quarters):
    all_filings = []
    for year in years:
        for quarter in quarters:
            print(f"Fetching QTR{quarter}, {year}...")
            file_path = download_bulk_data(year, quarter)
            if file_path:
                # Parse the .idx file
                filings = parse_idx_file(file_path)
                all_filings.extend(filings)
                # Remove the .idx file after parsing to save space
                os.remove(file_path)
    return all_filings


# ========== Main script logic ==========

# Specify the years and quarters you want to check
# For example, 2023 to 2025 and all 4 quarters
years = range(2024, 2026)   # Adjust as needed
# Could limit to [1, 2] if you only need the first two quarters, etc.
quarters = [1, 2, 3, 4]

# 1) Fetch bulk filings
bulk_filings = fetch_bulk_data(years, quarters)

# 2) Convert results to a DataFrame
df_bulk = pd.DataFrame(bulk_filings)

# 3) Convert 'Date Filed' to datetime, so we can sort properly
df_bulk['Date Filed'] = pd.to_datetime(df_bulk['Date Filed'], errors='coerce')

# 4) Sort by CIK and then by Date Filed (descending), so the most recent date is first
df_bulk = df_bulk.sort_values(['CIK', 'Date Filed'], ascending=[True, False])

# 5) Drop duplicates to keep only the most recent 10-Q filing for each CIK
df_recent_filings = df_bulk.drop_duplicates(subset='CIK', keep='first')

# 6) Save to CSV
df_recent_filings.to_csv('bulk_filings.csv', index=False)
print("Most recent 10-Q filings saved to 'bulk_filings.csv'.")


# %% [markdown]
# #Spliting into have and dont have information

# %%

try:
    from google.colab import drive
    _is_colab = True
except ModuleNotFoundError:
    drive = None
    _is_colab = False

# ===============================================================================
# 1) Load your newly pulled 10-Q filings (bulk_filings.csv) from local storage
# ===============================================================================
bulk_filings_path = 'bulk_filings.csv'
# ===============================================================================
# 2) Mount your Google Drive (same approach as your original example)
# ===============================================================================
if _is_colab and drive is not None:
    drive.mount('/content/drive')
    # Define the path to your folder in Google Drive
    folder_path = '/content/drive/My Drive/Fund LOL'
else:
    # Fallback to a local folder when not running in Colab
    folder_path = os.getcwd()

# ===============================================================================
# 3) Load and prepare your master file
# ===============================================================================
master_file_path = os.path.join(folder_path, 'AllSecDATA.csv')
if os.path.exists(master_file_path):
    master_df = pd.read_csv(master_file_path)
else:
    print(
        f"Warning: '{master_file_path}' not found. Using an empty master file.")
    master_df = pd.DataFrame(columns=['CIK', 'Company Name', 'Date'])

# Ensure the master CIK is zero-padded
if 'CIK' in master_df.columns:
    master_df['CIK'] = master_df['CIK'].astype(str).str.zfill(10)

# Identify the master file's date column (just like your example).
if 'Last Update' in master_df.columns:
    master_df['Last Update'] = pd.to_datetime(
        master_df['Last Update'], errors='coerce')
    date_column = 'Last Update'
elif 'Date' in master_df.columns:
    master_df['Date'] = pd.to_datetime(master_df['Date'], errors='coerce')
    date_column = 'Date'
else:
    raise ValueError(
        "Master file must contain a 'Date' or 'Last Update' column.")

# Define which SEC-related columns you want to keep from the master file.
# (Adapt as needed if your master file has different or additional columns.)
sec_columns = [
    'CIK',
    # or 'Company Name_x' / 'Company Name_y' if that matches your file
    'Company Name',
    'Current Assets',
    'Total Assets',
    'Accounts Receivable',
    'Inventory',
    'Prepaid Expenses',
    'Total Liabilities',
    'Stockholders Equity',
    'Cash and Cash Equivalents',
    'Net Income',
    'PP&E',
    'Equity Method Investments',
    'Marketable Securities Current',
    'Other Assets',
    date_column                   # Keep the date column to figure out recency
]

# Take only those columns from the master, if they exist
master_sec_df = master_df.reindex(columns=sec_columns).copy()

# Sort descending by date to get the latest entry for each CIK at the top
master_sec_df.sort_values(by=[date_column], ascending=False, inplace=True)

# Drop duplicates so only the most recent row per CIK remains
master_sec_df.drop_duplicates(subset='CIK', keep='first', inplace=True)

# Rename the master file date column to something consistent like 'Master Date'
master_sec_df.rename(columns={date_column: 'Master Date'}, inplace=True)

# ===============================================================================
# 4) Merge with df_new on CIK and compare dates
# ===============================================================================
# Use the already-loaded recent filings as df_new
df_new = df_recent_filings.copy()

# Ensure Date Filed is datetime for reliable comparison
df_new['Date Filed'] = pd.to_datetime(df_new['Date Filed'], errors='coerce')

df_merged = pd.merge(
    df_new,
    master_sec_df,
    on='CIK',
    how='left',
    suffixes=('_new', '_master')  # If any columns overlap in name
)

# Create a boolean indicating if the master file is as recent or newer
# than the newly found filing.
#   True  => Master Date >= new Date Filed (we already have the latest 10-Q)
#   False => Master Date is NaN or older than new Date Filed
df_merged['already_has_info'] = df_merged.apply(
    lambda row: pd.notna(row['Master Date']) and (
        row['Master Date'] >= row['Date Filed']),
    axis=1
)

# ===============================================================================
# 5) Split into two CSV files
# ===============================================================================
# (A) already_have_info.csv: Master file is up to date
df_already = df_merged[df_merged['already_has_info'] == True].copy()
df_already.drop(columns=['already_has_info'], inplace=True)
df_already.to_csv('already_have_info.csv', index=False)
print(
    f"Saved {len(df_already)} rows to 'already_have_info.csv' (master is up to date).")

# (B) missing_info.csv: Master file is missing or older
df_missing = df_merged[df_merged['already_has_info'] == False].copy()
df_missing.drop(columns=['already_has_info'], inplace=True)
df_missing.to_csv('missing_info.csv', index=False)
print(f"Saved {len(df_missing)} rows to 'missing_info.csv' (master is missing or outdated).")


# %% [markdown]
# #Pull all the information from Missing Info

# %%

# ==================== 1) MOUNT GOOGLE DRIVE AND LOAD MASTER FILE ====================
try:
    from google.colab import drive
    drive.mount('/content/drive')

    if 'folder_path' not in globals():
        # Define the path to your folder in Google Drive
        folder_path = '/content/drive/My Drive/Fund LOL'
except ModuleNotFoundError:
    if 'folder_path' not in globals():
        folder_path = os.getcwd()

# Path to master file in Drive
master_file_path = os.path.join(folder_path, 'AllSecDATA.csv')

# Load the master file (create empty if missing)
if os.path.exists(master_file_path):
    master_df = pd.read_csv(master_file_path)
else:
    master_df = pd.DataFrame(columns=['CIK', 'Company Name', 'Date'])

# Ensure the master CIK is zero-padded
master_df['CIK'] = master_df['CIK'].astype(str).str.zfill(10)

# Identify the date column in the master file (Date, Last Update, or Date Filed)
if 'Last Update' in master_df.columns:
    master_df['Last Update'] = pd.to_datetime(
        master_df['Last Update'], errors='coerce')
    date_column = 'Last Update'
elif 'Date' in master_df.columns:
    master_df['Date'] = pd.to_datetime(master_df['Date'], errors='coerce')
    date_column = 'Date'
elif 'Date Filed' in master_df.columns:
    master_df['Date Filed'] = pd.to_datetime(
        master_df['Date Filed'], errors='coerce')
    date_column = 'Date Filed'
else:
    raise ValueError(
        "Master file must contain a 'Last Update', 'Date', or 'Date Filed' column.")

print("Master file loaded. Shape:", master_df.shape)

# ==================== 2) LOAD THE MISSING INFO DATA ====================
missing_info_csv = 'missing_info.csv'
df_missing = pd.read_csv(missing_info_csv)
df_missing['CIK'] = df_missing['CIK'].astype(str).str.zfill(10)

# Identify which column in df_missing is the date field
if 'Date Filed_new' in df_missing.columns:
    df_missing['Date Filed_new'] = pd.to_datetime(
        df_missing['Date Filed_new'], errors='coerce')
    date_col_missing = 'Date Filed_new'
elif 'Date Filed' in df_missing.columns:
    df_missing['Date Filed'] = pd.to_datetime(
        df_missing['Date Filed'], errors='coerce')
    date_col_missing = 'Date Filed'
else:
    raise ValueError(
        "Could not find a 'Date Filed' or 'Date Filed_new' column in missing_info.csv.")

print("Missing info CSV loaded. Shape:", df_missing.shape)

# ==================== 3) EDGAR FETCH UTILITIES (WITH DELAY) ====================


def get_company_concept_data(cik, taxonomy, concept):
    """
    For a given CIK, taxonomy (e.g., 'us-gaap'), and concept name,
    fetch the XBRL JSON data from the SEC EDGAR API.
    We add a short delay to avoid hitting rate limits.
    """
    url = f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{concept}.json'
    headers = {
        # Replace with your details
        'User-Agent': 'Your Name (YourEmail@example.com)'
    }
    response = requests.get(url, headers=headers)
    # Delay between requests
    time.sleep(0.1)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def extract_financial_data(data):
    """
    From the JSON data, return a list of (end_date, value) for
    10-Q forms only, sorted by newest 'end' date.
    """
    if not data:
        return []

    facts = data.get('units', {}).get('USD', [])
    # Only keep items from 10-Q
    facts_10q = [fact for fact in facts if fact.get('form', '') == '10-Q']
    # Sort descending by end date
    facts_10q.sort(key=lambda x: x['end'], reverse=True)
    # Return only the first (newest) record if available
    return [(fact['end'], fact['val']) for fact in facts_10q[:1]]


# ==================== 4) BALANCE SHEET CONCEPTS ====================
extra_ppe_words = [
    "PropertyPlantAndEquipmentGross", "RealEstateInvestmentPropertyNet",
    "InventoryRealEstateHeldForSale", "PublicUtilitiesPropertyPlantAndEquipmentNet",
    "OperatingLeaseRightOfUseAsset"
]
inventory_words = [
    "InventoryRawMaterialsAndSuppliesNetOfReserves", "InventoryGross", "InventoryFinishedGoods",
    "InventoryWorkInProcessAndRawMaterials", "InventoryRawMaterialsAndSupplies", "InventoryMerchandise"
]
land_words = [
    "Land", "TimberAndTimberlands", "RealEstateAssets", "LandAndLandImprovements",
    "LandHeldForDevelopment", "AreaOfLand"
]
intangible_assets_words = [
    "IntangibleAssetsCurrent", "IntangibleAssetsNet", "FiniteLivedPatentsGross",
    "AmortizationOfIntangibleAssets", "FiniteLivedIntangibleAssetsGross"
]

balance_sheet_concepts = {
    'AssetsCurrent': ['AssetsCurrent'],
    'Assets': ['Assets'],
    'Liabilities': ['Liabilities', 'LiabilitiesCurrent'],
    'StockholdersEquity': ['StockholdersEquity'],
    'CashAndCashEquivalents': [
        'CashAndCashEquivalentsAtCarryingValue', 'Cash',
        'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'
    ],
    'AccountsReceivableNetCurrent': ['AccountsReceivableNetCurrent'],
    'InventoryNet': ['InventoryNet'] + inventory_words,
    'PrepaidExpenseAndOtherAssetsCurrent': [
        'PrepaidExpenseAndOtherAssetsCurrent', 'PrepaidExpenseCurrent',
        'PrepaidExpensesAndOtherCurrentAssets', 'OtherReceivablesCurrent'
    ],
    'PropertyPlantAndEquipmentNet': ['PropertyPlantAndEquipmentNet'] + extra_ppe_words,
    'EquityMethodInvestments': ['EquityMethodInvestments'],
    'MarketableSecuritiesCurrent': ['MarketableSecuritiesCurrent'],
    'OtherAssets': ['OtherAssets', 'OtherAssetsCurrent', 'OtherAssetsNoncurrent'],
    'Land': land_words,
    'IntangibleAssets': intangible_assets_words
}

net_income_concepts = [
    'NetIncomeLoss', 'NetIncome', 'NetIncomeAvailableToCommonStockholdersBasic'
]

# ==================== 5) GATHER NEW FINANCIAL DATA FOR MISSING CIKs ====================
new_company_data = []

print("Pulling updated 10-Q data from EDGAR for missing companies...")
for idx, row in tqdm(df_missing.iterrows(), total=len(df_missing)):
    cik_number = row['CIK']

    # Local placeholders for each concept
    current_assets = None
    total_assets = None
    liabilities = None
    stockholders_equity = None
    cash_and_cash_equivalents = None
    accounts_receivable = None
    inventory = None
    prepaid_expenses = None
    ppe = None
    equity_method_investments = None
    marketable_securities_current = None
    other_assets = None
    land = None
    intangible_assets = None
    net_income = None

    # (A) Fetch each balance sheet concept
    for concept_name, tags in balance_sheet_concepts.items():
        for tag in tags:
            data = get_company_concept_data(cik_number, 'us-gaap', tag)
            financial_data = extract_financial_data(data)
            if financial_data:
                # We'll use only the single newest (end_date, value)
                _, val = financial_data[0]
                if concept_name == 'AssetsCurrent':
                    current_assets = val
                elif concept_name == 'Assets':
                    total_assets = val
                elif concept_name == 'Liabilities':
                    liabilities = val
                elif concept_name == 'StockholdersEquity':
                    stockholders_equity = val
                elif concept_name == 'CashAndCashEquivalents':
                    cash_and_cash_equivalents = val
                elif concept_name == 'AccountsReceivableNetCurrent':
                    accounts_receivable = val
                elif concept_name == 'InventoryNet':
                    inventory = val
                elif concept_name == 'PrepaidExpenseAndOtherAssetsCurrent':
                    prepaid_expenses = val
                elif concept_name == 'PropertyPlantAndEquipmentNet':
                    ppe = val
                elif concept_name == 'EquityMethodInvestments':
                    equity_method_investments = val
                elif concept_name == 'MarketableSecuritiesCurrent':
                    marketable_securities_current = val
                elif concept_name == 'OtherAssets':
                    other_assets = val
                elif concept_name == 'Land':
                    land = val
                elif concept_name == 'IntangibleAssets':
                    intangible_assets = val
                break  # Found data for this concept, move to next

    # (B) Fetch net income
    for concept in net_income_concepts:
        data = get_company_concept_data(cik_number, 'us-gaap', concept)
        financial_data = extract_financial_data(data)
        if financial_data:
            net_income = financial_data[0][1]
            break

    # (C) Build the new data row
    new_row = {
        'CIK': cik_number,
        'Date Filed': row[date_col_missing],
        'Company Name': row.get('Company Name_new', row.get('Company Name', 'Unknown')),
        'Current Assets': current_assets,
        'Total Assets': total_assets,
        'Accounts Receivable': accounts_receivable,
        'Inventory': inventory,
        'Prepaid Expenses': prepaid_expenses,
        'Total Liabilities': liabilities,
        'Stockholders Equity': stockholders_equity,
        'Cash and Cash Equivalents': cash_and_cash_equivalents,
        'Net Income': net_income,
        'PP&E': ppe,
        'Equity Method Investments': equity_method_investments,
        'Marketable Securities Current': marketable_securities_current,
        'Other Assets': other_assets,
        'Land': land,
        'Intangible Assets': intangible_assets
    }

    new_company_data.append(new_row)

print("Finished pulling new data from EDGAR.")

# ==================== 6) TURN NEW DATA INTO A DATAFRAME ====================
df_new_data = pd.DataFrame(new_company_data)
df_new_data.fillna(0, inplace=True)

# Convert new data's 'Date Filed' to datetime to match the master date column
df_new_data['Date Filed'] = pd.to_datetime(
    df_new_data['Date Filed'], errors='coerce')

# If the master uses a different date column, rename in the new data to match
if date_column != 'Date Filed':
    df_new_data.rename(columns={'Date Filed': date_column}, inplace=True)

# ==================== 7) APPEND THE NEW ROWS TO THE MASTER FILE ====================
# We'll combine but keep old rows for historical data
combined_df = pd.concat([master_df, df_new_data], ignore_index=True)

# Sort so newest entries come first
combined_df.sort_values(by=[date_column], ascending=False, inplace=True)

# Drop duplicates ONLY if same CIK and same date
combined_df.drop_duplicates(
    subset=['CIK', date_column], keep='first', inplace=True)

# ==================== 8) SAVE UPDATED MASTER FILE TO DRIVE ====================
combined_df.to_csv(master_file_path, index=False)
print(f"Updated master file saved to: {master_file_path}")


# %% [markdown]
# #Finding Common CIKs and merging them onto DailyFinancials

# %%

# Step 1: Authenticate and mount your drive
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
drive.mount('/content/drive')

# Step 2: Read CIK values from KAHLERS SHEET
sheet_name = "CIK-Ticker"
spreadsheet = gc.open(sheet_name)
worksheet = spreadsheet.sheet1

# Get all rows from the sheet and find the CIK column index
data = worksheet.get_all_values()
header = data[0]
try:
    cik_index = header.index("CIK")
except ValueError:
    raise Exception("KAHLERS SHEET does not have a column named 'CIK'.")

# Extract the CIK values (skip the header) and remove empty strings
kahler_ciks = [row[cik_index]
               for row in data[1:] if row[cik_index].strip() != '']
print(f"Retrieved {len(kahler_ciks)} CIK values from KAHLERS SHEET.")

# Step 3: Read CIK values from AllSecDATA CSV
folder_path = '/content/drive/My Drive/Fund LOL'
allsec_file_path = os.path.join(folder_path, 'AllSecDATA.csv')
df_allsec = pd.read_csv(allsec_file_path)
# Ensure the CIK column is treated as string
allsec_ciks = df_allsec['CIK'].astype(str).tolist()
print(f"Retrieved {len(allsec_ciks)} CIK values from AllSecDATA.")

# Step 4: Normalize the CIK values


def normalize_cik(cik):
    # Remove leading/trailing spaces and any leading zeros
    return str(cik).strip().lstrip('0')


kahler_ciks_normalized = [normalize_cik(cik) for cik in kahler_ciks]
allsec_ciks_normalized = [normalize_cik(cik) for cik in allsec_ciks]

# Step 5: Compute the intersection (common normalized CIKs)
set_kahler = set(kahler_ciks_normalized)
set_allsec = set(allsec_ciks_normalized)
common_ciks = set_kahler.intersection(set_allsec)
print("Number of common normalized CIKs:", len(common_ciks))

# Step 6: Save the common normalized CIKs to a CSV for checking
df_common = pd.DataFrame(sorted(common_ciks), columns=[
                         "Normalized Common CIKs"])
output_csv_path = os.path.join(folder_path, "Common_CIKs_Normalized.csv")
df_common.to_csv(output_csv_path, index=False)
print("CSV file with common normalized CIKs saved to:", output_csv_path)


# %% [markdown]
# #Merging into Daily Financials Sheet
#

# %%

# ----- Step 1: Authenticate & Mount Drive -----
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
drive.mount('/content/drive')

# ----- Step 2: Load Ticker Data from the "CIK-Ticker" Google Sheet -----
ticker_sheet = gc.open("CIK-Ticker")
ticker_ws = ticker_sheet.sheet1
ticker_data = ticker_ws.get_all_records()
df_ticker = pd.DataFrame(ticker_data)
print(f"Ticker sheet loaded with {df_ticker.shape[0]} rows.")

# Ensure CIK values are padded to 10 digits (matching your master file)
df_ticker['CIK'] = df_ticker['CIK'].astype(str).str.zfill(10)

# Warn if the 'Ticker' column is missing.
if 'Ticker' not in df_ticker.columns:
    print("Warning: 'Ticker' column not found in the CIK-Ticker sheet.")

# ----- Step 3: Load AllSecDATA from CSV -----
folder_path = '/content/drive/My Drive/Fund LOL'
allsec_file_path = os.path.join(folder_path, 'AllSecDATA.csv')
df_allsec = pd.read_csv(allsec_file_path, dtype={'CIK': str})
df_allsec['Date'] = pd.to_datetime(df_allsec['Date'], errors='coerce')
print(f"AllSecDATA CSV loaded with {df_allsec.shape[0]} rows.")

# Pad the AllSecDATA CIK values to 10 digits
df_allsec['CIK'] = df_allsec['CIK'].astype(str).str.zfill(10)

# ----- Step 4: Select the Most Recent Financial Record for Each CIK -----
# Group by CIK and choose the record with the latest Date
df_allsec_recent = df_allsec.loc[df_allsec.groupby('CIK')['Date'].idxmax()]

# Define the financial columns (note: "Date" and "Land" are separate)
financial_columns = [
    "Current Assets", "Total Assets", "Accounts Receivable", "Inventory",
    "Prepaid Expenses", "Total Liabilities", "Stockholders Equity",
    "Cash and Cash Equivalents", "Net Income", "PP&E",
    "Equity Method Investments", "Marketable Securities Current",
    "Other Assets", "Date", "Land", "Intangible Assets", "Company Name"
]
df_allsec_recent = df_allsec_recent[['CIK'] + financial_columns]
print(
    f"Filtered to {df_allsec_recent.shape[0]} most recent financial records.")

# ----- Step 5: Merge the Ticker Data with the Most Recent Financial Data -----
# Use suffixes to differentiate overlapping columns (e.g. "Company Name")
merged_df = pd.merge(df_ticker, df_allsec_recent, on='CIK', how='inner',
                     suffixes=('_ticker', '_sec'))
print(f"Merged DataFrame contains {merged_df.shape[0]} rows!")

# ----- Step 5a: Reorder Columns to Put Ticker Data First -----
# Build a list of ticker sheet columns as they appear in the merged DataFrame.
existing_ticker_cols = []
for col in df_ticker.columns:
    if col in merged_df.columns:
        existing_ticker_cols.append(col)
    elif col + '_ticker' in merged_df.columns:
        existing_ticker_cols.append(col + '_ticker')
    else:
        print(f"Warning: Column '{col}' not found in merged DataFrame.")

# Now, order the merged DataFrame with ticker columns first.
other_cols = [
    col for col in merged_df.columns if col not in existing_ticker_cols]
merged_df = merged_df[existing_ticker_cols + other_cols]

# ----- Step 6: Write the Merged Data to the "DailyFinancials" Google Sheet -----
daily_sheet_name = "DailyFinancials"
try:
    daily_sheet = gc.open(daily_sheet_name)
except Exception as e:
    daily_sheet = gc.create(daily_sheet_name)
daily_ws = daily_sheet.sheet1
daily_ws.clear()  # Clear any existing data

set_with_dataframe(daily_ws, merged_df)
print("Merged financial data has been successfully written to the 'DailyFinancials' Google Sheet!")


# %% [markdown]
# #Adding in More Tickers

# %%
# ----- Step 1: Colab User Authentication -----
auth.authenticate_user()

# ----- Step 2: Initialize gspread with Default Credentials -----

creds, _ = default()
gc = gspread.authorize(creds)

# ----- Step 3: Open the Google Sheet -----
sheet_name = "CIK-Ticker"  # Your sheet name
try:
    sheet = gc.open(sheet_name)
except Exception as e:
    print("Could not open sheet:", e)
    exit(1)

worksheet = sheet.sheet1  # Adjust if you need a different worksheet

# ----- Step 4: Load Data from the Sheet into a DataFrame -----

df = get_as_dataframe(worksheet)

# Ensure the "Yahoo Ticker" column exists
if "Yahoo Ticker" not in df.columns:
    df["Yahoo Ticker"] = ""

# ----- Step 5: Define the Function to Query Yahoo Finance API -----


def get_yahoo_ticker(company_name):
    """
    Query Yahoo Finance's search API for the ticker.
    Returns the first ticker found or None if not found.
    """
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": company_name, "quotesCount": 1, "newsCount": 0}
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            quotes = data.get("quotes", [])
            if quotes:
                return quotes[0].get("symbol")
        return None
    except Exception as e:
        print(f"Error processing company '{company_name}': {e}")
        return None


# ----- Step 6: Process Each Company with a Progress Bar and Delay -----
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing companies"):
    company = row.get("Company Name_sec")
    if pd.notna(company):
        ticker = get_yahoo_ticker(company)
        df.at[idx, "Yahoo Ticker"] = ticker if ticker else "Not found"
    time.sleep(0.1)  # 0.1 second delay per API call

# ----- Step 7: Write the Updated Data Back to the Google Sheet -----
set_with_dataframe(worksheet, df)
print("Processing complete! The Google Sheet 'CIK-Ticker' has been updated with Yahoo Tickers.")


# %% [markdown]
# #Checking the Companies Printing Daily Financials

# %%

# ----- Step 1: Authenticate and Open the DailyFinancials Sheet -----
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

try:
    daily_sheet = gc.open("KAHLERS SHEET")
    worksheet = daily_sheet.sheet1
except Exception as e:
    raise Exception("Error opening DailyFinancials sheet: " + str(e))

# Load all values from the sheet.
data = worksheet.get_all_values()
if not data:
    raise ValueError("No data found in the DailyFinancials sheet.")

# Use the first row as headers; create unique header names if duplicates occur.
raw_headers = data[0]
unique_headers = []
counts = {}
for h in raw_headers:
    if h in counts:
        counts[h] += 1
        unique_headers.append(f"{h}_{counts[h]}")
    else:
        counts[h] = 0
        unique_headers.append(h)

# Create the DataFrame from the rest of the data.
df = pd.DataFrame(data[1:], columns=unique_headers)
print(f"Loaded DailyFinancials with {df.shape[0]} records.")
# Expected columns:
# Company Name_ticker, CIK, Company Name_sec, Company Name_sec_1, Industry,
# Current Assets, Total Assets, Accounts Receivable, Inventory, Prepaid Expenses,
# Total Liabilities, Stockholders Equity, Cash and Cash Equivalents, Net Income,
# PP&E, Equity Method Investments, Marketable Securities Current, Other Assets,
# Date, Land, Intangible Assets, marketcap

# ----- Step 2: Update with Ranking Metrics Using These Columns -----
# Add/update today's update date.
df['Update Date'] = datetime.today().strftime('%Y-%m-%d')

# Define the numeric columns (as provided).
numeric_columns = [
    'Current Assets', 'Total Assets', 'Accounts Receivable', 'Inventory',
    'Prepaid Expenses', 'Total Liabilities', 'Stockholders Equity',
    'Cash and Cash Equivalents', 'Net Income', 'PP&E', 'Equity Method Investments',
    'Marketable Securities Current', 'Other Assets', 'Land', 'Intangible Assets', 'marketcap'
]

# Fill missing values with 0 and convert to numeric.
df.fillna(0, inplace=True)
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert CIK to numeric.
if 'CIK' in df.columns:
    df['CIK'] = pd.to_numeric(df['CIK'], errors='coerce')

# Define industries to exclude from ranking calculations.
excluded_industries = [
    'Biotechnology',
    'Medical Devices',
    'Drug Manufacturers—Specialty & Generic',
    'Health Information Services'
]

# Create new flag columns with default value 0.
df['CASH_Net_Net'] = 0
df['Classic_Net_Net'] = 0
df['Property_Net_Net'] = 0
df['Big_Winners'] = 0

# Loop through each row to compute metrics and update flag columns.
for idx, row in df.iterrows():
    cash = float(row.get('Cash and Cash Equivalents', 0) or 0)
    ms = float(row.get('Marketable Securities Current', 0) or 0)
    emi = float(row.get('Equity Method Investments', 0) or 0)
    ppe = float(row.get('PP&E', 0) or 0)
    ar = float(row.get('Accounts Receivable', 0) or 0)
    inv = float(row.get('Inventory', 0) or 0)
    other_assets = float(row.get('Other Assets', 0) or 0)
    land = float(row.get('Land', 0) or 0)
    intangible = float(row.get('Intangible Assets', 0) or 0)
    total_liabilities = float(row.get('Total Liabilities', 0) or 0)
    stockholders_equity = float(row.get('Stockholders Equity', 0) or 0)
    net_income = float(row.get('Net Income', 0) or 0)
    market_cap = float(row.get('marketcap', 0) or 0)

    # Get industry and company name.
    industry = row.get('Industry', '')
    company_name = row.get('Company Name_ticker', 'Unknown')

    # Calculate full value capital.
    full_value_capital = cash + ms + emi

    # Calculate Ranking Metric (avoid division by zero).
    ranking_metric = ((0.1 * (ppe + ar + inv + other_assets + land + intangible))
                      + full_value_capital - total_liabilities) / market_cap if market_cap != 0 else 0

    # Compute flag conditions only if market_cap is valid and industry is not excluded.
    if pd.notnull(market_cap) and market_cap != 0 and (industry not in excluded_industries):
        if ((0.1 * (ppe + ar + inv + other_assets + land + intangible)) + full_value_capital - total_liabilities > market_cap) and (total_liabilities > 0) and (stockholders_equity > 0):
            df.at[idx, 'CASH_Net_Net'] = 1
        if (full_value_capital + (0.2 * (ppe + land)) + (0.66 * ar) + (0.5 * inv) + (0.1 * (other_assets + intangible)) - total_liabilities > market_cap) and (total_liabilities > 0) and (stockholders_equity > 0):
            df.at[idx, 'Classic_Net_Net'] = 1
        if (full_value_capital + (0.1 * intangible) + (0.75 * (ppe + land)) + (0.25 * ar) - total_liabilities > market_cap) and (total_liabilities > 0) and (stockholders_equity > 0):
            df.at[idx, 'Property_Net_Net'] = 1
        if ((0.2 * (ppe + ar + inv + other_assets)) + full_value_capital - total_liabilities > market_cap) and (net_income > 0) and (total_liabilities > 0) and (stockholders_equity > 0):
            df.at[idx, 'Big_Winners'] = 1

    # Save computed values.
    df.at[idx, 'full value capital'] = full_value_capital
    df.at[idx, 'Ranking_Metric'] = ranking_metric

# Calculate Rank based on Ranking_Metric (with 1 = highest rank).
df['Rank'] = df['Ranking_Metric'].rank(
    ascending=False, method='min').fillna(0).astype(int)

# ----- Step 3: Write Updated Data Back to the DailyFinancials Sheet -----
set_with_dataframe(worksheet, df)
print("DailyFinancials sheet updated with ranking metrics.")

# ----- Step 4: Interactive Widget for Data Filtering -----
# Define dropdown options for filtering based on flag columns.
dropdown_options = [
    ('Cash Net Net', 'cash_net_net'),
    ('Classic Net Net', 'classic_net_net_only'),
    ('Property Net Net', 'property_net_net_only'),
    ('Big Winners', 'big_winners_only'),
    ('Classic Net Net and Property Net Net', 'classic_and_property_net_net'),
    ('Everything', 'everything')
]

dropdown = widgets.Dropdown(
    options=dropdown_options,
    value='everything',
    description='Select Filter:'
)

ticker_input = widgets.Text(
    value='',
    description='Search Ticker:',
    placeholder='Enter company name or ticker'
)

output = widgets.Output()


def format_thousands(x):
    try:
        return f'{x/1000:,.0f}' if pd.notnull(x) else x
    except Exception:
        return x


def update_table(change):
    with output:
        clear_output()
        option = dropdown.value
        ticker = ticker_input.value.strip().upper()

        filtered_df = df.copy()
        if option == 'cash_net_net':
            filtered_df = filtered_df[filtered_df['CASH_Net_Net'] == 1]
        elif option == 'classic_net_net_only':
            filtered_df = filtered_df[
                (filtered_df['Classic_Net_Net'] == 1) &
                (filtered_df['CASH_Net_Net'] == 0) &
                (filtered_df['Property_Net_Net'] == 0) &
                (filtered_df['Big_Winners'] == 0)
            ]
        elif option == 'property_net_net_only':
            filtered_df = filtered_df[
                (filtered_df['Property_Net_Net'] == 1) &
                (filtered_df['Classic_Net_Net'] == 0) &
                (filtered_df['CASH_Net_Net'] == 0) &
                (filtered_df['Big_Winners'] == 0)
            ]
        elif option == 'big_winners_only':
            filtered_df = filtered_df[
                (filtered_df['Big_Winners'] == 1) &
                (filtered_df['Classic_Net_Net'] == 0) &
                (filtered_df['CASH_Net_Net'] == 0) &
                (filtered_df['Property_Net_Net'] == 0)
            ]
        elif option == 'classic_and_property_net_net':
            filtered_df = filtered_df[
                (filtered_df['Classic_Net_Net'] == 1) &
                (filtered_df['Property_Net_Net'] == 1) &
                (filtered_df['CASH_Net_Net'] == 0) &
                (filtered_df['Big_Winners'] == 0)
            ]
        # "Everything" returns the full DataFrame.

        if ticker:
            filtered_df = filtered_df[filtered_df['Company Name_ticker'].astype(
                str).str.upper().str.contains(ticker, na=False)]

        filtered_df = filtered_df.sort_values(
            by='Ranking_Metric', ascending=False)
        filtered_df['Rank'] = filtered_df['Ranking_Metric'].rank(
            ascending=False, method='min').fillna(0).astype(int)

        display_columns = [
            'Update Date', 'marketcap', 'Company Name_ticker', 'CIK', 'Ranking_Metric', 'Industry',
            'Current Assets', 'Total Assets', 'Accounts Receivable', 'Inventory', 'Prepaid Expenses',
            'Total Liabilities', 'Stockholders Equity', 'Cash and Cash Equivalents', 'Net Income',
            'PP&E', 'Equity Method Investments', 'Marketable Securities Current'
        ]
        numeric_cols = [
            'marketcap', 'Current Assets', 'Total Assets', 'Accounts Receivable', 'Inventory',
            'Prepaid Expenses', 'Total Liabilities', 'Stockholders Equity', 'Cash and Cash Equivalents',
            'Net Income', 'PP&E', 'Equity Method Investments', 'Marketable Securities Current'
        ]
        formatted_df = filtered_df[['Rank'] + display_columns].copy()
        formatted_df[numeric_cols] = formatted_df[numeric_cols].applymap(
            format_thousands)
        display(formatted_df)


dropdown.observe(update_table, names='value')
ticker_input.observe(update_table, names='value')

ui = widgets.VBox([dropdown, ticker_input, output])
display(ui)
update_table(None)


# %% [markdown]
# #All Garbage below here
#

# %%

# ==================== 1) MOUNT GOOGLE DRIVE AND LOAD MASTER FILE ====================
drive.mount('/content/drive')

# Define the path to your folder in Google Drive
folder_path = '/content/drive/My Drive/Fund LOL'

# Path to master file in Drive
master_file_path = os.path.join(folder_path, 'AllSecDATA.csv')

# Load the master file
master_df = pd.read_csv(master_file_path)

# Ensure the master CIK is zero-padded
master_df['CIK'] = master_df['CIK'].astype(str).str.zfill(10)

# Identify the date column in the master file (Date, Last Update, or Date Filed)
if 'Last Update' in master_df.columns:
    master_df['Last Update'] = pd.to_datetime(
        master_df['Last Update'], errors='coerce')
    date_column = 'Last Update'
elif 'Date' in master_df.columns:
    master_df['Date'] = pd.to_datetime(master_df['Date'], errors='coerce')
    date_column = 'Date'
elif 'Date Filed' in master_df.columns:
    master_df['Date Filed'] = pd.to_datetime(
        master_df['Date Filed'], errors='coerce')
    date_column = 'Date Filed'
else:
    raise ValueError(
        "Master file must contain a 'Last Update', 'Date', or 'Date Filed' column.")

print("Master file loaded. Shape:", master_df.shape)

# ==================== 2) LOAD THE MISSING INFO DATA ====================
missing_info_csv = 'missing_info.csv'
df_missing = pd.read_csv(missing_info_csv)
df_missing['CIK'] = df_missing['CIK'].astype(str).str.zfill(10)

# Identify which column in df_missing is the date field
if 'Date Filed_new' in df_missing.columns:
    df_missing['Date Filed_new'] = pd.to_datetime(
        df_missing['Date Filed_new'], errors='coerce')
    date_col_missing = 'Date Filed_new'
elif 'Date Filed' in df_missing.columns:
    df_missing['Date Filed'] = pd.to_datetime(
        df_missing['Date Filed'], errors='coerce')
    date_col_missing = 'Date Filed'
else:
    raise ValueError(
        "Could not find a 'Date Filed' or 'Date Filed_new' column in missing_info.csv.")

print("Missing info CSV loaded. Shape:", df_missing.shape)

# ==================== 3) EDGAR FETCH UTILITIES (WITH DELAY) ====================


def get_company_concept_data(cik, taxonomy, concept):
    """
    For a given CIK, taxonomy (e.g., 'us-gaap'), and concept name,
    fetch the XBRL JSON data from the SEC EDGAR API.
    We add a short delay to avoid hitting rate limits.
    """
    url = f'https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{concept}.json'
    headers = {
        # Replace with your details
        'User-Agent': 'Zach Kinzler (zkinzler@sandiego.edu)'
    }
    response = requests.get(url, headers=headers)
    # Delay between requests
    time.sleep(0.1)

    if response.status_code == 200:
        return response.json()
    else:
        return None


def extract_financial_data(data):
    """
    From the JSON data, return a list of (end_date, value) for
    10-Q forms only, sorted by newest 'end' date.
    """
    if not data:
        return []

    facts = data.get('units', {}).get('USD', [])
    # Only keep items from 10-Q
    facts_10q = [fact for fact in facts if fact.get('form', '') == '10-Q']
    # Sort descending by end date
    facts_10q.sort(key=lambda x: x['end'], reverse=True)
    # Return only the first (newest) record if available
    return [(fact['end'], fact['val']) for fact in facts_10q[:1]]


# ==================== 4) BALANCE SHEET CONCEPTS ====================
extra_ppe_words = [
    "PropertyPlantAndEquipmentGross", "RealEstateInvestmentPropertyNet",
    "InventoryRealEstateHeldForSale", "PublicUtilitiesPropertyPlantAndEquipmentNet",
    "OperatingLeaseRightOfUseAsset"
]
inventory_words = [
    "InventoryRawMaterialsAndSuppliesNetOfReserves", "InventoryGross", "InventoryFinishedGoods",
    "InventoryWorkInProcessAndRawMaterials", "InventoryRawMaterialsAndSupplies", "InventoryMerchandise"
]
land_words = [
    "Land", "TimberAndTimberlands", "RealEstateAssets", "LandAndLandImprovements",
    "LandHeldForDevelopment", "AreaOfLand"
]
intangible_assets_words = [
    "IntangibleAssetsCurrent", "IntangibleAssetsNet", "FiniteLivedPatentsGross",
    "AmortizationOfIntangibleAssets", "FiniteLivedIntangibleAssetsGross"
]

balance_sheet_concepts = {
    'AssetsCurrent': ['AssetsCurrent'],
    'Assets': ['Assets'],
    'Liabilities': ['Liabilities', 'LiabilitiesCurrent'],
    'StockholdersEquity': ['StockholdersEquity'],
    'CashAndCashEquivalents': [
        'CashAndCashEquivalentsAtCarryingValue', 'Cash',
        'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'
    ],
    'AccountsReceivableNetCurrent': ['AccountsReceivableNetCurrent'],
    'InventoryNet': ['InventoryNet'] + inventory_words,
    'PrepaidExpenseAndOtherAssetsCurrent': [
        'PrepaidExpenseAndOtherAssetsCurrent', 'PrepaidExpenseCurrent',
        'PrepaidExpensesAndOtherCurrentAssets', 'OtherReceivablesCurrent'
    ],
    'PropertyPlantAndEquipmentNet': ['PropertyPlantAndEquipmentNet'] + extra_ppe_words,
    'EquityMethodInvestments': ['EquityMethodInvestments'],
    'MarketableSecuritiesCurrent': ['MarketableSecuritiesCurrent'],
    'OtherAssets': ['OtherAssets', 'OtherAssetsCurrent', 'OtherAssetsNoncurrent'],
    'Land': land_words,
    'IntangibleAssets': intangible_assets_words
}

net_income_concepts = [
    'NetIncomeLoss', 'NetIncome', 'NetIncomeAvailableToCommonStockholdersBasic'
]

# ==================== 5) GATHER NEW FINANCIAL DATA FOR MISSING CIKs ====================
new_company_data = []  # Will accumulate new rows until flushed

print("Pulling updated 10-Q data from EDGAR for missing companies...")
for idx, row in tqdm(df_missing.iterrows(), total=len(df_missing)):
    cik_number = row['CIK']

    # Local placeholders for each concept
    current_assets = None
    total_assets = None
    liabilities = None
    stockholders_equity = None
    cash_and_cash_equivalents = None
    accounts_receivable = None
    inventory = None
    prepaid_expenses = None
    ppe = None
    equity_method_investments = None
    marketable_securities_current = None
    other_assets = None
    land = None
    intangible_assets = None
    net_income = None

    # (A) Fetch each balance sheet concept
    for concept_name, tags in balance_sheet_concepts.items():
        for tag in tags:
            data = get_company_concept_data(cik_number, 'us-gaap', tag)
            financial_data = extract_financial_data(data)
            if financial_data:
                # We'll use only the single newest (end_date, value)
                _, val = financial_data[0]
                if concept_name == 'AssetsCurrent':
                    current_assets = val
                elif concept_name == 'Assets':
                    total_assets = val
                elif concept_name == 'Liabilities':
                    liabilities = val
                elif concept_name == 'StockholdersEquity':
                    stockholders_equity = val
                elif concept_name == 'CashAndCashEquivalents':
                    cash_and_cash_equivalents = val
                elif concept_name == 'AccountsReceivableNetCurrent':
                    accounts_receivable = val
                elif concept_name == 'InventoryNet':
                    inventory = val
                elif concept_name == 'PrepaidExpenseAndOtherAssetsCurrent':
                    prepaid_expenses = val
                elif concept_name == 'PropertyPlantAndEquipmentNet':
                    ppe = val
                elif concept_name == 'EquityMethodInvestments':
                    equity_method_investments = val
                elif concept_name == 'MarketableSecuritiesCurrent':
                    marketable_securities_current = val
                elif concept_name == 'OtherAssets':
                    other_assets = val
                elif concept_name == 'Land':
                    land = val
                elif concept_name == 'IntangibleAssets':
                    intangible_assets = val
                break  # Found data for this concept, move to next

    # (B) Fetch net income
    for concept in net_income_concepts:
        data = get_company_concept_data(cik_number, 'us-gaap', concept)
        financial_data = extract_financial_data(data)
        if financial_data:
            net_income = financial_data[0][1]
            break

    # (C) Build the new data row
    new_row = {
        'CIK': cik_number,
        'Date Filed': row[date_col_missing],
        'Company Name': row.get('Company Name_new', row.get('Company Name', 'Unknown')),
        'Current Assets': current_assets,
        'Total Assets': total_assets,
        'Accounts Receivable': accounts_receivable,
        'Inventory': inventory,
        'Prepaid Expenses': prepaid_expenses,
        'Total Liabilities': liabilities,
        'Stockholders Equity': stockholders_equity,
        'Cash and Cash Equivalents': cash_and_cash_equivalents,
        'Net Income': net_income,
        'PP&E': ppe,
        'Equity Method Investments': equity_method_investments,
        'Marketable Securities Current': marketable_securities_current,
        'Other Assets': other_assets,
        'Land': land,
        'Intangible Assets': intangible_assets
    }

    new_company_data.append(new_row)

    # Flush to master file every 100 entries
    if (idx + 1) % 100 == 0:
        print(
            f"Processed {idx+1} entries. Updating master file with current batch...")
        df_new_chunk = pd.DataFrame(new_company_data)
        df_new_chunk.fillna(0, inplace=True)
        # Convert 'Date Filed' to datetime
        df_new_chunk['Date Filed'] = pd.to_datetime(
            df_new_chunk['Date Filed'], errors='coerce')
        # If the master file uses a different date column, rename accordingly
        if date_column != 'Date Filed':
            df_new_chunk.rename(
                columns={'Date Filed': date_column}, inplace=True)
        # Reload the latest master file from Drive
        master_df = pd.read_csv(master_file_path)
        master_df['CIK'] = master_df['CIK'].astype(str).str.zfill(10)
        master_df[date_column] = pd.to_datetime(
            master_df[date_column], errors='coerce')
        # Combine and update master file
        combined_df = pd.concat([master_df, df_new_chunk], ignore_index=True)
        combined_df.sort_values(
            by=[date_column], ascending=False, inplace=True)
        combined_df.drop_duplicates(
            subset=['CIK', date_column], keep='first', inplace=True)
        combined_df.to_csv(master_file_path, index=False)
        print(f"Master file updated after {idx+1} entries.")
        # Reset the new_company_data list
        new_company_data = []

print("Finished pulling new data from EDGAR.")

# Flush any remaining new data that hasn't been written yet
if new_company_data:
    print("Flushing final batch of new data to master file...")
    df_new_chunk = pd.DataFrame(new_company_data)
    df_new_chunk.fillna(0, inplace=True)
    df_new_chunk['Date Filed'] = pd.to_datetime(
        df_new_chunk['Date Filed'], errors='coerce')
    if date_column != 'Date Filed':
        df_new_chunk.rename(columns={'Date Filed': date_column}, inplace=True)
    master_df = pd.read_csv(master_file_path)
    master_df['CIK'] = master_df['CIK'].astype(str).str.zfill(10)
    master_df[date_column] = pd.to_datetime(
        master_df[date_column], errors='coerce')
    combined_df = pd.concat([master_df, df_new_chunk], ignore_index=True)
    combined_df.sort_values(by=[date_column], ascending=False, inplace=True)
    combined_df.drop_duplicates(
        subset=['CIK', date_column], keep='first', inplace=True)
    combined_df.to_csv(master_file_path, index=False)
    print("Final master file update complete.")

# ==================== END OF SCRIPT ====================


# %% [markdown]
# #Updating CIK-Ticker file with tickers
#

# %%
!pip install yahooquery


# %%

# ----- Step 1: Authenticate & Mount Drive -----
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
drive.mount('/content/drive')

# ----- Step 2: Open the CIK-Ticker Google Sheet -----
sheet_name = "CIK-Ticker"
sheet = gc.open(sheet_name)
ws = sheet.sheet1

# ----- Step 3: Load the CIK-Ticker sheet into a DataFrame -----
df_ticker = pd.DataFrame(ws.get_all_records())
print(f"CIK-Ticker sheet loaded with {df_ticker.shape[0]} rows.")

# ----- Step 4: Define the Columns for Ticker and Company Name for Search -----
# "Company Name" holds the ticker; "Company Name_sec" is used for the search query.
ticker_col = "Company Name"
company_name_col = "Company Name_sec"

# ----- Step 5: Iterate Through Rows & Update Tickers Without Adding Duplicates -----
for idx, row in tqdm(df_ticker.iterrows(), total=df_ticker.shape[0], desc="Updating tickers"):
    current_ticker = str(row[ticker_col]).strip().upper()
    # Process only rows marked as "Ticker not found"
    if current_ticker != "TICKER NOT FOUND":
        continue

    search_query = str(row[company_name_col]).strip()
    if not search_query:
        continue  # Skip if the search query is empty

    try:
        results = search(search_query, quotes_count=1, news_count=0)
        if results and "quotes" in results:
            quotes = results["quotes"]
            if quotes:
                best_match = quotes[0]
                found_ticker = best_match.get("symbol", None)
                if found_ticker:
                    found_ticker = found_ticker.upper()
                    # Check if the found ticker already exists in other rows
                    other_tickers = df_ticker.loc[df_ticker.index != idx, ticker_col].str.upper(
                    ).tolist()
                    if found_ticker in other_tickers:
                        print(
                            f"Duplicate ticker '{found_ticker}' found for '{search_query}' at row {idx}; skipping update.")
                    else:
                        print(
                            f"Found ticker for '{search_query}': {found_ticker} at row {idx}.")
                        df_ticker.at[idx, ticker_col] = found_ticker
                else:
                    print(
                        f"No valid ticker found for '{search_query}' at row {idx}.")
            else:
                print(f"No quotes found for '{search_query}' at row {idx}.")
        else:
            print(f"No results returned for '{search_query}' at row {idx}.")
    except Exception as e:
        print(f"Error searching Yahoo for '{search_query}' at row {idx}: {e}")

    # Short delay to respect potential Yahoo Finance rate limits
    time.sleep(0.01)

# ----- Step 6: Write the Updated DataFrame Back to the CIK-Ticker Sheet -----
ws.clear()  # Clear existing content
set_with_dataframe(ws, df_ticker)
print("Updated CIK-Ticker sheet with new ticker data.")


# %% [markdown]
# #Adding Industry

# %%

# ----- Step 1: Authenticate & Mount Drive -----
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
drive.mount('/content/drive')

# ----- Step 2: Open the CIK-Ticker Google Sheet -----
sheet_name = "CIK-Ticker"
sheet = gc.open(sheet_name)
ws = sheet.sheet1

# ----- Step 3: Load the Google Sheet Data into a DataFrame -----
df_ticker = pd.DataFrame(ws.get_all_records())
print(f"Loaded sheet with {df_ticker.shape[0]} rows.")

# ----- Step 4: Define the Ticker Column & Add the Industry Column -----
# "Company Name" holds the ticker. We add a new column "Industry" if it doesn't exist.
ticker_col = "Company Name"
if "Industry" not in df_ticker.columns:
    df_ticker["Industry"] = ""

# ----- Step 5: Iterate Through Each Row to Fetch Industry Data -----
for idx, row in df_ticker.iterrows():
    ticker = str(row[ticker_col]).strip()

    # Skip rows with an empty ticker or with "Ticker not found"
    if ticker == "" or ticker.upper() == "TICKER NOT FOUND":
        continue

    try:
        # Create a Ticker object for the given ticker
        yticker = Ticker(ticker)
        # Retrieve the asset profile which contains the industry information
        profile = yticker.asset_profile.get(ticker, None)
        if profile:
            industry = profile.get("industry", "")
            df_ticker.at[idx, "Industry"] = industry
            print(f"Row {idx}: Ticker {ticker} - Industry: {industry}")
        else:
            print(f"Row {idx}: No asset profile found for ticker {ticker}.")
    except Exception as e:
        print(f"Error fetching industry for ticker {ticker} at row {idx}: {e}")

    # Short delay to respect potential rate limits
    time.sleep(0.01)

# ----- Step 6: Write the Updated DataFrame Back to the Google Sheet -----
ws.clear()  # Clear existing content
set_with_dataframe(ws, df_ticker)
print("Updated CIK-Ticker sheet with Industry data.")


# %% [markdown]
# #Yahoo Finance Step

# %%

# ---------------------- Configuration ----------------------
# Mount Google Drive
drive.mount('/content/drive')

# The SEC base file (containing your SEC info) is stored locally in Colab
sec_master_path = '/content/already_have_info.csv'

# Verify that the SEC file exists
if not os.path.exists(sec_master_path):
    raise FileNotFoundError(
        f"The SEC master file was not found at {sec_master_path}. Please upload it to /content/.")

# Folder in Google Drive where the combined master file will be saved
folder_path = '/content/drive/My Drive/Fund LOL'
master_file_path = os.path.join(
    folder_path, 'combined2_final_master_with_yahoo.csv')

# Batch size for iterative updates
batch_size = 15

# ---------------------- STEP 1: Load SEC Data ----------------------
# Read the SEC file (base SEC info) once and store it in memory.
df_sec_all = pd.read_csv(sec_master_path)
df_sec_all['CIK'] = df_sec_all['CIK'].astype(str).str.zfill(10)

# Sort by 'Date Filed' (if available) so that the most recent filing is used per CIK
if 'Date Filed' in df_sec_all.columns:
    df_sec_all['Date Filed'] = pd.to_datetime(
        df_sec_all['Date Filed'], errors='coerce')
    df_sec_all.sort_values(by='Date Filed', ascending=False, inplace=True)
else:
    print("Warning: 'Date Filed' column not found; skipping date sort.")

# Keep the most recent filing per CIK
df_most_recent = df_sec_all.drop_duplicates(subset='CIK', keep='first').copy()
print(
    f"SEC data loaded and filtered: {df_most_recent.shape[0]} most recent rows.")

# ---------------------- STEP 2: Define Yahoo Fetching Functions ----------------------
excluded_industries = [
    'Biotechnology',
    'Medical Devices',
    'Drug Manufacturers—Specialty & Generic',
    'Health Information Services'
]


def get_additional_info(ticker, retries=3, delay=0.2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            sector = stock.info.get('sector', 'N/A')
            full_time_employees = stock.info.get('fullTimeEmployees', 'N/A')
            current_price = stock.info.get('currentPrice', 'N/A')
            pe_ratio = stock.info.get('trailingPE', 'N/A')
            country = stock.info.get('country', 'N/A')
            return sector, full_time_employees, current_price, pe_ratio, country
        except Exception as e:
            print(f"Error fetching additional info for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'


def validate_ticker(ticker, retries=3, delay=0.2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap', None)
            return market_cap is not None
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
    return False


def get_market_cap(ticker, retries=3, delay=0.2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('marketCap', 'N/A')
        except Exception as e:
            print(f"Error fetching market cap for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return 'N/A'


def get_industry(ticker, retries=3, delay=0.2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('industry', 'N/A')
        except Exception as e:
            print(f"Error fetching industry for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return 'N/A'


def get_ticker_from_sec(cik, retries=3, delay=0.2):
    base_url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
    headers = {'User-Agent': 'Your Name (YourEmail@example.com)'}
    for attempt in range(retries):
        try:
            response = requests.get(base_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            tickers = data.get('tickers', [])
            if tickers:
                return tickers[0]
        except HTTPError as e:
            if response.status_code == 429:
                print(
                    f"Rate limit exceeded for {cik}. Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"HTTP error for {cik}: {e}")
                break
        except Exception as e:
            print(f"An error occurred for {cik}: {e}")
            break
        if attempt < retries - 1:
            time.sleep(delay)
    return None


# ---------------------- STEP 3: Iterative Yahoo Data Collection & Merging ----------------------
# Load existing master file if it exists (this file accumulates historical records).
if os.path.exists(master_file_path):
    combined_master = pd.read_csv(master_file_path)
    combined_master['CIK'] = combined_master['CIK'].astype(str).str.zfill(10)
else:
    combined_master = pd.DataFrame()

# List to accumulate Yahoo data in the current batch.
batch_enriched_companies = []

# Process each SEC row with a progress bar.
for i, (idx, row) in enumerate(tqdm(df_most_recent.iterrows(), total=df_most_recent.shape[0], desc="Processing CIKs"), start=1):
    cik = row['CIK']
    ticker = get_ticker_from_sec(cik)
    time.sleep(0.2)  # Delay between SEC API calls

    if ticker and validate_ticker(ticker):
        market_cap = get_market_cap(ticker)
        industry = get_industry(ticker)
        sector, full_time_employees, current_price, pe_ratio, country = get_additional_info(
            ticker)

        # Skip companies from China or in excluded industries.
        if country == "China" or industry in excluded_industries:
            continue

        batch_enriched_companies.append({
            'CIK': cik,
            'Ticker': ticker,
            'Market Cap': market_cap,
            'Industry': industry,
            'Sector': sector,
            'Full-Time Employees': full_time_employees,
            'Current Price': current_price,
            'P/E Ratio': pe_ratio,
            'Country': country,
            'Yahoo Pull Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    time.sleep(0.8)  # Delay to avoid Yahoo rate limits

    # Every batch_size entries (or at the end), merge the Yahoo batch with the SEC info.
    if i % batch_size == 0 or i == df_most_recent.shape[0]:
        if batch_enriched_companies:
            # Convert the current batch to a DataFrame.
            df_batch = pd.DataFrame(batch_enriched_companies)

            # Instead of re-reading the SEC file, use the already loaded SEC data.
            # Merge the SEC info (from df_most_recent) with the new Yahoo batch on CIK.
            merged_batch = pd.merge(
                df_most_recent, df_batch, on='CIK', how='inner', suffixes=('_sec', '_yahoo'))
            # Use inner join so that we only merge for companies we have new Yahoo data for.

            # Append the new merged batch to the existing combined master file.
            combined_master = pd.concat(
                [combined_master, merged_batch], ignore_index=True)

            # Save the updated master file back to Google Drive.
            combined_master.to_csv(master_file_path, index=False)
            print(
                f"Batch ending at entry {i} merged and saved. Total records: {combined_master.shape[0]}")

            # Clear the batch list for the next set of entries.
            batch_enriched_companies = []

print(f"Final combined master file saved to {master_file_path}.")


# %% [markdown]
# #Checking new try

# %%
# Install gspread if not already installed (uncomment if needed)
# !pip install --upgrade gspread


# ---------------------- AUTHENTICATION & DRIVE MOUNT ----------------------
# Authenticate for Google Sheets access
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

# Mount Google Drive
drive.mount('/content/drive')

# ---------------------- CONFIGURATION ----------------------
# Path to your SEC master CSV (ensure it's uploaded to /content/)
sec_master_path = '/content/already_have_info.csv'
if not os.path.exists(sec_master_path):
    raise FileNotFoundError(
        f"SEC master file not found at {sec_master_path}. Please upload it to /content/.")

# Google Sheet configuration:
sheet_name = "MasterFile"  # Desired name for your master Google Sheet

# Try to open the spreadsheet; if it doesn't exist, create one.
try:
    spreadsheet = gc.open(sheet_name)
    print(f"Opened existing spreadsheet: {sheet_name}")
except gspread.SpreadsheetNotFound:
    print(f"Spreadsheet '{sheet_name}' not found. Creating a new one...")
    spreadsheet = gc.create(sheet_name)
    # Optionally, move the new spreadsheet to your desired Drive folder.
    print(f"Created new spreadsheet: {sheet_name}")

# Use the first worksheet (or adjust if you want a specific tab)
worksheet = spreadsheet.sheet1

# Batch size for logging
batch_size = 15

# ---------------------- STEP 1: LOAD AND PROCESS SEC DATA ----------------------
df_sec_all = pd.read_csv(sec_master_path)
df_sec_all['CIK'] = df_sec_all['CIK'].astype(str).str.zfill(10)

if 'Date Filed' in df_sec_all.columns:
    df_sec_all['Date Filed'] = pd.to_datetime(
        df_sec_all['Date Filed'], errors='coerce')
    df_sec_all.sort_values(by='Date Filed', ascending=False, inplace=True)
else:
    print("Warning: 'Date Filed' column not found; skipping date sort.")

# Keep the most recent filing per CIK
df_most_recent = df_sec_all.drop_duplicates(subset='CIK', keep='first').copy()
print(f"SEC data loaded and filtered: {df_most_recent.shape[0]} rows.")

# ---------------------- STEP 2: DEFINE THE SEC TICKER FETCH FUNCTION ----------------------


def get_ticker_from_sec(cik, retries=3, delay=0.2):
    base_url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
    # Update with your details!
    headers = {'User-Agent': 'Your Name (YourEmail@example.com)'}
    for attempt in range(retries):
        try:
            response = requests.get(base_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            tickers = data.get('tickers', [])
            if tickers:
                return tickers[0]
        except HTTPError as e:
            if response.status_code == 429:
                print(
                    f"Rate limit exceeded for CIK {cik}. Retrying after {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"HTTP error for CIK {cik}: {e}")
                break
        except Exception as e:
            print(f"An error occurred for CIK {cik}: {e}")
            break
        if attempt < retries - 1:
            time.sleep(delay)
    return None


# ---------------------- STEP 3: PROCESS SEC DATA & APPEND TICKER ----------------------
new_data = []
for i, (idx, row) in enumerate(df_most_recent.iterrows(), start=1):
    cik = row['CIK']
    ticker = get_ticker_from_sec(cik)
    time.sleep(0.2)  # Respect SEC rate limits
    if ticker:
        company_data = row.to_dict()
        company_data['Ticker'] = ticker
        company_data['SEC Pull Date'] = datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S')
        new_data.append(company_data)
    if i % batch_size == 0 or i == df_most_recent.shape[0]:
        print(f"Processed {i} of {df_most_recent.shape[0]} records.")

final_df = pd.DataFrame(new_data)
print(f"Final data prepared: {final_df.shape[0]} records.")

# ---------------------- STEP 4: MERGE WITH EXISTING GOOGLE SHEET DATA ----------------------
# Get all existing data from the Google Sheet
existing_data = worksheet.get_all_values()
if existing_data:
    header = existing_data[0]
    existing_df = pd.DataFrame(existing_data[1:], columns=header)
    if "Date Filed" in existing_df.columns:
        existing_df["Date Filed"] = pd.to_datetime(
            existing_df["Date Filed"], errors='coerce')
else:
    existing_df = pd.DataFrame(columns=final_df.columns)

# Ensure "Date Filed" in new data is datetime
if "Date Filed" in final_df.columns:
    final_df["Date Filed"] = pd.to_datetime(
        final_df["Date Filed"], errors='coerce')

# Combine existing and new data, keeping the record with the latest filing date per CIK
combined_df = pd.concat([existing_df, final_df], ignore_index=True)
if "CIK" in combined_df.columns and "Date Filed" in combined_df.columns:
    combined_df.sort_values(by='Date Filed', ascending=False, inplace=True)
    merged_df = combined_df.drop_duplicates(subset='CIK', keep='first')
else:
    merged_df = final_df.copy()

merged_df.sort_values(by='CIK', inplace=True)

# ---------------------- STEP 5: UPDATE GOOGLE SHEET ----------------------
# Clear existing sheet data
worksheet.clear()

# Prepare data for updating the sheet: header row plus data rows
data_to_update = [merged_df.columns.tolist()] + \
    merged_df.astype(str).values.tolist()
worksheet.update('A1', data_to_update)

print("Google Sheet master file updated successfully!")


# %% [markdown]
# #Add in Ranking metrics and things

# %%

# ---------------------- Mount Google Drive ----------------------
drive.mount('/content/drive', force_remount=True)

# ---------------------- Load SEC File ----------------------
sec_master_path = '/content/already_have_info.csv'
if not os.path.exists(sec_master_path):
    raise FileNotFoundError(f"SEC master file not found at {sec_master_path}")

df_sec = pd.read_csv(sec_master_path)

# Use "Company Name_new" if available; otherwise, fallback to "Company Name"
if 'Company Name_new' in df_sec.columns:
    df_sec['Company_Name'] = df_sec['Company Name_new']
elif 'Company Name' in df_sec.columns:
    df_sec['Company_Name'] = df_sec['Company Name']
else:
    raise ValueError("No company name column found in the SEC file.")

# ---------------------- Function to Get Ticker from Yahoo ----------------------


def get_ticker_from_company_name(company_name, retries=3, delay=1.0):
    """
    Uses Yahoo's unofficial search API to find a ticker based on the company name.
    Returns the first matching ticker, or None if no match is found.
    """
    base_url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": company_name, "quotesCount": 1, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0"}  # Generic UA string
    for attempt in range(retries):
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            quotes = data.get("quotes", [])
            if quotes and len(quotes) > 0:
                ticker = quotes[0].get("symbol")
                return ticker
        except Exception as e:
            print(f"Error retrieving ticker for '{company_name}': {e}")
            time.sleep(delay)
    return None


# ---------------------- Iterate Over Companies to Get Tickers ----------------------
results = []

for idx, row in tqdm(df_sec.iterrows(), total=len(df_sec), desc="Processing companies"):
    company_name = row["Company_Name"]
    ticker = get_ticker_from_company_name(company_name)
    if ticker:
        print(f"Found ticker for '{company_name}': {ticker}")
    else:
        print(f"No ticker found for '{company_name}'")
    results.append({
        "Company_Name": company_name,
        "Ticker": ticker
    })
    time.sleep(0.01)  # Delay between API calls to avoid rate limits

# Convert results to a DataFrame and show a preview
df_results = pd.DataFrame(results)
print(df_results.head())

# ---------------------- Save Ticker Mapping ----------------------
output_path = "/content/drive/My Drive/Fund LOL/tickers_by_company.csv"
df_results.to_csv(output_path, index=False)
print(f"Ticker mapping saved to {output_path}")


# %%

# ---------------------- Configuration ----------------------
# Mount Google Drive
drive.mount('/content/drive')

# The SEC base file (containing your SEC info) is stored locally in Colab
sec_master_path = '/content/already_have_info.csv'
if not os.path.exists(sec_master_path):
    raise FileNotFoundError(
        f"The SEC master file was not found at {sec_master_path}. Please upload it to /content/.")

# Folder in Google Drive where the combined master file will be saved
folder_path = '/content/drive/My Drive/Fund LOL'
master_file_path = os.path.join(
    folder_path, 'combined2_final_master_with_yahoo.csv')

# Batch size for iterative updates
batch_size = 15

# ---------------------- STEP 1: Load SEC Data ----------------------
df_sec_all = pd.read_csv(sec_master_path)
df_sec_all['CIK'] = df_sec_all['CIK'].astype(str).str.zfill(10)

if 'Date Filed' in df_sec_all.columns:
    df_sec_all['Date Filed'] = pd.to_datetime(
        df_sec_all['Date Filed'], errors='coerce')
    df_sec_all.sort_values(by='Date Filed', ascending=False, inplace=True)
else:
    print("Warning: 'Date Filed' column not found; skipping date sort.")

df_most_recent = df_sec_all.drop_duplicates(subset='CIK', keep='first').copy()
print(
    f"SEC data loaded and filtered: {df_most_recent.shape[0]} most recent rows.")

# ---------------------- STEP 2: Define Yahoo Fetching Functions ----------------------
excluded_industries = [
    'Biotechnology',
    'Medical Devices',
    'Drug Manufacturers—Specialty & Generic',
    'Health Information Services'
]


def get_additional_info(ticker, retries=3, delay=0.2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            sector = stock.info.get('sector', 'N/A')
            full_time_employees = stock.info.get('fullTimeEmployees', 'N/A')
            current_price = stock.info.get('currentPrice', 'N/A')
            pe_ratio = stock.info.get('trailingPE', 'N/A')
            country = stock.info.get('country', 'N/A')
            return sector, full_time_employees, current_price, pe_ratio, country
        except Exception as e:
            print(f"Error fetching additional info for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'


def validate_ticker(ticker, retries=3, delay=0.2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            market_cap = stock.info.get('marketCap', None)
            return market_cap is not None
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay)
    return False


def get_market_cap(ticker, retries=3, delay=0.2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('marketCap', 'N/A')
        except Exception as e:
            print(f"Error fetching market cap for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return 'N/A'


def get_industry(ticker, retries=3, delay=0.2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('industry', 'N/A')
        except Exception as e:
            print(f"Error fetching industry for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return 'N/A'


def get_ticker_from_company_name(company_name, retries=3, delay=0.2):
    """
    Uses Yahoo's unofficial search API to find a ticker based on the company name.
    Returns the first matching ticker, or None if no match is found.
    """
    base_url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": company_name, "quotesCount": 1, "newsCount": 0}
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries):
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            quotes = data.get("quotes", [])
            if quotes:
                return quotes[0].get("symbol")
        except Exception as e:
            print(f"Error retrieving ticker for '{company_name}': {e}")
            if attempt < retries - 1:
                time.sleep(delay)
    return None


# ---------------------- STEP 3: Iterative Yahoo Data Collection & Merging ----------------------
if os.path.exists(master_file_path):
    combined_master = pd.read_csv(master_file_path)
    combined_master['CIK'] = combined_master['CIK'].astype(str).str.zfill(10)
else:
    combined_master = pd.DataFrame()

batch_enriched_companies = []

for i, (idx, row) in enumerate(tqdm(df_most_recent.iterrows(), total=df_most_recent.shape[0], desc="Processing Companies"), start=1):
    # Use "Company Name_new" if available; otherwise, fallback to "Company Name"
    if 'Company Name_new' in row and pd.notnull(row['Company Name_new']):
        company_name = row['Company Name_new']
    elif 'Company Name' in row and pd.notnull(row['Company Name']):
        company_name = row['Company Name']
    else:
        print(f"No company name found for CIK {row['CIK']}")
        continue

    cik = row['CIK']
    ticker = get_ticker_from_company_name(company_name)
    time.sleep(0.2)  # Delay between Yahoo search API calls

    if ticker and validate_ticker(ticker):
        market_cap = get_market_cap(ticker)
        industry = get_industry(ticker)
        sector, full_time_employees, current_price, pe_ratio, country = get_additional_info(
            ticker)

        # Skip companies from China or in excluded industries.
        if country == "China" or industry in excluded_industries:
            continue

        print(f"Adding {company_name} (CIK {cik}) with ticker {ticker}")
        batch_enriched_companies.append({
            'CIK': cik,
            'Ticker': ticker,
            'Market Cap': market_cap,
            'Industry': industry,
            'Sector': sector,
            'Full-Time Employees': full_time_employees,
            'Current Price': current_price,
            'P/E Ratio': pe_ratio,
            'Country': country,
            'Yahoo Pull Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    else:
        print(f"No valid ticker found for {company_name} (CIK {cik})")

    time.sleep(0.8)  # Delay to avoid Yahoo rate limits

    # Every batch_size entries (or at the end), merge the batch and update the master file.
    if i % batch_size == 0 or i == df_most_recent.shape[0]:
        if batch_enriched_companies:
            df_batch = pd.DataFrame(batch_enriched_companies)
            merged_batch = pd.merge(
                df_most_recent, df_batch, on='CIK', how='inner', suffixes=('_sec', '_yahoo'))
            combined_master = pd.concat(
                [combined_master, merged_batch], ignore_index=True)
            combined_master.to_csv(master_file_path, index=False)
            print(
                f"Batch ending at entry {i} merged and saved. Total records: {combined_master.shape[0]}")
            batch_enriched_companies = []

print(f"Final combined master file saved to {master_file_path}.")


# %%
