"""
Climate Data Acquisition Script — Regina (Multi-Station)
-------------------------------------------------------
Climate Variability, Urban Crime, and Public Perception Study

This script retrieves, cleans, and aggregates climate data from
Environment and Climate Change Canada (ECCC) for Regina using
multiple weather stations to ensure continuous temporal coverage.

Purpose
-------
- Download hourly and daily climate observations (2010–2025)
- Combine multiple stations to address missing data and station gaps
- Standardize variables across inconsistent ECCC formats
- Generate daily, monthly, and yearly datasets
- Prepare climate inputs for climate–crime and sentiment analysis

Data Source
-----------
Environment and Climate Change Canada (ECCC) Bulk Data API

Outputs
-------
- Daily climate dataset (CSV)
- Monthly aggregated dataset (CSV)
- Yearly aggregated dataset (CSV)

Methodological Notes
-------------------
- Multiple stations are queried in priority order to maximize coverage
- First valid dataset per month is used (station priority preserved)
- Relative humidity is derived from hourly data and aggregated to daily means
- Temperature and precipitation are sourced from daily summaries
- Duplicate timestamps are resolved by keeping the first occurrence
"""

import requests
import pandas as pd
from datetime import date
from io import StringIO
import numpy as np

# --- CONFIGURATION (Regina - Multiple Stations) ---
CITY_NAME = "Regina"

# Station priority order (highest priority first)
STATION_IDS = [
    50877,  # Regina Intl A (current)
    28011,  # Regina RCS
    3002,   # Regina Airport (historical)
    47707,  # Regina A (backup)
]

START_YEAR = 2010
END_YEAR = 2025
TIMEFRAME_HOURLY = 1 
TIMEFRAME_DAILY_SUMMARY = 2

BASE_URL = (
    "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
    "format=csv&stationID={station_id}&"
    "Year={year}&Month={month}&Day=1&"
    "timeframe={timeframe}&submit=Download+Data"
)

# --- COLUMN DEFINITIONS ---
POTENTIAL_DATE_COLUMNS = [
    'Date/Time', 'Date/Time (LST)', 'Date/Time (CST)', 'Date/Time (MST)', 
    'Date/Time (PST)', 'Date/Time (EST)', 'Date/Time (AST)', 'True UTC Time', 'Date'
]

HOURLY_COLUMN_MAPPING = {
    'Temp (°C)': 'Temperature_C', 'Temperature (°C)': 'Temperature_C', 'Temp': 'Temperature_C', 
    'Rel Hum (%)': 'Relative_Humidity_Pct', 'Relative Humidity (%)': 'Relative_Humidity_Pct',
    'Rel Hum': 'Relative_Humidity_Pct', 'Relative Humidity': 'Relative_Humidity_Pct',
    'Precip. Amount (mm)': 'Precip_Amount_mm', 'Total Precip (mm)': 'Precip_Amount_mm',
    'Total Precip': 'Precip_Amount_mm', 'Total Precip.': 'Precip_Amount_mm'
}

DAILY_SUMMARY_COLUMN_MAPPING = {
    'Mean Temp (°C)': 'Temperature_C_mean', 
    'Max Temp (°C)': 'Temperature_C_max', 
    'Min Temp (°C)': 'Temperature_C_min', 
    'Total Precip (mm)': 'Precip_Amount_mm_sum'
}

MONTHLY_YEARLY_AGG_RULES = {
    'Temperature_C_mean': 'mean', 
    'Temperature_C_max': 'mean', 
    'Temperature_C_min': 'mean', 
    'Relative_Humidity_Pct_mean': 'mean', 
    'Precip_Amount_mm_sum': 'sum',
}


def clean_and_standardize(df, mapping, timeframe):
    """
    Clean and standardize raw climate data.

    Description
    -----------
    Harmonizes column names, converts values to numeric types, and
    enforces a consistent datetime index for downstream analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataset retrieved from ECCC API.
    mapping : dict
        Mapping from original column names to standardized variables.
    timeframe : int
        Data resolution indicator:
        - 1 = hourly data
        - 2 = daily summary data

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset indexed by datetime.

    Notes
    -----
    - Handles inconsistent formatting across stations
    - Removes invalid timestamps
    - Designed for integration with aggregation and modeling pipeline
    """
    if df.empty: return pd.DataFrame()

    df.columns = df.columns.str.strip()

    date_col_to_use = next(
        (col for col in df.columns if any(col.strip() == p for p in POTENTIAL_DATE_COLUMNS)),
        None
    )
    if date_col_to_use is None:
        return pd.DataFrame()

    df.rename(columns={date_col_to_use: 'Date_Time'}, inplace=True)

    for source, target in mapping.items():
        if source in df.columns:
            df[target] = pd.to_numeric(df[source], errors='coerce')

    df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')
    df.dropna(subset=['Date_Time'], inplace=True)

    df.sort_values(by='Date_Time', inplace=True)
    df.set_index('Date_Time', inplace=True)

    return df


def download_raw_data(station_id, year, month, timeframe):
    """
    Download raw climate data from a specific station.

    Description
    -----------
    Retrieves monthly climate data for a given station and timeframe,
    parsing the CSV response into a DataFrame.

    Parameters
    ----------
    station_id : int
        Weather station identifier.
    year : int
        Target year.
    month : int
        Target month.
    timeframe : int
        Data resolution:
        - 1 = hourly
        - 2 = daily summary

    Returns
    -------
    tuple
        (DataFrame or None, row count)

    Notes
    -----
    - Automatically detects header row due to inconsistent ECCC formatting
    - Returns (None, 0) if no valid data is found
    """
    url = BASE_URL.format(station_id=station_id, year=year, month=month, timeframe=timeframe)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        raw_text = response.text
        lines = raw_text.splitlines()

        header_row_index = -1
        for i, line in enumerate(lines[:30]):
            if 'Date/Time' in line or (timeframe == 2 and 'Date' in line):
                if timeframe == 1 and ('Temp' in line or 'Rel Hum' in line):
                    header_row_index = i; break
                elif timeframe == 2 and ('Mean Temp' in line or 'Total Precip' in line):
                    header_row_index = i; break

        if header_row_index == -1:
            return None, 0

        df = pd.read_csv(StringIO(raw_text), header=header_row_index)

        if df.empty or len(df) < 20:
            return None, 0

        return df, len(df)

    except Exception:
        return None, 0


def print_summary_table(df, name, agg='D'):
    """
    Print summary statistics for a dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to summarize.
    name : str
        Label describing dataset.
    agg : str
        Aggregation level indicator ('D', 'M', 'Y').

    Returns
    -------
    None
    """
    if df.empty:
        print(f" No data available for {name}")
        return

    start, end = df.index.min().date(), df.index.max().date()

    print(f"\n {name} Summary ({agg}-level):")
    print(f"   Date Range : {start} → {end}")
    print(f"   Total Rows : {len(df)}")
    print(f"   Columns    : {', '.join(df.columns)}")


def process_city():
    """
    Execute full multi-station climate data pipeline for Regina.

    Description
    -----------
    Iterates through multiple stations to retrieve the most complete
    dataset for each month. Ensures continuity despite station outages
    or historical gaps.

    Returns
    -------
    None

    Notes
    -----
    - Stations are queried in priority order
    - First successful download per month is used
    - Duplicate timestamps resolved by priority order
    - Outputs are saved as CSV files for modeling pipeline
    """
    print(f"\n=== Collecting {CITY_NAME} Climate {START_YEAR}–{END_YEAR} ===")
    print(f"Using multiple stations: {STATION_IDS}")
    
    hourly_data_list = []
    daily_summary_list = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            # Try each station until we get data
            hourly_downloaded = False
            daily_downloaded = False
            
            for station_id in STATION_IDS:
                # Download HOURLY data for RH
                if not hourly_downloaded:
                    hourly_df, hourly_rows = download_raw_data(station_id, year, month, TIMEFRAME_HOURLY)
                    if hourly_rows > 0:
                        print(f"Hourly data {year}-{month:02d} from Station {station_id} ({hourly_rows} rows)")
                        hourly_data_list.append(clean_and_standardize(hourly_df, HOURLY_COLUMN_MAPPING, TIMEFRAME_HOURLY))
                        hourly_downloaded = True

                # Download DAILY summary for Temp & Precip
                if not daily_downloaded:
                    daily_df, daily_rows = download_raw_data(station_id, year, month, TIMEFRAME_DAILY_SUMMARY)
                    if daily_rows > 0:
                        print(f" Daily summary {year}-{month:02d} from Station {station_id} ({daily_rows} rows)")
                        daily_summary_list.append(clean_and_standardize(daily_df, DAILY_SUMMARY_COLUMN_MAPPING, TIMEFRAME_DAILY_SUMMARY))
                        daily_downloaded = True
                
                # If both are downloaded, move to next month
                if hourly_downloaded and daily_downloaded:
                    break

    if not daily_summary_list:
        print(f" No daily summary data found for {CITY_NAME}. Exiting.")
        return

    # Process Hourly RH → Daily Mean
    hourly_rh_df = pd.DataFrame()
    if hourly_data_list:
        master_hourly_df = pd.concat(hourly_data_list)
        # Remove duplicates, keeping first occurrence
        master_hourly_df = master_hourly_df[~master_hourly_df.index.duplicated(keep='first')]
        hourly_rh_df = master_hourly_df[['Relative_Humidity_Pct']].resample('D').mean()
        hourly_rh_df.rename(columns={'Relative_Humidity_Pct': 'Relative_Humidity_Pct_mean'}, inplace=True)
        hourly_rh_df.index.name = 'Date'

    # Process Daily Summary
    daily_tp_df = pd.concat(daily_summary_list)
    daily_tp_df.index = daily_tp_df.index.normalize()
    # Remove duplicates, keeping first occurrence (priority to earlier stations in list)
    daily_tp_df = daily_tp_df[~daily_tp_df.index.duplicated(keep='first')]
    daily_tp_df.index.name = 'Date'

    #  Merge Daily + Hourly
    final_daily_df = daily_tp_df.merge(hourly_rh_df, how='left', left_index=True, right_index=True)
    final_daily_df.dropna(subset=['Temperature_C_mean'], inplace=True)

    # Save & Summarize
    final_cols = list(MONTHLY_YEARLY_AGG_RULES.keys())
    final_daily_df = final_daily_df[[c for c in final_cols if c in final_daily_df.columns]]

    # --- DAILY ---
    daily_df_filename = f"{CITY_NAME}_Daily_Climate_{START_YEAR}_to_{END_YEAR}.csv"
    final_daily_df.to_csv(daily_df_filename)
    print_summary_table(final_daily_df, "Daily Data", "D")
    print(f"Saved → {daily_df_filename}")

    # --- MONTHLY ---
    monthly_df = final_daily_df.resample('ME').agg(MONTHLY_YEARLY_AGG_RULES)
    monthly_df.index.name = 'Date'
    monthly_df_filename = f"{CITY_NAME}_Monthly_Climate_{START_YEAR}_to_{END_YEAR}.csv"
    monthly_df.to_csv(monthly_df_filename)
    print_summary_table(monthly_df, "Monthly Data", "M")
    print(f" Saved → {monthly_df_filename}")

    # --- YEARLY ---
    yearly_df = final_daily_df.resample('YE').agg(MONTHLY_YEARLY_AGG_RULES)
    yearly_df.index.name = 'Date'
    yearly_df_filename = f"{CITY_NAME}_Yearly_Climate_{START_YEAR}_to_{END_YEAR}.csv"
    yearly_df.to_csv(yearly_df_filename)
    print_summary_table(yearly_df, "Yearly Data", "Y")
    print(f"Saved → {yearly_df_filename}")

    print(f"\n SUCCESS: {CITY_NAME} (Stations {STATION_IDS})")
    print(f"   Total Daily Rows: {len(final_daily_df)}")
    print(f"   Coverage: {final_daily_df.index.min().date()} → {final_daily_df.index.max().date()}\n")


if __name__ == "__main__":
    process_city()