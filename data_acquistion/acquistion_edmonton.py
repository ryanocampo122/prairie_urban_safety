"""
Climate Data Acquisition Script — Edmonton
------------------------------------------
Climate Variability, Urban Crime, and Public Perception Study

This script retrieves, cleans, and aggregates climate data from
Environment and Climate Change Canada (ECCC) for Edmonton.

Purpose
-------
- Download hourly and daily climate observations (2010–2025)
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
- Relative humidity is derived from hourly data and aggregated to daily means
- Temperature and precipitation are sourced from daily summaries
- Data is standardized for integration with crime and Reddit sentiment datasets
"""

import requests
import pandas as pd
from datetime import date
from io import StringIO
import numpy as np

# --- CONFIGURATION (Edmonton) ---
CITY_NAME = "Edmonton"
STATION_ID = 27214      
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
    Clean and standardize raw climate data from ECCC.

    Description
    -----------
    This function harmonizes inconsistent column naming across ECCC datasets,
    converts values to numeric types, and ensures a consistent datetime index.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw dataset retrieved from the ECCC API.
    mapping : dict
        Dictionary mapping original column names to standardized variable names.
    timeframe : int
        Data resolution indicator:
        - 1 = hourly data
        - 2 = daily summary data

    Returns
    -------
    pandas.DataFrame
        Cleaned and standardized DataFrame indexed by datetime.

    Notes
    -----
    - Invalid or missing timestamps are removed.
    - Numeric conversion uses coercion to prevent parsing failures.
    - Designed to support downstream aggregation and modeling.
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
    Download raw climate data from the ECCC Bulk Data API.

    Description
    -----------
    Retrieves monthly climate data for a specified station and timeframe,
    then parses the CSV response into a pandas DataFrame.

    Parameters
    ----------
    station_id : int
        Identifier for the climate station.
    year : int
        Year of data to download.
    month : int
        Month of data to download.
    timeframe : int
        Data resolution:
        - 1 = hourly
        - 2 = daily summary

    Returns
    -------
    pandas.DataFrame or None
        Raw DataFrame if successful, otherwise None.
    int
        Number of rows retrieved (0 if unsuccessful).

    Notes
    -----
    - Automatically detects header row due to inconsistent CSV formatting.
    - Filters out incomplete or malformed downloads.
    """
    url = BASE_URL.format(station_id=station_id, year=year, month=month, timeframe=timeframe)

    try:
        response = requests.get(url)
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

        data_io = StringIO(raw_text)
        df = pd.read_csv(data_io, header=header_row_index)

        if df.empty or len(df) < 20:
            return None, 0

        return df, len(df)

    except Exception:
        return None, 0


def print_summary_table(df, name, agg='D'):
    """
    Print a concise summary of a climate dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to summarize.
    name : str
        Label describing the dataset (e.g., Daily, Monthly).
    agg : str, optional
        Aggregation level indicator:
        - 'D' = daily
        - 'M' = monthly
        - 'Y' = yearly

    Returns
    -------
    None

    Notes
    -----
    - Outputs summary statistics directly to console.
    - Used for validation of data coverage and structure.
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
    Execute full climate data pipeline for Edmonton.

    Description
    -----------
    Coordinates data download, cleaning, aggregation, and export for
    Edmonton climate data. Produces daily, monthly, and yearly datasets.

    Returns
    -------
    None

    Notes
    -----
    - Hourly data is used to compute daily relative humidity means.
    - Daily summary data provides temperature and precipitation variables.
    - Final datasets are saved as CSV files for integration with
      crime and sentiment analysis pipelines.
    """
    print(f"\n=== Collecting {CITY_NAME} Climate {START_YEAR}–{END_YEAR} ===")

    hourly_data_list = []
    daily_summary_list = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):

            # Download hourly data (humidity)
            hourly_df, hourly_rows = download_raw_data(STATION_ID, year, month, TIMEFRAME_HOURLY)
            if hourly_rows > 0:
                print(f" Hourly data {year}-{month:02d} ({hourly_rows} rows)")
                hourly_data_list.append(
                    clean_and_standardize(hourly_df, HOURLY_COLUMN_MAPPING, TIMEFRAME_HOURLY)
                )

            # Download daily summary data (temperature + precipitation)
            daily_df, daily_rows = download_raw_data(STATION_ID, year, month, TIMEFRAME_DAILY_SUMMARY)
            if daily_rows > 0:
                print(f" Daily summary {year}-{month:02d} ({daily_rows} rows)")
                daily_summary_list.append(
                    clean_and_standardize(daily_df, DAILY_SUMMARY_COLUMN_MAPPING, TIMEFRAME_DAILY_SUMMARY)
                )

    if not daily_summary_list:
        print(f" No daily summary data found for {CITY_NAME}. Exiting.")
        return

    # Hourly → Daily RH
    hourly_rh_df = pd.DataFrame()
    if hourly_data_list:
        master_hourly_df = pd.concat(hourly_data_list)
        hourly_rh_df = master_hourly_df[['Relative_Humidity_Pct']].resample('D').mean()
        hourly_rh_df.rename(columns={'Relative_Humidity_Pct': 'Relative_Humidity_Pct_mean'}, inplace=True)
        hourly_rh_df.index.name = 'Date'

    # Daily summary
    daily_tp_df = pd.concat(daily_summary_list)
    daily_tp_df.index = daily_tp_df.index.normalize()
    daily_tp_df = daily_tp_df[~daily_tp_df.index.duplicated(keep='first')]
    daily_tp_df.index.name = 'Date'

    # Merge
    final_daily_df = daily_tp_df.merge(hourly_rh_df, how='left', left_index=True, right_index=True)
    final_daily_df.dropna(subset=['Temperature_C_mean'], inplace=True)

    final_cols = list(MONTHLY_YEARLY_AGG_RULES.keys())
    final_daily_df = final_daily_df[[c for c in final_cols if c in final_daily_df.columns]]

    # Save outputs
    daily_df_filename = f"{CITY_NAME}_Daily_Climate_{START_YEAR}_to_{END_YEAR}.csv"
    final_daily_df.to_csv(daily_df_filename)
    print_summary_table(final_daily_df, "Daily Data", "D")
    print(f"Saved → {daily_df_filename}")

    monthly_df = final_daily_df.resample('ME').agg(MONTHLY_YEARLY_AGG_RULES)
    monthly_df.index.name = 'Date'
    monthly_df_filename = f"{CITY_NAME}_Monthly_Climate_{START_YEAR}_to_{END_YEAR}.csv"
    monthly_df.to_csv(monthly_df_filename)
    print_summary_table(monthly_df, "Monthly Data", "M")
    print(f" Saved → {monthly_df_filename}")

    yearly_df = final_daily_df.resample('YE').agg(MONTHLY_YEARLY_AGG_RULES)
    yearly_df.index.name = 'Date'
    yearly_df_filename = f"{CITY_NAME}_Yearly_Climate_{START_YEAR}_to_{END_YEAR}.csv"
    yearly_df.to_csv(yearly_df_filename)
    print_summary_table(yearly_df, "Yearly Data", "Y")
    print(f" Saved → {yearly_df_filename}")

    print(f"\n SUCCESS: {CITY_NAME} (Station {STATION_ID})")
    print(f"   Total Daily Rows: {len(final_daily_df)}")
    print(f"   Coverage: {final_daily_df.index.min().date()} → {final_daily_df.index.max().date()}\n")


if __name__ == "__main__":
    process_city()