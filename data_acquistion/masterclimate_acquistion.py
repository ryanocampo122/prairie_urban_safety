"""
Master Climate Data Acquisition and Integration Pipeline
--------------------------------------------------------
Climate Variability, Urban Crime, and Public Perception Study
Prairie Cities (Saskatoon, Edmonton, Regina, Winnipeg)

This script orchestrates the collection, cleaning, integration, and aggregation
of climate data across multiple Prairie cities using Environment and Climate
Change Canada (ECCC) data.

Purpose
-------
- Collect climate data across multiple cities and stations (2010–2025)
- Resolve station gaps using prioritized multi-station fallback logic
- Standardize variables across heterogeneous datasets
- Aggregate data into daily, monthly, and yearly resolutions
- Produce a unified multi-city dataset for climate–crime analysis

Data Source
-----------
Environment and Climate Change Canada (ECCC) Bulk Data API

Outputs
-------
- Master daily dataset (all cities combined)
- Master monthly dataset
- Master yearly dataset

Methodological Notes
-------------------
- Multiple stations per city are used to maximize temporal coverage
- Station priority ensures consistency when overlapping data exists
- Hourly data is aggregated to daily metrics (mean or sum depending on variable)
- Daily summaries provide temperature, precipitation, and derived indices
- Final datasets are structured for integration with crime and sentiment data
"""
import requests
import pandas as pd
from datetime import date
from io import StringIO
import numpy as np

# --- CONFIGURATION (All Cities) ---
CITIES_CONFIG = {
    "Saskatoon": [47707, 28011, 40578, 3328],  # RCS, Diefenbaker Airport, Airport, Historical
    "Edmonton": [27214, 1867, 50149, 27211],  # Blatchford, Int'l A, City Centre, Historical
    "Regina": [50877, 28011, 3002, 47707],  # Int'l A, RCS, Airport, Historical
    "Winnipeg": [27174, 3698, 51097, 27375]  # Richardson Int'l, Int'l A, James Armstrong, Historical
}

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

# --- EXPANDED COLUMN DEFINITIONS ---
POTENTIAL_DATE_COLUMNS = [
    'Date/Time', 'Date/Time (LST)', 'Date/Time (CST)', 'Date/Time (MST)', 
    'Date/Time (PST)', 'Date/Time (EST)', 'Date/Time (AST)', 'True UTC Time', 'Date'
]

# Hourly data mapping (more comprehensive)
HOURLY_COLUMN_MAPPING = {
    # Temperature
    'Temp (°C)': 'Temperature_C', 'Temperature (°C)': 'Temperature_C', 'Temp': 'Temperature_C',
    'Dew Point Temp (°C)': 'Dew_Point_C', 'Dew Point Temperature (°C)': 'Dew_Point_C',
    
    # Humidity
    'Rel Hum (%)': 'Relative_Humidity_Pct', 'Relative Humidity (%)': 'Relative_Humidity_Pct',
    'Rel Hum': 'Relative_Humidity_Pct', 'Relative Humidity': 'Relative_Humidity_Pct',
    
    # Precipitation
    'Precip. Amount (mm)': 'Precip_Amount_mm', 'Total Precip (mm)': 'Precip_Amount_mm',
    'Total Precip': 'Precip_Amount_mm', 'Total Precip.': 'Precip_Amount_mm',
    
    # Wind
    'Wind Spd (km/h)': 'Wind_Speed_kmh', 'Wind Speed (km/h)': 'Wind_Speed_kmh',
    'Wind Dir (10s deg)': 'Wind_Direction_deg', 'Wind Direction (10s deg)': 'Wind_Direction_deg',
    
    # Pressure
    'Stn Press (kPa)': 'Station_Pressure_kPa', 'Station Pressure (kPa)': 'Station_Pressure_kPa',
    
    # Visibility
    'Visibility (km)': 'Visibility_km', 'Visib (km)': 'Visibility_km',
    
    # Comfort indices
    'Hmdx': 'Humidex', 'Humidex': 'Humidex',
    'Wind Chill': 'Wind_Chill', 'Wnd Chl': 'Wind_Chill',
}

# Daily summary mapping (more comprehensive)
DAILY_SUMMARY_COLUMN_MAPPING = {
    # Temperature
    'Mean Temp (°C)': 'Temperature_C_mean', 'Max Temp (°C)': 'Temperature_C_max', 
    'Min Temp (°C)': 'Temperature_C_min',
    
    # Precipitation
    'Total Precip (mm)': 'Precip_Amount_mm_sum',
    'Total Rain (mm)': 'Rain_mm_sum',
    'Total Snow (cm)': 'Snow_cm_sum',
    'Snow on Grnd (cm)': 'Snow_On_Ground_cm',
    
    # Wind
    'Spd of Max Gust (km/h)': 'Max_Gust_Speed_kmh',
    'Dir of Max Gust (10s deg)': 'Max_Gust_Direction_deg',
    
    # Degree Days
    'Cool Deg Days (°C)': 'Cooling_Degree_Days',
    'Heat Deg Days (°C)': 'Heating_Degree_Days',
}

# Aggregation rules for monthly/yearly summaries
MONTHLY_YEARLY_AGG_RULES = {
    # Temperature
    'Temperature_C_mean': 'mean', 
    'Temperature_C_max': 'mean', 
    'Temperature_C_min': 'mean',
    'Dew_Point_C_mean': 'mean',
    
    # Humidity
    'Relative_Humidity_Pct_mean': 'mean',
    
    # Precipitation
    'Precip_Amount_mm_sum': 'sum',
    'Rain_mm_sum': 'sum',
    'Snow_cm_sum': 'sum',
    'Snow_On_Ground_cm_mean': 'mean',
    
    # Wind
    'Wind_Speed_kmh_mean': 'mean',
    'Max_Gust_Speed_kmh_max': 'max',
    
    # Pressure
    'Station_Pressure_kPa_mean': 'mean',
    
    # Visibility
    'Visibility_km_mean': 'mean',
    
    # Degree Days
    'Cooling_Degree_Days_sum': 'sum',
    'Heating_Degree_Days_sum': 'sum',
    
    # Comfort Indices
    'Humidex_mean': 'mean',
    'Wind_Chill_mean': 'mean',
}


def clean_and_standardize(df, mapping, timeframe):
    """
    Clean and standardize the given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to clean and standardize.
    mapping : dict
        A dictionary mapping original column names to standardized column names.
    timeframe : int
        The time frame of the data (1 for hourly, 2 for daily, etc.).

    Returns
    -------
    pandas.DataFrame
        The cleaned and standardized DataFrame.
    """
    if df.empty: return pd.DataFrame()
    df.columns = df.columns.str.strip()
    date_col_to_use = next((col for col in df.columns if any(col.strip() == p for p in POTENTIAL_DATE_COLUMNS)), None)
    if date_col_to_use is None: return pd.DataFrame()

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
    Download raw climate data for a given station, year, month, and time frame.

    Parameters
    ----------
    station_id : int
        The ID of the climate station.
    year : int
        The year of the data to download.
    month : int
        The month of the data to download.
    timeframe : int
        The time frame of the data to download (1 for hourly, 2 for daily, etc.).

    Returns
    -------
    tuple
        A tuple containing the downloaded DataFrame and the number of rows in the DataFrame.
        If there is an error downloading the data, the function returns (None, 0).
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
        
        if header_row_index == -1: return None, 0
        data_io = StringIO(raw_text)
        df = pd.read_csv(data_io, header=header_row_index)
        if df.empty or len(df) < 20: return None, 0
        return df, len(df)
    except Exception:
        return None, 0


def print_summary_table(df, name, agg='D'):
    """
    Prints a summary table of the given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to print a summary of.
    name : str
        The name of the DataFrame to print in the summary table.
    agg : str, optional
        The aggregation level of the summary table (default is 'D' for daily).

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


def process_city(city_name, station_ids):
    """
    Process climate data for a given city.

    Parameters
    ----------
    city_name : str
        The name of the city to process.
    station_ids : list of int
        A list of station IDs to use when downloading data.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the processed climate data for the given city.
    """
    print(f"\n{'='*60}")
    print(f"=== Collecting {city_name} Climate Data {START_YEAR}–{END_YEAR} ===")
    print(f"{'='*60}")
    print(f"Using stations: {station_ids}")
    
    hourly_data_list = []
    daily_summary_list = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            hourly_downloaded = False
            daily_downloaded = False
            
            for station_id in station_ids:
                # Download HOURLY data
                if not hourly_downloaded:
                  hourly_df, hourly_rows = download_raw_data(station_id, year, month, TIMEFRAME_HOURLY)
                if hourly_rows > 0:
                        print(f" Hourly {year}-{month:02d} Station {station_id} ({hourly_rows} rows)")
                        hourly_data_list.append(clean_and_standardize(hourly_df, HOURLY_COLUMN_MAPPING, TIMEFRAME_HOURLY))
                        hourly_downloaded = True

                # Download DAILY summary
                if not daily_downloaded:
                    daily_df, daily_rows = download_raw_data(station_id, year, month, TIMEFRAME_DAILY_SUMMARY)
                    if daily_rows > 0:
                        print(f" Daily  {year}-{month:02d} Station {station_id} ({daily_rows} rows)")
                        daily_summary_list.append(clean_and_standardize(daily_df, DAILY_SUMMARY_COLUMN_MAPPING, TIMEFRAME_DAILY_SUMMARY))
                        daily_downloaded = True
                
                if hourly_downloaded and daily_downloaded:
                    break

    if not daily_summary_list:
        print(f" No daily summary data found for {city_name}. Skipping.")
        return None

    # Process Hourly → Daily Aggregation
    hourly_daily_df = pd.DataFrame()
    if hourly_data_list:
        master_hourly_df = pd.concat(hourly_data_list)
        master_hourly_df = master_hourly_df[~master_hourly_df.index.duplicated(keep='first')]
        
        # Aggregate hourly to daily
        agg_dict = {}
        for col in master_hourly_df.columns:
            if 'Temperature' in col or 'Dew_Point' in col or 'Humidity' in col or 'Pressure' in col or 'Visibility' in col or 'Humidex' in col or 'Wind_Chill' in col:
                agg_dict[col] = 'mean'
            elif 'Wind_Speed' in col:
                agg_dict[col] = 'mean'
            elif 'Precip' in col:
                agg_dict[col] = 'sum'
        
        if agg_dict:
            hourly_daily_df = master_hourly_df.resample('D').agg(agg_dict)
            # Rename columns to indicate they're means/sums
            rename_dict = {}
            for col in hourly_daily_df.columns:
                if 'Precip' in col:
                    rename_dict[col] = col + '_sum'
                else:
                    rename_dict[col] = col + '_mean'
            hourly_daily_df.rename(columns=rename_dict, inplace=True)
            hourly_daily_df.index.name = 'Date'

    # Process Daily Summary
    daily_tp_df = pd.concat(daily_summary_list)
    daily_tp_df.index = daily_tp_df.index.normalize()
    daily_tp_df = daily_tp_df[~daily_tp_df.index.duplicated(keep='first')]
    daily_tp_df.index.name = 'Date'
    
    # Rename degree days columns for aggregation
    if 'Cooling_Degree_Days' in daily_tp_df.columns:
        daily_tp_df.rename(columns={'Cooling_Degree_Days': 'Cooling_Degree_Days_sum'}, inplace=True)
    if 'Heating_Degree_Days' in daily_tp_df.columns:
        daily_tp_df.rename(columns={'Heating_Degree_Days': 'Heating_Degree_Days_sum'}, inplace=True)
    if 'Snow_On_Ground_cm' in daily_tp_df.columns:
        daily_tp_df.rename(columns={'Snow_On_Ground_cm': 'Snow_On_Ground_cm_mean'}, inplace=True)
    if 'Max_Gust_Speed_kmh' in daily_tp_df.columns:
        daily_tp_df.rename(columns={'Max_Gust_Speed_kmh': 'Max_Gust_Speed_kmh_max'}, inplace=True)

    # Merge Daily + Hourly
    final_daily_df = daily_tp_df.merge(hourly_daily_df, how='outer', left_index=True, right_index=True)
    final_daily_df.sort_index(inplace=True)
    
    # Add city column
    final_daily_df.insert(0, 'City', city_name)
    
    return final_daily_df


def main():
    print(f"\n{'#'*60}")
    print(f"### MASTER CLIMATE DATA COLLECTION FOR ALL CITIES ###")
    print(f"### Period: {START_YEAR} - {END_YEAR} ###")
    print(f"{'#'*60}")
    
    all_cities_data = []
    
    # Process each city
    for city_name, station_ids in CITIES_CONFIG.items():
        city_data = process_city(city_name, station_ids)
        if city_data is not None:
            all_cities_data.append(city_data)
    
    if not all_cities_data:
        print("\n No data collected from any city. Exiting.")
        return
    
    # Combine all cities
    master_daily_df = pd.concat(all_cities_data)
    master_daily_df.sort_values(by=['City', master_daily_df.index.name], inplace=True)
    
    # Save Master Daily
    master_daily_filename = f"Master_All_Cities_Daily_Climate_{START_YEAR}_to_{END_YEAR}.csv"
    master_daily_df.to_csv(master_daily_filename)
    print(f"\n{'='*60}")
    print(f"  MASTER DAILY DATA SUMMARY:")
    print(f"   Total Rows: {len(master_daily_df)}")
    print(f"   Cities: {', '.join(CITIES_CONFIG.keys())}")
    print(f"   Date Range: {master_daily_df.index.min().date()} → {master_daily_df.index.max().date()}")
    print(f"   Total Columns: {len(master_daily_df.columns)}")
    print(f"   Columns: {', '.join(master_daily_df.columns)}")
    print(f" Saved → {master_daily_filename}")
    
    # Create Monthly aggregations
    monthly_data_list = []
    for city_name in CITIES_CONFIG.keys():
        city_daily = master_daily_df[master_daily_df['City'] == city_name].drop('City', axis=1)
        if not city_daily.empty:
            # Only aggregate columns that exist and are in the rules
            agg_dict = {col: agg for col, agg in MONTHLY_YEARLY_AGG_RULES.items() if col in city_daily.columns}
            if agg_dict:
                monthly_df = city_daily.resample('ME').agg(agg_dict)
                monthly_df.insert(0, 'City', city_name)
                monthly_data_list.append(monthly_df)
    
    if monthly_data_list:
        master_monthly_df = pd.concat(monthly_data_list)
        master_monthly_df.sort_values(by=['City', master_monthly_df.index.name], inplace=True)
        master_monthly_filename = f"Master_All_Cities_Monthly_Climate_{START_YEAR}_to_{END_YEAR}.csv"
        master_monthly_df.to_csv(master_monthly_filename)
        print(f"\n MASTER MONTHLY DATA:")
        print(f"   Total Rows: {len(master_monthly_df)}")
        print(f"  Saved → {master_monthly_filename}")
    
    # Create Yearly aggregations
    yearly_data_list = []
    for city_name in CITIES_CONFIG.keys():
        city_daily = master_daily_df[master_daily_df['City'] == city_name].drop('City', axis=1)
        if not city_daily.empty:
            agg_dict = {col: agg for col, agg in MONTHLY_YEARLY_AGG_RULES.items() if col in city_daily.columns}
            if agg_dict:
                yearly_df = city_daily.resample('YE').agg(agg_dict)
                yearly_df.insert(0, 'City', city_name)
                yearly_data_list.append(yearly_df)
    
    if yearly_data_list:
        master_yearly_df = pd.concat(yearly_data_list)
        master_yearly_df.sort_values(by=['City', master_yearly_df.index.name], inplace=True)
        master_yearly_filename = f"Master_All_Cities_Yearly_Climate_{START_YEAR}_to_{END_YEAR}.csv"
        master_yearly_df.to_csv(master_yearly_filename)
        print(f"\n MASTER YEARLY DATA:")
        print(f"   Total Rows: {len(master_yearly_df)}")
        print(f"  Saved → {master_yearly_filename}")
    
    print(f"\n{'#'*60}")
    print(f"### SUCCESS! ALL CITIES PROCESSED ###")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
