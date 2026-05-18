"""
Reddit Historical Data Collection — Pre-Phase (2009–2013)
--------------------------------------------------------
Climate Variability, Urban Crime, and Public Perception Study

This script collects historical Reddit posts related to crime and weather
for Prairie cities during the early baseline period (2009–2013).

Purpose
-------
- Capture early-stage public discourse on crime and climate
- Establish a pre-growth baseline for longitudinal sentiment analysis
- Collect post-level textual data for downstream sentiment modeling

Data Source
-----------
PullPush Reddit Archive API (historical Reddit data)

Methodology
-----------
- Query-based retrieval using combined city + keyword searches
- Keywords include expanded crime and climate-related terms
- Data collected in 6-month windows to ensure temporal coverage
- Results limited per query to maintain API efficiency

Outputs
-------
- CSV file containing:
    - post metadata (id, timestamp, subreddit, score)
    - textual content (title, body)
    - associated city and keyword

Research Context
----------------
- Represents "Pre-Phase (2009–2013)" in the study
- Used as baseline for comparing later Reddit growth and sentiment shifts
- Enables analysis of early climate–crime perception patterns
"""

import requests
import time
import random
import csv
from datetime import datetime, timedelta

# --- CONFIGURATION: EXPANDED KEYWORDS ---
CITIES = ["Saskatoon", "Regina", "Winnipeg", "Edmonton"]

# Crime-related keywords
CRIME_KEYWORDS = [
    "crime", "theft", "assault", "police", "vandalism", "homicide", 
    "stolen", "gang", "drug crime", "break-in", "robbery", "shooting", 
    "violence", "incident", "arrest", "jail", "suspect", "charge", 
    "weapon", "property crime", "victim", "car prowl", "shoplifting",
    "bail", "law enforcement", "911"
]

# Climate and weather-related keywords
WEATHER_KEYWORDS = [
    "weather", "heatwave", "blizzard", "flood", "drought", "extreme cold", 
    "severe weather", "ice storm", "storm", "hail", "cold snap", 
    "record high", "record low", "humidity", "snow", "rain", 
    "power outage", "emergency", "temperature", "heat advisory",
    "freezing rain", "wind chill", "precipitation", "urban heat"
]

BASE_URL = "https://api.pullpush.io/reddit/search/submission/"

START_DATE = datetime(2009, 1, 1)
END_DATE = datetime(2014, 1, 1)


def date_to_epoch(dt):
    """
    Convert datetime to Unix epoch time.

    Parameters
    ----------
    dt : datetime
        Input datetime object.

    Returns
    -------
    int
        Unix timestamp (seconds).
    """
    return int(dt.timestamp())


def search_pullpush_archive(query, after, before, size=100):
    """
    Query the PullPush API for Reddit submissions.

    Parameters
    ----------
    query : str
        Search query (city + keyword).
    after : int
        Start time (epoch).
    before : int
        End time (epoch).
    size : int
        Maximum number of results per request.

    Returns
    -------
    list
        List of Reddit post dictionaries.

    Notes
    -----
    - Results are sorted by most recent within the time window
    - Returns empty list if request fails or no data is found
    """
    params = {
        'q': query,
        'after': after,
        'before': before,
        'size': size,
        'sort': 'desc',
        'sort_type': 'created_utc'
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data:
            return data['data']
        return []

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch data for query '{query}': {e}")
        return []


# --- MAIN EXECUTION ---
OUTPUT_FILE = "historical_reddit_sentiment_2009_2013_expanded.csv"
all_collected_data = []
current_date = START_DATE

# 6-month windows for temporal coverage
TIME_INCREMENT = timedelta(days=180)

print("Starting historical Reddit data collection (2009–2013) with expanded keywords...")


while current_date < END_DATE:
    start_epoch = date_to_epoch(current_date)
    end_date_window = min(current_date + TIME_INCREMENT, END_DATE)
    end_epoch = date_to_epoch(end_date_window)
    
    print(f"\nProcessing window: {current_date.year}-{current_date.month:02d} to {end_date_window.year}-{end_date_window.month:02d}")

    for city in CITIES:
        for keyword in CRIME_KEYWORDS + WEATHER_KEYWORDS:

            # Combined semantic query (city + topic)
            search_query = f'"{city}" AND "{keyword}"'

            data = search_pullpush_archive(search_query, start_epoch, end_epoch)
            
            if data:
                for item in data:
                    cleaned_item = {
                        'id': item.get('id'),
                        'created_utc': datetime.fromtimestamp(
                            int(item.get('created_utc'))
                        ).strftime('%Y-%m-%d %H:%M:%S'),
                        'city': city,
                        'search_keyword': keyword,
                        'title': item.get('title'),
                        'text': item.get('selftext'),
                        'subreddit': item.get('subreddit'),
                        'score': item.get('score', 0)
                    }
                    all_collected_data.append(cleaned_item)
                
                print(f"    Collected {len(data)} items for {city} + {keyword}. Total: {len(all_collected_data)}")

                # API rate limiting
                time.sleep(random.uniform(0.5, 1.5))

    current_date += TIME_INCREMENT


# --- SAVE TO CSV ---
if all_collected_data:
    fieldnames = ['id', 'created_utc', 'city', 'search_keyword', 'title', 'text', 'subreddit', 'score']

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_collected_data)
    
    print(f"\n\n**Data collection complete!** Saved {len(all_collected_data)} records to {OUTPUT_FILE}")
    print("Dataset ready for sentiment and temporal analysis.")

else:
    print("\n\nNo historical Reddit data was found for the specified queries.")