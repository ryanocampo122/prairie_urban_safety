"""
Reddit Data Collection Pipeline — Modern/Post-Disruption Phase (2024–Present)
----------------------------------------------------------------------------
Climate Variability, Urban Crime, and Public Perception Study

This script collects, processes, and analyzes Reddit data from Prairie
cities during the modern post-API-disruption period (2024–present).

Purpose
-------
- Capture current public discourse on crime and climate in Prairie cities
- Establish the most recent phase in a longitudinal social media dataset
- Track post-pandemic and post-API behavioral stabilization
- Generate high-resolution, analysis-ready datasets for integration with:
    - climate data (ECCC)
    - crime statistics
    - longitudinal sentiment analysis

Data Source
-----------
Reddit API via PRAW (Python Reddit API Wrapper)

Outputs
-------
- Monthly intermediate datasets (CSV)
- Final cleaned dataset (CSV)
- Metadata summary (JSON)
- Console-based summary statistics
- Collection logs

Methodology
-----------
- Month-by-month data collection to ensure temporal precision
- Multi-strategy retrieval:
    1. Keyword-based search (crime + weather)
    2. Top posts (high-impact events)
- Post-level filtering using expanded keyword sets
- Feature engineering includes:
    - Temporal features (season, day, platform phase)
    - Content features (keyword counts, overlap indicators)
    - Engagement metrics
- Text preprocessing for NLP and sentiment analysis
- Sentiment analysis using VADER (if available)
- Author anonymization via hashing

Research Context
----------------
- Represents "Modern/Post-Disruption Phase (2024–Present)"
- Captures stabilized Reddit usage after:
    - COVID-19 disruption (2020–2021)
    - API changes and subreddit blackouts (2023)
- Enables comparison across all study phases:
    - Baseline (2009–2013)
    - Growth (2014–2019)
    - Pandemic (2020–2021)
    - API disruption (2022–2023)
    - Modern period (2024–present)
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import re
import json
import logging
import hashlib
from calendar import monthrange

# API imports
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    print("Error: praw not installed. Install with: pip install praw")
    PRAW_AVAILABLE = False
    exit(1)

# Sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    print("Warning: vaderSentiment not installed. Install with: pip install vaderSentiment")
    VADER_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_monthly_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RedditMonthlyCollector:
    """
    Reddit data collection pipeline for modern monthly data (2024–present).

    Description
    -----------
    This class manages end-to-end acquisition, filtering, processing,
    feature engineering, sentiment analysis, and export of Reddit posts
    related to climate and crime discourse.

    Key Responsibilities
    --------------------
    - Collect Reddit posts month-by-month for precise temporal analysis
    - Filter posts using crime and weather keyword sets
    - Generate temporal and content-based analytical features
    - Compute sentiment scores using VADER (if available)
    - Anonymize user data for ethical compliance
    - Export datasets for statistical and geospatial analysis

    Research Role
    -------------
    - Provides the most recent observational layer in the dataset
    - Enables real-time comparison with historical phases
    - Supports validation of long-term trends and model generalization
    """
    
    def __init__(self, praw_credentials):
        """
        Initialize collector with PRAW credentials
        
        Args:
            praw_credentials: Dict with keys: client_id, client_secret, user_agent
        """
        # Initialize PRAW
        if PRAW_AVAILABLE and praw_credentials:
            self.reddit = praw.Reddit(
                client_id=praw_credentials['client_id'],
                client_secret=praw_credentials['client_secret'],
                user_agent=praw_credentials['user_agent']
            )
            logger.info("PRAW initialized")
        else:
            logger.error("PRAW credentials required")
            exit(1)
        
        # Initialize sentiment analyzer
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        else:
            self.sentiment_analyzer = None
            logger.warning("VADER not available - sentiment analysis disabled")
        
        # Target subreddits
        self.target_subreddits = [
            'Saskatoon', 'Regina', 'Winnipeg', 'Edmonton',
            'saskatchewan', 'manitoba', 'alberta'
        ]
        
        # Keywords for filtering
        self.crime_keywords = [
            'crime', 'theft', 'break-in', 'break in', 'burglary', 'assault',
            'robbery', 'vandalism', 'police', 'cops', 'rcmp', 'safety',
            'dangerous', 'stolen', 'stole', 'carjack', 'mugging', 'murder',
            'shooting', 'stabbing', 'domestic violence', 'gang', 'drug deal',
            'arrest', 'criminal', 'victim', 'attacked', 'security'
        ]
        
        self.weather_keywords = [
            'cold', 'hot', 'weather', 'temperature', 'snow', 'blizzard',
            'heat wave', 'winter', 'summer', 'freeze', 'freezing', 'chinook',
            'windchill', 'frost', 'ice storm', 'precipitation', 'celsius',
            'fahrenheit', 'warm', 'humid', 'drought', 'rain', 'storm',
            'climate', 'seasonal', 'spring', 'fall', 'autumn'
        ]
    
    def generate_month_list(self, start_year, start_month, end_year, end_month):
        """
        Generate list of monthly intervals.

        Parameters
        ----------
        start_year : int
        start_month : int
        end_year : int
        end_month : int

        Returns
        -------
        list of tuple
            List of (year, month) pairs covering the specified range.

        Notes
        -----
        Ensures continuous temporal coverage for longitudinal analysis.
        """
        months = []
        current_year = start_year
        current_month = start_month
        
        while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
            months.append((current_year, current_month))
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        
        return months
    
    def collect_month_data(self, subreddit, year, month, search_terms=None):
        """
        Collect posts from a specific month using multiple search strategies
        
        Args:
            subreddit: Subreddit name (e.g., 'Saskatoon')
            year: Year (e.g., 2024)
            month: Month (1-12)
            search_terms: List of search terms (default: crime + weather keywords)
            
        Returns:
            List of post dictionaries
        """
        if search_terms is None:
            # Use top crime and weather keywords for searching
            search_terms = ['crime', 'police', 'theft', 'weather', 'cold', 'snow', 'temperature']
        
        month_name = datetime(year, month, 1).strftime('%B')
        logger.info(f"Collecting {month_name} {year} data from r/{subreddit}")
        
        # Get month boundaries
        first_day = datetime(year, month, 1, tzinfo=timezone.utc)
        last_day_num = monthrange(year, month)[1]
        last_day = datetime(year, month, last_day_num, 23, 59, 59, tzinfo=timezone.utc)
        
        first_timestamp = int(first_day.timestamp())
        last_timestamp = int(last_day.timestamp())
        
        posts = []
        seen_ids = set()
        
        try:
            subreddit_obj = self.reddit.subreddit(subreddit)
            
            # Strategy 1: Search with keywords
            for term in search_terms:
                try:
                    search_query = f"{term} self:yes"  # Only self posts (text posts)
                    
                    for submission in subreddit_obj.search(search_query, time_filter='all', limit=100):
                        # Check if post is from target month
                        post_timestamp = int(submission.created_utc)
                        
                        if first_timestamp <= post_timestamp <= last_timestamp:
                            if submission.id not in seen_ids:
                                posts.append({
                                    'id': submission.id,
                                    'title': submission.title,
                                    'selftext': submission.selftext,
                                    'created_utc': submission.created_utc,
                                    'author': str(submission.author) if submission.author else '[deleted]',
                                    'score': submission.score,
                                    'num_comments': submission.num_comments,
                                    'link_flair_text': submission.link_flair_text,
                                    'upvote_ratio': submission.upvote_ratio,
                                    'permalink': submission.permalink,
                                    'url': submission.url,
                                    'is_self': submission.is_self
                                })
                                seen_ids.add(submission.id)
                    
                    time.sleep(2)  # Rate limiting between searches
                    
                except Exception as e:
                    logger.warning(f"Search failed for term '{term}': {str(e)}")
                    continue
            
            # Strategy 2: Get top posts from that time period
            try:
                for submission in subreddit_obj.top(time_filter='year', limit=500):
                    post_timestamp = int(submission.created_utc)
                    
                    if first_timestamp <= post_timestamp <= last_timestamp:
                        if submission.id not in seen_ids:
                            posts.append({
                                'id': submission.id,
                                'title': submission.title,
                                'selftext': submission.selftext,
                                'created_utc': submission.created_utc,
                                'author': str(submission.author) if submission.author else '[deleted]',
                                'score': submission.score,
                                'num_comments': submission.num_comments,
                                'link_flair_text': submission.link_flair_text,
                                'upvote_ratio': submission.upvote_ratio,
                                'permalink': submission.permalink,
                                'url': submission.url,
                                'is_self': submission.is_self
                            })
                            seen_ids.add(submission.id)
                
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Top posts collection failed: {str(e)}")
            
            logger.info(f"  Collected {len(posts)} posts from {month_name} {year}")
            
        except Exception as e:
            logger.error(f"Error collecting from r/{subreddit} for {month_name} {year}: {str(e)}")
        
        return posts
    
    def filter_relevant_posts(self, posts):
        """
        Filter posts based on crime or weather relevance.

        Parameters
        ----------
        posts : list
            Raw post dictionaries.

        Returns
        -------
        list
            Posts containing at least one relevant keyword.

        Notes
        -----
        A post is retained if it contains:
        - ≥1 crime keyword OR
        - ≥1 weather/climate keyword
        """
        relevant_posts = []
        
        for post in posts:
            full_text = f"{post.get('title', '')} {post.get('selftext', '')}".lower()
            
            # Check for crime keywords
            has_crime = any(keyword in full_text for keyword in self.crime_keywords)
            
            # Check for weather keywords
            has_weather = any(keyword in full_text for keyword in self.weather_keywords)
            
            if has_crime or has_weather:
                relevant_posts.append(post)
        
        return relevant_posts
    
    def process_posts(self, posts, subreddit):
        """
        Filter posts based on crime or weather relevance.

        Parameters
        ----------
        posts : list
            Raw post dictionaries.

        Returns
        -------
        list
            Posts containing at least one relevant keyword.

        Notes
        -----
        A post is retained if it contains:
        - ≥1 crime keyword OR
        - ≥1 weather/climate keyword
        """
        if not posts:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(posts)
        
        # Add subreddit column
        df['subreddit'] = subreddit
        
        # Clean and process
        df = self._clean_data(df)
        df = self._add_temporal_features(df)
        df = self._add_content_features(df)
        df = self._anonymize_data(df)
        
        if self.sentiment_analyzer:
            df = self._add_sentiment(df)
        
        return df
    
    def _clean_data(self, df):
        """
        Convert raw posts into structured dataset.

        Parameters
        ----------
        posts : list
        subreddit : str

        Returns
        -------
        pandas.DataFrame

        Pipeline Steps
        --------------
        1. Data cleaning
        2. Temporal feature extraction
        3. Content feature generation
        4. Author anonymization
        5. Sentiment analysis (if available)
        """
        # Remove deleted/removed content
        df = df[~df['selftext'].isin(['[deleted]', '[removed]'])]
        df = df[~df['author'].isin(['[deleted]', 'AutoModerator', '[removed]'])]
        
        # Handle missing values
        df['selftext'] = df['selftext'].fillna('')
        df['title'] = df['title'].fillna('')
        df['link_flair_text'] = df['link_flair_text'].fillna('None')
        
        # Create full text field
        df['full_text'] = df['title'] + ' ' + df['selftext']
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['id'])
        
        # Convert timestamps
        df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
        df['created_date'] = pd.to_datetime(df['created_utc'], unit='s', utc=True)
        
        # Remove posts with invalid dates
        df = df.dropna(subset=['created_date'])
        
        return df
    
    def _add_temporal_features(self, df):
        """
        Add time-based analytical features.

        Features
        --------
        - Year, month, day
        - Day of week and name
        - Week of year
        - Season classification
        - Platform phase indicator

        Notes
        -----
        Supports temporal trend and seasonal analysis.
        """
        df['year'] = df['created_date'].dt.year
        df['month'] = df['created_date'].dt.month
        df['day'] = df['created_date'].dt.day
        df['day_of_week'] = df['created_date'].dt.dayofweek
        df['day_name'] = df['created_date'].dt.day_name()
        df['week_of_year'] = df['created_date'].dt.isocalendar().week
        
        # Season classification
        def get_season(month):
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'fall'
        
        df['season'] = df['month'].apply(get_season)
        
        # Platform period
        def get_platform_period(year):
            if year < 2014:
                return 'early_growth'
            elif 2014 <= year < 2020:
                return 'global_scaling'
            elif 2020 <= year < 2023:
                return 'pandemic_era'
            elif year == 2023:
                return 'api_changes'
            else:
                return 'post_api'
        
        df['platform_period'] = df['year'].apply(get_platform_period)
        df['is_pandemic'] = df['year'].isin([2020, 2021])
        
        return df
    
    def _add_content_features(self, df):
        """
        Add time-based analytical features.

        Features
        --------
        - Year, month, day
        - Day of week and name
        - Week of year
        - Season classification
        - Platform phase indicator

        Notes
        -----
        Supports temporal trend and seasonal analysis.
        """
        # Clean text
        df['cleaned_text'] = df['full_text'].apply(self._clean_text)
        
        # Text length
        df['text_length'] = df['cleaned_text'].str.len()
        df['word_count'] = df['cleaned_text'].str.split().str.len()
        
        # Topic flags
        crime_pattern = '|'.join(self.crime_keywords)
        weather_pattern = '|'.join(self.weather_keywords)
        
        df['contains_crime_keywords'] = df['full_text'].str.contains(
            crime_pattern, case=False, na=False, regex=True
        )
        df['contains_weather_keywords'] = df['full_text'].str.contains(
            weather_pattern, case=False, na=False, regex=True
        )
        
        # Count keywords
        df['crime_keyword_count'] = df['full_text'].str.lower().apply(
            lambda x: sum(keyword in x for keyword in self.crime_keywords)
        )
        df['weather_keyword_count'] = df['full_text'].str.lower().apply(
            lambda x: sum(keyword in x for keyword in self.weather_keywords)
        )
        
        # Engagement metrics
        df['engagement_rate'] = df['num_comments'] / (df['score'] + 1)
        
        return df
    
    def _clean_text(self, text):
        """
        Normalize text for analysis.

        Actions
        -------
        - Lowercase normalization
        - Remove URLs and identifiers
        - Remove personal information (email, phone, address)
        - Remove non-alphabetic characters
        - Normalize whitespace

        Returns
        -------
        str
        """
        if pd.isna(text):
            return ''
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'\d+\s+[\w\s]+\s+(street|st|avenue|ave|road|rd|drive|dr|boulevard|blvd)',
                     '[ADDRESS]', text, flags=re.IGNORECASE)
        text = re.sub(r'[^a-zA-Z\s\.\!\?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _anonymize_data(self, df):
        """
        Anonymize user identifiers.

        Actions
        -------
        - Hash author usernames
        - Remove original author field

        Notes
        -----
        Ensures ethical handling of user-generated content.
        """
        df['author_hash'] = df['author'].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
        )
        df = df.drop('author', axis=1)
        return df
    
    def _add_sentiment(self, df):
        """
        Anonymize user identifiers.

        Actions
        -------
        - Hash author usernames
        - Remove original author field

        Notes
        -----
        Ensures ethical handling of user-generated content.
        """
        def get_sentiment_scores(text):
            if pd.isna(text) or text == '':
                return {
                    'sentiment_compound': 0,
                    'sentiment_positive': 0,
                    'sentiment_neutral': 1,
                    'sentiment_negative': 0
                }
            
            scores = self.sentiment_analyzer.polarity_scores(text)
            return {
                'sentiment_compound': scores['compound'],
                'sentiment_positive': scores['pos'],
                'sentiment_neutral': scores['neu'],
                'sentiment_negative': scores['neg']
            }
        
        sentiment_scores = df['cleaned_text'].apply(get_sentiment_scores)
        
        df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['sentiment_compound'])
        df['sentiment_positive'] = sentiment_scores.apply(lambda x: x['sentiment_positive'])
        df['sentiment_neutral'] = sentiment_scores.apply(lambda x: x['sentiment_neutral'])
        df['sentiment_negative'] = sentiment_scores.apply(lambda x: x['sentiment_negative'])
        
        def categorize_sentiment(compound):
            if compound >= 0.05:
                return 'positive'
            elif compound <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        
        df['sentiment_category'] = df['sentiment_compound'].apply(categorize_sentiment)
        
        return df
    
    def collect_all_monthly_data(self, start_year=2024, start_month=1, 
                                  end_year=None, end_month=None,
                                  save_intermediate=True):
        """
        Collect data month-by-month for all subreddits
        
        Args:
            start_year: Starting year (default: 2024)
            start_month: Starting month (default: 1)
            end_year: Ending year (default: current year)
            end_month: Ending month (default: current month)
            save_intermediate: Save monthly CSV files
            
        Returns:
            Combined DataFrame with all posts
        """
        # Default to current date if not specified
        if end_year is None:
            end_year = datetime.now().year
        if end_month is None:
            end_month = datetime.now().month
        
        # Generate list of months to collect
        months_to_collect = self.generate_month_list(
            start_year, start_month, end_year, end_month
        )
        
        logger.info(f"Collecting data for {len(months_to_collect)} months from "
                   f"{start_year}-{start_month:02d} to {end_year}-{end_month:02d}")
        
        all_data = []
        
        for year, month in months_to_collect:
            month_name = datetime(year, month, 1).strftime('%Y-%m')
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {month_name}")
            logger.info(f"{'='*60}")
            
            month_posts = []
            
            for subreddit in self.target_subreddits:
                # Collect posts for this month
                posts = self.collect_month_data(subreddit, year, month)
                
                if posts:
                    # Filter for relevant content
                    relevant_posts = self.filter_relevant_posts(posts)
                    logger.info(f"  r/{subreddit}: {len(posts)} collected, "
                               f"{len(relevant_posts)} relevant")
                    
                    # Process posts
                    if relevant_posts:
                        df = self.process_posts(relevant_posts, subreddit)
                        month_posts.append(df)
                
                time.sleep(2)  # Rate limiting between subreddits
            
            # Combine month data
            if month_posts:
                month_df = pd.concat(month_posts, ignore_index=True)
                month_df = month_df.drop_duplicates(subset=['id'])
                
                all_data.append(month_df)
                
                if save_intermediate:
                    filename = f"reddit_monthly_{month_name}.csv"
                    month_df.to_csv(filename, index=False)
                    logger.info(f"Saved: {filename} ({len(month_df)} posts)")
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['id'])
            combined_df = combined_df.sort_values('created_date')
            
            logger.info(f"\n{'='*60}")
            logger.info(f"COLLECTION COMPLETE")
            logger.info(f"{'='*60}")
            logger.info(f"Total posts collected: {len(combined_df)}")
            logger.info(f"Date range: {combined_df['created_date'].min()} to "
                       f"{combined_df['created_date'].max()}")
            logger.info(f"Months covered: {len(months_to_collect)}")
            
            return combined_df
        else:
            logger.warning("No data collected")
            return pd.DataFrame()
    
    def save_final_dataset(self, df, filename='reddit_climate_crime_monthly_dataset.csv'):
        """
        Execute full monthly data collection pipeline.

        Parameters
        ----------
        start_year : int
        start_month : int
        end_year : int, optional
        end_month : int, optional
        save_intermediate : bool

        Returns
        -------
        pandas.DataFrame
            Combined dataset across all months and subreddits.

        Notes
        -----
        - Iterates through each month sequentially
        - Ensures complete temporal coverage
        - Saves intermediate datasets for validation and recovery
        """
        if df.empty:
            logger.error("Cannot save empty dataset")
            return
        
        # Select columns for final dataset
        columns_to_keep = [
            'id', 'subreddit', 'author_hash', 'created_date', 'created_utc',
            'year', 'month', 'day', 'day_of_week', 'day_name', 'week_of_year',
            'season', 'platform_period', 'is_pandemic', 'title', 'cleaned_text',
            'text_length', 'word_count', 'link_flair_text',
            'contains_crime_keywords', 'contains_weather_keywords',
            'crime_keyword_count', 'weather_keyword_count',
            'score', 'upvote_ratio', 'num_comments', 'engagement_rate'
        ]
        
        if 'sentiment_compound' in df.columns:
            columns_to_keep.extend([
                'sentiment_compound', 'sentiment_positive',
                'sentiment_neutral', 'sentiment_negative', 'sentiment_category'
            ])
        
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        final_df = df[columns_to_keep].copy()
        
        # Save to CSV
        final_df.to_csv(filename, index=False)
        logger.info(f"\nFinal dataset saved: {filename}")
        logger.info(f"Shape: {final_df.shape}")
        
        # Save metadata
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'total_posts': len(final_df),
            'date_range': {
                'start': str(final_df['created_date'].min()),
                'end': str(final_df['created_date'].max())
            },
            'months_covered': final_df['year'].nunique() * 12 + 
                            final_df.groupby('year')['month'].nunique().sum(),
            'subreddits': final_df['subreddit'].value_counts().to_dict(),
            'posts_by_year': final_df['year'].value_counts().sort_index().to_dict(),
            'posts_by_month': final_df.groupby(['year', 'month']).size().to_dict(),
            'crime_related': int(final_df['contains_crime_keywords'].sum()),
            'weather_related': int(final_df['contains_weather_keywords'].sum()),
        }
        
        metadata_filename = filename.replace('.csv', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved: {metadata_filename}")
        
        # Print summary
        self._print_summary(final_df)
    
    def _print_summary(self, df):
        """
        Save final processed dataset and metadata.

        Outputs
        -------
        - CSV dataset
        - JSON metadata file

        Notes
        -----
        Produces analysis-ready dataset for modeling and visualization.
        """
        logger.info("\n" + "="*60)
        logger.info("DATASET SUMMARY")
        logger.info("="*60)
        
        logger.info(f"\nTotal Posts: {len(df)}")
        logger.info(f"Date Range: {df['created_date'].min()} to {df['created_date'].max()}")
        
        logger.info("\nPosts by Subreddit:")
        for sub, count in df['subreddit'].value_counts().items():
            logger.info(f"  r/{sub}: {count} ({count/len(df)*100:.1f}%)")
        
        logger.info("\nPosts by Year:")
        for year, count in df['year'].value_counts().sort_index().items():
            logger.info(f"  {year}: {count}")
        
        logger.info("\nPosts by Month (2024-2025):")
        monthly = df.groupby(['year', 'month']).size().sort_index()
        for (year, month), count in monthly.items():
            month_name = datetime(year, month, 1).strftime('%B')
            logger.info(f"  {month_name} {year}: {count}")
        
        logger.info("\nPosts by Season:")
        for season, count in df['season'].value_counts().items():
            logger.info(f"  {season}: {count}")
        
        logger.info(f"\nCrime-related posts: {df['contains_crime_keywords'].sum()} "
                   f"({df['contains_crime_keywords'].mean()*100:.1f}%)")
        logger.info(f"Weather-related posts: {df['contains_weather_keywords'].sum()} "
                   f"({df['contains_weather_keywords'].mean()*100:.1f}%)")
        
        if 'sentiment_compound' in df.columns:
            logger.info(f"\nSentiment Distribution:")
            for cat, count in df['sentiment_category'].value_counts().items():
                logger.info(f"  {cat}: {count} ({count/len(df)*100:.1f}%)")
            logger.info(f"Average sentiment: {df['sentiment_compound'].mean():.3f}")
        
        logger.info("\n" + "="*60)


def main():
    """Automatically start Reddit data collection."""
    logger.info("=" * 70)
    logger.info("Reddit Monthly Data Collection Script")
    logger.info("Prairie Cities Climate-Crime Research")
    logger.info("=" * 70)
    # ── Reddit API Credentials ─────────────────────────────────────────────
    # Set credentials via environment variables (recommended for sharing):
    #
    #   export REDDIT_CLIENT_ID="your_client_id_here"
    #   export REDDIT_CLIENT_SECRET="your_client_secret_here"
    #   export REDDIT_USER_AGENT="your_app_name/1.0 (by u/your_username)"
    #
    # Alternatively, create a .env file in the same directory:
    #   REDDIT_CLIENT_ID=your_client_id_here
    #   REDDIT_CLIENT_SECRET=your_client_secret_here
    #   REDDIT_USER_AGENT=prairie-urban-safety/1.0 (by u/your_username)
    #
    # How to get credentials:
    #   1. Go to https://www.reddit.com/prefs/apps
    #   2. Click "Create another app" at the bottom
    #   3. Select "script", give it any name, set redirect URI to http://localhost:8080
    #   4. Your client_id is the string under the app name
    #   5. Your client_secret is labelled "secret"
    #
    # Never commit credentials to a public repository.
    # ────────────────────────────────────────────────────────────────────────

    # Load from environment (try python-dotenv if available)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not required — use export commands instead

    client_id     = os.environ.get('REDDIT_CLIENT_ID')
    client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
    user_agent    = os.environ.get('REDDIT_USER_AGENT', 'prairie-urban-safety/1.0')

    if not client_id or not client_secret:
        logger.error("Reddit API credentials not found in environment variables.")
        logger.error("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET before running.")
        logger.error("See script header for instructions.")
        exit(1)

    praw_credentials = {
        'client_id':     client_id,
        'client_secret': client_secret,
        'user_agent':    user_agent,
    }


    start_year = 2024
    start_month = 1
    end_year = datetime.now().year
    end_month = datetime.now().month
    output_filename = 'reddit_climate_crime_monthly_dataset.csv'

    logger.info(f"Starting collection from {start_year}-{start_month:02d} "
                f"to {end_year}-{end_month:02d}")

    collector = RedditMonthlyCollector(praw_credentials=praw_credentials)
    df = collector.collect_all_monthly_data(
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
        save_intermediate=True
    )

    if not df.empty:
        collector.save_final_dataset(df, filename=output_filename)
        logger.info(f"Collection complete. Data saved to {output_filename}")
    else:
        logger.error("No data collected.")


if __name__ == "__main__":
    if not PRAW_AVAILABLE:
        print("Error: PRAW is required. Install with: pip install praw")
        exit(1)

    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
