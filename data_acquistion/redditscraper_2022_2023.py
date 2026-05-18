"""
Reddit Data Collection Pipeline — API Transition Phase (2022–2023)
-----------------------------------------------------------------
Climate Variability, Urban Crime, and Public Perception Study

This script collects, processes, and analyzes Reddit data from Prairie
cities during the Reddit API disruption and community blackout period (2022–2023).

Purpose
-------
- Capture shifts in public discourse during platform-level disruption
- Analyze the impact of Reddit API changes and subreddit blackouts
- Continue longitudinal tracking of crime and climate discussions
- Provide continuity between pandemic (2020–2021) and modern (2024–2025) periods

Data Source
-----------
Reddit API via PRAW (Python Reddit API Wrapper)

Outputs
-------
- Monthly intermediate datasets (CSV)
- Final cleaned dataset (CSV)
- Metadata summary (JSON)
- Research summary report (TXT)
- GIS-ready dataset (CSV)
- Collection logs

Methodology
-----------
- Multi-strategy data collection:
    1. Keyword search (crime + weather)
    2. Top posts (major events)
    3. Hot posts (trending content)
    4. New posts (broad coverage)
- Monthly segmentation ensures temporal completeness
- Feature engineering includes:
    - Temporal indicators (season, quarter, API crisis phases)
    - Content features (keyword detection and overlap)
    - Engagement metrics
- Sentiment analysis performed using VADER (if available)
- User data anonymized for ethical compliance

Research Context
----------------
- Represents "API Transition Phase (2022–2023)"
- Captures structural disruption in:
    - data availability
    - user behavior
    - platform engagement
- Enables analysis of:
    - pre/post API announcement effects
    - blackout period impacts
    - post-change stabilization
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
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
        logging.FileHandler('reddit_2022_2023_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Reddit2022_2023Collector:
    """
    Reddit data collection pipeline for API disruption period (2022–2023).

    Description
    -----------
    This class manages end-to-end acquisition, filtering, processing,
    feature engineering, and export of Reddit posts related to crime
    and climate discourse during the API transition era.

    Key Responsibilities
    --------------------
    - Collect Reddit data using multiple retrieval strategies
    - Filter posts for crime and weather relevance
    - Annotate posts with API disruption timeline features
    - Generate sentiment and engagement metrics
    - Prepare datasets for statistical and geospatial analysis

    Research Role
    -------------
    - Captures effects of platform-level disruption on discourse
    - Enables comparison with stable (2014–2019) and disrupted (2020–2021) periods
    - Supports analysis of data reliability and behavioral shifts
    """
    
    def __init__(self, praw_credentials):
        """
        Initialize collector with PRAW credentials
        
        Args:
            praw_credentials: Dict with keys: client_id, client_secret, user_agent
        """
        if PRAW_AVAILABLE and praw_credentials:
            self.reddit = praw.Reddit(
                client_id=praw_credentials['client_id'],
                client_secret=praw_credentials['client_secret'],
                user_agent=praw_credentials['user_agent']
            )
            logger.info("PRAW initialized for 2022-2023 historical collection")
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
        
        # Target subreddits - Prairie cities
        self.target_subreddits = [
            'Saskatoon', 'Regina', 'Winnipeg', 'Edmonton',
            'saskatchewan', 'manitoba', 'alberta'
        ]
        
        # Enhanced keywords for 2022-2023 period
        self.crime_keywords = [
            'crime', 'theft', 'break-in', 'break in', 'burglary', 'assault',
            'robbery', 'vandalism', 'police', 'cops', 'rcmp', 'safety',
            'dangerous', 'stolen', 'stole', 'carjack', 'mugging', 'murder',
            'shooting', 'stabbing', 'domestic violence', 'gang', 'drug deal',
            'arrest', 'criminal', 'victim', 'attacked', 'security', 'violence',
            'homicide', 'fraud', 'scam', 'harassment', 'break and enter'
        ]
        
        self.weather_keywords = [
            'cold', 'hot', 'weather', 'temperature', 'snow', 'blizzard',
            'heat wave', 'heatwave', 'winter', 'summer', 'freeze', 'freezing',
            'chinook', 'windchill', 'frost', 'ice storm', 'precipitation',
            'celsius', 'fahrenheit', 'warm', 'humid', 'drought', 'rain',
            'storm', 'climate', 'seasonal', 'spring', 'fall', 'autumn',
            'extreme cold', 'extreme heat', 'tornado', 'flooding', 'wildfire'
        ]
        
        # Key dates in Reddit's 2023 API crisis
        self.api_crisis_dates = {
            'api_announcement': datetime(2023, 4, 18, tzinfo=timezone.utc),
            'blackout_start': datetime(2023, 6, 12, tzinfo=timezone.utc),
            'blackout_end': datetime(2023, 6, 14, tzinfo=timezone.utc),
            'api_changes_live': datetime(2023, 7, 1, tzinfo=timezone.utc)
        }
    
    def generate_month_list(self, start_year=2022, start_month=1, 
                           end_year=2023, end_month=12):
        """
        Generate list of (year, month) tuples for 2022-2023
        
        Returns:
            List of (year, month) tuples
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
        
        logger.info(f"Generated {len(months)} months from {start_year}-{start_month:02d} "
                   f"to {end_year}-{end_month:02d}")
        return months
    
    def collect_month_data(self, subreddit, year, month, max_per_search=150):
        """
        Collect posts from a specific month using multiple comprehensive strategies
        Enhanced for historical data collection
        
        Args:
            subreddit: Subreddit name
            year: Year (2022 or 2023)
            month: Month (1-12)
            max_per_search: Maximum posts per search query
            
        Returns:
            List of post dictionaries
        """
        month_name = datetime(year, month, 1).strftime('%B %Y')
        logger.info(f"Collecting {month_name} data from r/{subreddit}")
        
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
            
            # Strategy 1: Targeted keyword searches (crime + weather combined)
            search_terms = [
                'crime', 'police', 'theft', 'assault', 'robbery', 'break-in',
                'weather', 'cold', 'hot', 'snow', 'temperature', 'storm',
                'safety', 'dangerous', 'stolen', 'freeze', 'heat'
            ]
            
            for term in search_terms:
                try:
                    # Search with time filter
                    for submission in subreddit_obj.search(
                        term, 
                        time_filter='all', 
                        limit=max_per_search,
                        sort='relevance'
                    ):
                        post_timestamp = int(submission.created_utc)
                        
                        if first_timestamp <= post_timestamp <= last_timestamp:
                            if submission.id not in seen_ids:
                                posts.append(self._extract_post_data(submission))
                                seen_ids.add(submission.id)
                    
                    time.sleep(2.5)  # Conservative rate limiting
                    
                except Exception as e:
                    logger.warning(f"Search failed for '{term}': {str(e)}")
                    time.sleep(5)
                    continue
            
            # Strategy 2: Top posts from year
            try:
                for submission in subreddit_obj.top(time_filter='year', limit=500):
                    post_timestamp = int(submission.created_utc)
                    
                    if first_timestamp <= post_timestamp <= last_timestamp:
                        if submission.id not in seen_ids:
                            posts.append(self._extract_post_data(submission))
                            seen_ids.add(submission.id)
                
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Top posts collection failed: {str(e)}")
            
            # Strategy 3: Hot posts (captures what was trending)
            try:
                for submission in subreddit_obj.hot(limit=200):
                    post_timestamp = int(submission.created_utc)
                    
                    if first_timestamp <= post_timestamp <= last_timestamp:
                        if submission.id not in seen_ids:
                            posts.append(self._extract_post_data(submission))
                            seen_ids.add(submission.id)
                
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Hot posts collection failed: {str(e)}")
            
            # Strategy 4: New posts (good for complete coverage)
            try:
                for submission in subreddit_obj.new(limit=500):
                    post_timestamp = int(submission.created_utc)
                    
                    if first_timestamp <= post_timestamp <= last_timestamp:
                        if submission.id not in seen_ids:
                            posts.append(self._extract_post_data(submission))
                            seen_ids.add(submission.id)
                
                time.sleep(2)
            except Exception as e:
                logger.warning(f"New posts collection failed: {str(e)}")
            
            logger.info(f"  Collected {len(posts)} unique posts from {month_name}")
            
        except Exception as e:
            logger.error(f"Error collecting from r/{subreddit} for {month_name}: {str(e)}")
        
        return posts
    
    def _extract_post_data(self, submission):
        """Extract relevant data from a Reddit submission"""
        return {
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
            'is_self': submission.is_self,
            'over_18': submission.over_18,
            'spoiler': submission.spoiler,
            'stickied': submission.stickied,
            'locked': submission.locked
        }
    
    def filter_relevant_posts(self, posts):
        """Filter posts for crime or weather keywords"""
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
        """Process raw posts into structured data"""
        if not posts:
            return pd.DataFrame()
        
        df = pd.DataFrame(posts)
        df['subreddit'] = subreddit
        
        df = self._clean_data(df)
        df = self._add_temporal_features(df)
        df = self._add_content_features(df)
        df = self._add_api_crisis_flags(df)
        df = self._anonymize_data(df)
        
        if self.sentiment_analyzer:
            df = self._add_sentiment(df)
        
        return df
    
    def _clean_data(self, df):
        """Clean and standardize data"""
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
        """Add temporal features for analysis"""
        df['year'] = df['created_date'].dt.year
        df['month'] = df['created_date'].dt.month
        df['day'] = df['created_date'].dt.day
        df['day_of_week'] = df['created_date'].dt.dayofweek
        df['day_name'] = df['created_date'].dt.day_name()
        df['week_of_year'] = df['created_date'].dt.isocalendar().week
        df['hour'] = df['created_date'].dt.hour
        
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
        
        # Platform period (all posts in this dataset are in API changes era)
        df['platform_period'] = '2022-2023_api_transition'
        
        return df
    
    def _add_api_crisis_flags(self, df):
        """
        Add Reddit API disruption timeline features.

        Description
        -----------
        Annotates posts based on key events during the 2023 API crisis,
        enabling phase-based temporal analysis.

        Features
        --------
        - Pre vs post API pricing announcement (April 2023)
        - Blackout period flags (June 12–14, 2023)
        - Pre vs post API changes implementation (July 1, 2023)
        - Quarterly segmentation across 2022–2023

        Research Value
        --------------
        Enables:
        - Analysis of discourse shifts during platform disruption
        - Identification of data gaps or anomalies
        - Comparison of user behavior before and after API changes
        """
        df['is_2022'] = df['year'] == 2022
        df['is_2023'] = df['year'] == 2023
        
        # Flag posts around key API crisis dates
        df['pre_api_announcement'] = df['created_date'] < self.api_crisis_dates['api_announcement']
        df['post_api_announcement'] = df['created_date'] >= self.api_crisis_dates['api_announcement']
        
        # During blackout period
        df['during_blackout'] = (
            (df['created_date'] >= self.api_crisis_dates['blackout_start']) &
            (df['created_date'] <= self.api_crisis_dates['blackout_end'])
        )
        
        # Before/after API changes went live
        df['pre_api_changes'] = df['created_date'] < self.api_crisis_dates['api_changes_live']
        df['post_api_changes'] = df['created_date'] >= self.api_crisis_dates['api_changes_live']
        
        # Quarterly breakdown for 2022-2023
        def get_quarter(row):
            if row['year'] == 2022:
                if row['month'] <= 3:
                    return '2022_Q1'
                elif row['month'] <= 6:
                    return '2022_Q2'
                elif row['month'] <= 9:
                    return '2022_Q3'
                else:
                    return '2022_Q4'
            else:  # 2023
                if row['month'] <= 3:
                    return '2023_Q1'
                elif row['month'] <= 6:
                    return '2023_Q2'
                elif row['month'] <= 9:
                    return '2023_Q3'
                else:
                    return '2023_Q4'
        
        df['quarter'] = df.apply(get_quarter, axis=1)
        
        return df
    
    def _add_content_features(self, df):
        """Add content-based features"""
        # Clean text
        df['cleaned_text'] = df['full_text'].apply(self._clean_text)
        
        # Text length metrics
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
        
        # Both crime AND weather mentioned (key for research questions 3 & 4)
        df['crime_and_weather'] = (
            df['contains_crime_keywords'] & df['contains_weather_keywords']
        )
        
        # Engagement metrics
        df['engagement_rate'] = df['num_comments'] / (df['score'] + 1)
        df['controversy_score'] = 1 - df['upvote_ratio']
        
        return df
    
    def _clean_text(self, text):
        """Clean text for sentiment analysis"""
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
        """Anonymize author information"""
        df['author_hash'] = df['author'].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
        )
        df = df.drop('author', axis=1)
        return df
    
    def _add_sentiment(self, df):
        """Add sentiment scores using VADER"""
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
    
    def collect_2022_2023_data(self, save_intermediate=True):
        """
        Execute full data collection pipeline for API transition period.

        Description
        -----------
        Iterates through all months (2022–2023), collecting and processing
        Reddit data across all target subreddits.

        Parameters
        ----------
        save_intermediate : bool
            Whether to save monthly datasets for backup and validation.

        Returns
        -------
        pandas.DataFrame
            Combined dataset across all cities and months.

        Notes
        -----
        - Covers 24 months of API transition-era data
        - Includes disruption-aware feature engineering
        - Outputs analysis-ready dataset for integration with:
        - climate data (ECCC)
        - crime statistics (municipal datasets)
        """
        logger.info("="*70)
        logger.info("Starting 2022-2023 Historical Data Collection")
        logger.info("Target Period: API Changes & Community Blackout Era")
        logger.info("="*70)
        
        # Generate all months in 2022-2023
        months_to_collect = self.generate_month_list(
            start_year=2022, start_month=1,
            end_year=2023, end_month=12
        )
        
        all_data = []
        
        for year, month in months_to_collect:
            month_name = datetime(year, month, 1).strftime('%Y-%m')
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {month_name}")
            logger.info(f"{'='*60}")
            
            month_posts = []
            
            for subreddit in self.target_subreddits:
                try:
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
                    
                    time.sleep(3)  # Conservative rate limiting
                    
                except Exception as e:
                    logger.error(f"Error processing r/{subreddit}: {str(e)}")
                    time.sleep(5)
                    continue
            
            # Combine month data
            if month_posts:
                month_df = pd.concat(month_posts, ignore_index=True)
                month_df = month_df.drop_duplicates(subset=['id'])
                
                all_data.append(month_df)
                
                if save_intermediate:
                    filename = f"reddit_2022_2023_{month_name}.csv"
                    month_df.to_csv(filename, index=False)
                    logger.info(f"Saved: {filename} ({len(month_df)} posts)")
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['id'])
            combined_df = combined_df.sort_values('created_date')
            
            logger.info(f"\n{'='*60}")
            logger.info(f"COLLECTION COMPLETE - 2022-2023 DATASET")
            logger.info(f"{'='*60}")
            logger.info(f"Total posts collected: {len(combined_df)}")
            logger.info(f"Date range: {combined_df['created_date'].min()} to "
                       f"{combined_df['created_date'].max()}")
            logger.info(f"Months covered: {len(months_to_collect)}")
            
            return combined_df
        else:
            logger.warning("No data collected")
            return pd.DataFrame()
    
    def _print_summary(self, df):
        """Print comprehensive summary statistics"""
        logger.info("\n" + "="*70)
        logger.info("2022-2023 DATASET SUMMARY")
        logger.info("="*70)
        
        logger.info(f"\nTotal Posts: {len(df)}")
        logger.info(f"Date Range: {df['created_date'].min()} to {df['created_date'].max()}")
        
        logger.info("\n--- Posts by Subreddit ---")
        for sub, count in df['subreddit'].value_counts().items():
            logger.info(f"  r/{sub}: {count} ({count/len(df)*100:.1f}%)")
        
        logger.info("\n--- Posts by Year ---")
        for year, count in df['year'].value_counts().sort_index().items():
            logger.info(f"  {year}: {count}")
        
        logger.info("\n--- Posts by Quarter ---")
        for quarter, count in df['quarter'].value_counts().sort_index().items():
            logger.info(f"  {quarter}: {count}")
        
        logger.info("\n--- Posts by Season ---")
        for season, count in df['season'].value_counts().items():
            logger.info(f"  {season}: {count}")
        
        logger.info(f"\n--- Content Analysis ---")
        logger.info(f"Crime-related posts: {df['contains_crime_keywords'].sum()} "
                   f"({df['contains_crime_keywords'].mean()*100:.1f}%)")
        logger.info(f"Weather-related posts: {df['contains_weather_keywords'].sum()} "
                   f"({df['contains_weather_keywords'].mean()*100:.1f}%)")
        logger.info(f"Both crime AND weather: {df['crime_and_weather'].sum()} "
                   f"({df['crime_and_weather'].mean()*100:.1f}%)")
        
        logger.info(f"\n--- API Crisis Timeline ---")
        logger.info(f"Pre-API announcement: {df['pre_api_announcement'].sum()}")
        logger.info(f"Post-API announcement: {df['post_api_announcement'].sum()}")
        logger.info(f"During blackout (Jun 12–14, 2023): {df['during_blackout'].sum()}")
        logger.info(f"Pre-API changes: {df['pre_api_changes'].sum()}")
        logger.info(f"Post-API changes (July 1+): {df['post_api_changes'].sum()}")
        
        # Add sentiment summary if present
        if 'sentiment_category' in df.columns:
            logger.info(f"\n--- Sentiment Summary ---")
            sentiment_counts = df['sentiment_category'].value_counts()
            for cat, count in sentiment_counts.items():
                logger.info(f"  {cat}: {count} ({count/len(df)*100:.1f}%)")
            logger.info(f"Average compound sentiment: {df['sentiment_compound'].mean():.3f}")
        
        # Engagement metrics
        logger.info(f"\n--- Engagement Statistics ---")
        logger.info(f"Average score: {df['score'].mean():.1f}")
        logger.info(f"Average comments: {df['num_comments'].mean():.1f}")
        logger.info(f"Average upvote ratio: {df['upvote_ratio'].mean():.3f}")
        logger.info(f"Average engagement rate: {df['engagement_rate'].mean():.3f}")
        logger.info(f"Average controversy score: {df['controversy_score'].mean():.3f}")
        
        logger.info("\n" + "="*70)
        logger.info("SUMMARY COMPLETE")
        logger.info("="*70)
    
    def save_final_dataset(self, df, filename='reddit_climate_crime_2022_2023.csv'):
        """Save final analysis-ready dataset and metadata"""
        if df.empty:
            logger.error("Cannot save empty dataset")
            return
        
        # Select columns for final dataset
        columns_to_keep = [
            'id', 'subreddit', 'author_hash', 'created_date', 'created_utc',
            'year', 'month', 'day', 'day_of_week', 'day_name', 'week_of_year',
            'hour', 'season', 'quarter', 'platform_period',
            'is_2022', 'is_2023', 'pre_api_announcement', 'post_api_announcement',
            'during_blackout', 'pre_api_changes', 'post_api_changes',
            'title', 'cleaned_text', 'text_length', 'word_count',
            'link_flair_text', 'contains_crime_keywords', 'contains_weather_keywords',
            'crime_and_weather', 'crime_keyword_count', 'weather_keyword_count',
            'score', 'upvote_ratio', 'num_comments', 'engagement_rate',
            'controversy_score', 'over_18', 'spoiler', 'stickied', 'locked'
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
            'time_period': '2022-2023 (API Changes Era)',
            'total_posts': len(final_df),
            'date_range': {
                'start': str(final_df['created_date'].min()),
                'end': str(final_df['created_date'].max())
            },
            'subreddits': final_df['subreddit'].value_counts().to_dict(),
            'posts_by_year': final_df['year'].value_counts().sort_index().to_dict(),
            'posts_by_quarter': final_df['quarter'].value_counts().sort_index().to_dict(),
            'crime_related': int(final_df['contains_crime_keywords'].sum()),
            'weather_related': int(final_df['contains_weather_keywords'].sum()),
            'crime_and_weather': int(final_df['crime_and_weather'].sum()),
            'api_crisis_breakdown': {
                'pre_announcement': int(final_df['pre_api_announcement'].sum()),
                'post_announcement': int(final_df['post_api_announcement'].sum()),
                'during_blackout': int(final_df['during_blackout'].sum()),
                'pre_changes': int(final_df['pre_api_changes'].sum()),
                'post_changes': int(final_df['post_api_changes'].sum())
            }
        }
        
        metadata_filename = filename.replace('.csv', '_metadata.json')
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved: {metadata_filename}")
        
        # Print summary
        self._print_summary(final_df)
    
    def generate_research_summary(self, df, filename='research_questions_3_4_summary.txt'):
        """Generate a focused summary for research questions 3 and 4"""
        if df.empty:
            return
        
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RESEARCH QUESTIONS 3 & 4 - ANALYSIS SUMMARY\n")
            f.write("2022-2023 Reddit Data Collection\n")
            f.write("="*70 + "\n\n")
            
            f.write("QUESTION 3: How do public perceptions and discussions of crime on social media\n")
            f.write("(Reddit) vary with climate and seasonal changes?\n")
            f.write("-"*70 + "\n\n")
            
            # Sentiment by season
            f.write("SENTIMENT BY SEASON:\n")
            for season in ['winter', 'spring', 'summer', 'fall']:
                season_data = df[df['season'] == season]
                if len(season_data) > 0 and 'sentiment_compound' in df.columns:
                    avg_sentiment = season_data['sentiment_compound'].mean()
                    f.write(f"  {season.capitalize()}: {avg_sentiment:.3f} "
                           f"(n={len(season_data)} posts)\n")
            
            f.write("\nCRIME DISCUSSIONS BY SEASON:\n")
            for season in ['winter', 'spring', 'summer', 'fall']:
                season_data = df[df['season'] == season]
                crime_posts = season_data['contains_crime_keywords'].sum()
                if len(season_data) > 0:
                    pct = crime_posts / len(season_data) * 100
                    f.write(f"  {season.capitalize()}: {crime_posts} posts ({pct:.1f}%)\n")
            
            f.write("\nWEATHER DISCUSSIONS BY SEASON:\n")
            for season in ['winter', 'spring', 'summer', 'fall']:
                season_data = df[df['season'] == season]
                weather_posts = season_data['contains_weather_keywords'].sum()
                if len(season_data) > 0:
                    pct = weather_posts / len(season_data) * 100
                    f.write(f"  {season.capitalize()}: {weather_posts} posts ({pct:.1f}%)\n")
            
            f.write("\n" + "="*70 + "\n\n")
            f.write("QUESTION 4: What spatial patterns emerge when mapping climate trends,\n")
            f.write("crime data, and social media sentiment in Prairie cities?\n")
            f.write("-"*70 + "\n\n")
            
            # Posts by city
            f.write("POSTS BY CITY/REGION:\n")
            for sub in df['subreddit'].value_counts().index:
                sub_data = df[df['subreddit'] == sub]
                crime_pct = sub_data['contains_crime_keywords'].mean() * 100
                weather_pct = sub_data['contains_weather_keywords'].mean() * 100
                
                if 'sentiment_compound' in df.columns:
                    avg_sentiment = sub_data['sentiment_compound'].mean()
                    f.write(f"  r/{sub}: {len(sub_data)} posts\n")
                    f.write(f"    - Crime mentions: {crime_pct:.1f}%\n")
                    f.write(f"    - Weather mentions: {weather_pct:.1f}%\n")
                    f.write(f"    - Avg sentiment: {avg_sentiment:.3f}\n")
                else:
                    f.write(f"  r/{sub}: {len(sub_data)} posts\n")
                    f.write(f"    - Crime mentions: {crime_pct:.1f}%\n")
                    f.write(f"    - Weather mentions: {weather_pct:.1f}%\n")
            
            f.write("\nCRIME-WEATHER CORRELATION BY CITY:\n")
            for sub in df['subreddit'].value_counts().index:
                sub_data = df[df['subreddit'] == sub]
                crime_weather = sub_data['crime_and_weather'].sum()
                if len(sub_data) > 0:
                    pct = crime_weather / len(sub_data) * 100
                    f.write(f"  r/{sub}: {crime_weather} posts mention both ({pct:.1f}%)\n")
            
            f.write("\n" + "="*70 + "\n\n")
            f.write("TEMPORAL ANALYSIS:\n")
            f.write("-"*70 + "\n\n")
            
            f.write("POSTS BY QUARTER:\n")
            for quarter in sorted(df['quarter'].unique()):
                quarter_data = df[df['quarter'] == quarter]
                f.write(f"  {quarter}: {len(quarter_data)} posts\n")
            
            f.write("\nAPI CRISIS IMPACT:\n")
            f.write(f"  Pre-API announcement: {df['pre_api_announcement'].sum()} posts\n")
            f.write(f"  Post-API announcement: {df['post_api_announcement'].sum()} posts\n")
            f.write(f"  During blackout: {df['during_blackout'].sum()} posts\n")
            f.write(f"  Post-API changes: {df['post_api_changes'].sum()} posts\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("DATA READY FOR:\n")
            f.write("1. Correlation analysis with climate data (Environment Canada)\n")
            f.write("2. Correlation analysis with crime data (municipal databases)\n")
            f.write("3. Geospatial mapping (ArcGIS StoryMaps)\n")
            f.write("4. Time-series sentiment analysis\n")
            f.write("5. Statistical testing (regression, ANOVA, etc.)\n")
            f.write("="*70 + "\n")
        
        logger.info(f"Research summary saved: {filename}")
    
    def export_for_gis(self, df, filename='reddit_2022_2023_for_gis.csv'):
        """
        Export a simplified dataset optimized for GIS integration
        For use with ArcGIS StoryMaps as mentioned in the research proposal
        """
        if df.empty:
            return
        
        # City coordinates for mapping (approximate city centers)
        city_coords = {
            'Saskatoon': {'lat': 52.1332, 'lon': -106.6700, 'province': 'Saskatchewan'},
            'Regina': {'lat': 50.4452, 'lon': -104.6189, 'province': 'Saskatchewan'},
            'Winnipeg': {'lat': 49.8951, 'lon': -97.1384, 'province': 'Manitoba'},
            'Edmonton': {'lat': 53.5461, 'lon': -113.4938, 'province': 'Alberta'},
            'saskatchewan': {'lat': 52.9399, 'lon': -106.4509, 'province': 'Saskatchewan'},
            'manitoba': {'lat': 49.8951, 'lon': -97.1384, 'province': 'Manitoba'},
            'alberta': {'lat': 53.9333, 'lon': -116.5765, 'province': 'Alberta'}
        }
        
        # Add coordinates
        df_gis = df.copy()
        df_gis['latitude'] = df_gis['subreddit'].map(lambda x: city_coords.get(x, {}).get('lat'))
        df_gis['longitude'] = df_gis['subreddit'].map(lambda x: city_coords.get(x, {}).get('lon'))
        df_gis['province'] = df_gis['subreddit'].map(lambda x: city_coords.get(x, {}).get('province'))
        
        # Select key columns for GIS
        gis_columns = [
            'id', 'subreddit', 'province', 'latitude', 'longitude',
            'created_date', 'year', 'month', 'season', 'quarter',
            'contains_crime_keywords', 'contains_weather_keywords', 'crime_and_weather',
            'crime_keyword_count', 'weather_keyword_count',
            'score', 'num_comments', 'engagement_rate'
        ]
        
        if 'sentiment_compound' in df.columns:
            gis_columns.extend(['sentiment_compound', 'sentiment_category'])
        
        gis_columns = [col for col in gis_columns if col in df_gis.columns]
        df_gis = df_gis[gis_columns]
        
        # Save
        df_gis.to_csv(filename, index=False)
        logger.info(f"GIS-ready dataset saved: {filename}")
        logger.info(f"  Includes lat/lon coordinates for mapping")
        logger.info(f"  Ready for ArcGIS StoryMaps integration")


def main():
    """Main execution function for 2022-2023 data collection"""
    logger.info("=" * 70)
    logger.info("Reddit 2022-2023 Historical Data Collection")
    logger.info("Prairie Cities Climate-Crime Research")
    logger.info("Focus: API Changes & Community Blackout Era")
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

    
    output_filename = 'reddit_climate_crime_2022_2023.csv'
    
    logger.info("\nInitializing collector...")
    collector = Reddit2022_2023Collector(praw_credentials=praw_credentials)
    
    logger.info("\nStarting data collection for 2022-2023 period...")
    logger.info("This will collect 24 months of historical data")
    logger.info("Estimated time: 30-60 minutes depending on data volume\n")
    
    # Collect all data
    df = collector.collect_2022_2023_data(save_intermediate=True)
    
    if not df.empty:
        # Save final dataset
        collector.save_final_dataset(df, filename=output_filename)
        
        # Generate research-focused summary
        collector.generate_research_summary(df)
        
        # Export GIS-ready data
        collector.export_for_gis(df)
        
        logger.info("\n" + "="*70)
        logger.info("COLLECTION SUCCESSFUL")
        logger.info("="*70)
        logger.info(f"Main dataset: {output_filename}")
        logger.info(f"Metadata: {output_filename.replace('.csv', '_metadata.json')}")
        logger.info(f"Research summary: research_questions_3_4_summary.txt")
        logger.info(f"GIS dataset: reddit_2022_2023_for_gis.csv")
        logger.info(f"Individual monthly files: reddit_2022_2023_YYYY-MM.csv")
        logger.info(f"Log file: reddit_2022_2023_scraper.log")
        
        # Research-specific insights
        logger.info("\n" + "="*70)
        logger.info("RESEARCH QUESTIONS 3 & 4 - KEY INSIGHTS")
        logger.info("="*70)
        
        logger.info("\nQuestion 3: Public perceptions and discussions variation")
        logger.info("Dataset includes temporal markers for:")
        logger.info("- Seasonal variations (winter/spring/summer/fall)")
        logger.info("- API crisis timeline (pre/during/post blackout)")
        logger.info("- Sentiment analysis for each post")
        logger.info("- Day of week, hour, and quarterly breakdowns")
        
        logger.info("\nQuestion 4: Spatial patterns")
        logger.info("Dataset covers 4 cities + 3 provincial subreddits:")
        for city in collector.target_subreddits:
            city_count = len(df[df['subreddit'] == city])
            logger.info(f"- r/{city}: {city_count} posts")
        
        logger.info("\nNext steps for analysis:")
        logger.info("1. Merge with climate data (Environment Canada)")
        logger.info("2. Merge with crime data (municipal police services)")
        logger.info("3. Perform temporal correlation analysis")
        logger.info("4. Create spatial visualizations (GIS/StoryMaps)")
        logger.info("5. Analyze sentiment variations by season/weather")
        logger.info("6. Statistical testing (regression, time-series)")
        
        logger.info("\n" + "="*70)
        logger.info("FILES GENERATED:")
        logger.info("="*70)
        logger.info(f"✓ {output_filename} - Main analysis dataset")
        logger.info(f"✓ {output_filename.replace('.csv', '_metadata.json')} - Collection metadata")
        logger.info(f"✓ research_questions_3_4_summary.txt - Focused research summary")
        logger.info(f"✓ reddit_2022_2023_for_gis.csv - GIS-ready with coordinates")
        logger.info(f"✓ reddit_2022_2023_YYYY-MM.csv - Monthly backup files (24 files)")
        logger.info(f"✓ reddit_2022_2023_scraper.log - Detailed collection log")
        
    else:
        logger.error("\n" + "="*70)
        logger.error("COLLECTION FAILED - No data collected")
        logger.error("="*70)
        logger.error("Please check:")
        logger.error("1. PRAW credentials are valid")
        logger.error("2. Internet connection is stable")
        logger.error("3. Reddit API is accessible")
        logger.error("4. Rate limits have not been exceeded")
        logger.error("\nCheck reddit_2022_2023_scraper.log for details")


if __name__ == "__main__":
    if not PRAW_AVAILABLE:
        print("Error: PRAW is required. Install with: pip install praw")
        exit(1)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nCollection interrupted by user")
        logger.info("Partial data may have been saved to monthly files")
    except Exception as e:
        logger.error(f"\nFatal error: {str(e)}", exc_info=True)
        logger.info("\nTroubleshooting tips:")
        logger.info("1. Check reddit_2022_2023_scraper.log for detailed errors")
        logger.info("2. Verify PRAW credentials")
        logger.info("3. Check if any monthly files were created")
        logger.info("4. Try reducing max_per_search parameter if rate limited")