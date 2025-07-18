"""
Enhanced USC Reddit Data Cleaner
A comprehensive data cleaning script for USC Reddit data with advanced text normalization.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from tqdm import tqdm
import emoji
import unicodedata


class USCDataCleaner:
    """
    An enhanced data cleaning tool for USC Reddit data.
    
    This cleaner handles missing data, duplicates, emoji removal, text normalization,
    link removal, number handling, and case normalization for research-ready datasets.
    """
    
    def __init__(self, input_file: str, output_dir: str = "data/processed"):
        """
        Initialize the USC Data Cleaner.
        
        Args:
            input_file: Path to the input CSV file
            output_dir: Directory to save cleaned data
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()
        self.df = None
        self.cleaning_stats = {
            'original_rows': 0,
            'missing_text_removed': 0,
            'duplicates_removed': 0,
            'emoji_only_removed': 0,
            'short_content_removed': 0,
            'final_rows': 0
        }
        
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the CSV data with proper error handling.
        
        Returns:
            Loaded DataFrame
        """
        try:
            self.logger.info(f"Loading data from {self.input_file}")
            self.df = pd.read_csv(self.input_file)
            self.cleaning_stats['original_rows'] = len(self.df)
            self.logger.info(f"Successfully loaded {len(self.df)} rows")
            return self.df
            
        except FileNotFoundError:
            self.logger.error(f"Input file not found: {self.input_file}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
            
    def remove_missing_text(self) -> pd.DataFrame:
        """
        Remove rows with missing or empty text/combined_text fields.
        
        Returns:
            DataFrame with missing text removed
        """
        self.logger.info("Removing rows with missing or empty text...")
        
        initial_count = len(self.df)
        
        # Check for missing or empty text fields
        missing_mask = (
            self.df['text'].isna() | 
            self.df['combined_text'].isna() |
            (self.df['text'].astype(str).str.strip() == '') |
            (self.df['combined_text'].astype(str).str.strip() == '') |
            (self.df['text'].astype(str).str.lower().isin(['[deleted]', '[removed]', 'nan', 'none'])) |
            (self.df['combined_text'].astype(str).str.lower().isin(['[deleted]', '[removed]', 'nan', 'none']))
        )
        
        self.df = self.df[~missing_mask].copy()
        
        removed_count = initial_count - len(self.df)
        self.cleaning_stats['missing_text_removed'] = removed_count
        
        self.logger.info(f"Removed {removed_count} rows with missing/empty text")
        return self.df
        
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove exact duplicate combined_text entries.
        
        Returns:
            DataFrame with duplicates removed
        """
        self.logger.info("Removing duplicate entries...")
        
        initial_count = len(self.df)
        
        # Remove duplicates based on combined_text, keeping first occurrence
        self.df = self.df.drop_duplicates(subset=['combined_text'], keep='first')
        
        removed_count = initial_count - len(self.df)
        self.cleaning_stats['duplicates_removed'] = removed_count
        
        self.logger.info(f"Removed {removed_count} duplicate entries")
        return self.df
        
    def clean_text_content(self, text: str) -> str:
        """
        Comprehensive text cleaning including emojis, links, numbers, and formatting.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to string and handle encoding issues
        text = str(text)
        
        # Remove URLs and links (improved pattern)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '', text)  # domain.com patterns
        
        # Remove Reddit-specific formatting
        text = re.sub(r'/u/[A-Za-z0-9_-]+', '', text)  # Remove usernames
        text = re.sub(r'/r/[A-Za-z0-9_-]+', '', text)  # Remove subreddit mentions
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove markdown links
        text = re.sub(r'^\s*&gt;.*$', '', text, flags=re.MULTILINE)  # Remove quoted text
        
        # Remove all emojis and emoticons (comprehensive)
        text = emoji.replace_emoji(text, replace='')
        
        # Remove additional emoji-like characters and symbols
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\'\"]', '', text)
        
        # Remove numbers (optional - you can modify this based on your needs)
        text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers
        text = re.sub(r'\d+', '', text)  # Remove all digits
        
        # Remove mathematical notation and superscripts
        text = re.sub(r'\^[0-9]+', '', text)  # Remove ^2, ^3, etc.
        text = re.sub(r'[₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]', '', text)  # Remove unicode subscripts/superscripts
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n+', ' ', text)  # Multiple newlines to single space
        text = re.sub(r'\t+', ' ', text)  # Tabs to single space
        
        # Clean up punctuation
        text = re.sub(r'([.!?]){3,}', r'\1\1', text)  # Reduce excessive punctuation
        text = re.sub(r'([,;:]){2,}', r'\1', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([a-zA-Z])', r'\1 \2', text)
        
        # Remove extra whitespace
        text = text.strip()
        
        return text
        
    def is_meaningful_content(self, text: str, min_words: int = 3, min_length: int = 10) -> bool:
        """
        Check if text contains meaningful content after cleaning.
        
        Args:
            text: Text to check
            min_words: Minimum number of words required
            min_length: Minimum character length required
            
        Returns:
            True if text is meaningful, False otherwise
        """
        if not isinstance(text, str) or len(text.strip()) < min_length:
            return False
            
        # Count meaningful words (length > 1)
        words = text.split()
        meaningful_words = [word for word in words if len(word) > 1 and word.isalpha()]
        
        return len(meaningful_words) >= min_words
        
    def remove_low_quality_content(self, min_words: int = 3, min_length: int = 10) -> pd.DataFrame:
        """
        Remove rows with low-quality content (too short, emoji-only, etc.).
        
        Args:
            min_words: Minimum number of meaningful words required
            min_length: Minimum character length required
            
        Returns:
            DataFrame with low-quality content removed
        """
        self.logger.info("Removing low-quality content...")
        
        initial_count = len(self.df)
        
        # Apply content quality check with progress bar
        tqdm.pandas(desc="Checking content quality")
        
        # First clean the text to check quality
        cleaned_text = self.df['combined_text'].progress_apply(self.clean_text_content)
        
        # Check if cleaned text is meaningful
        quality_mask = cleaned_text.apply(
            lambda x: self.is_meaningful_content(x, min_words, min_length)
        )
        
        self.df = self.df[quality_mask].copy()
        
        removed_count = initial_count - len(self.df)
        self.cleaning_stats['short_content_removed'] = removed_count
        
        self.logger.info(f"Removed {removed_count} rows with low-quality content")
        return self.df
        
    def apply_text_cleaning(self) -> pd.DataFrame:
        """
        Apply comprehensive text cleaning to all text fields.
        
        Returns:
            DataFrame with cleaned text
        """
        self.logger.info("Applying comprehensive text cleaning...")
        
        # Clean text field
        tqdm.pandas(desc="Cleaning text field")
        self.df['text'] = self.df['text'].progress_apply(self.clean_text_content)
        
        # Clean combined_text field
        tqdm.pandas(desc="Cleaning combined_text field")
        self.df['combined_text'] = self.df['combined_text'].progress_apply(self.clean_text_content)
        
        self.logger.info("Text cleaning completed")
        return self.df
        
    def remove_empty_after_cleaning(self) -> pd.DataFrame:
        """
        Remove rows that became empty after text cleaning.
        
        Returns:
            DataFrame with empty rows removed
        """
        self.logger.info("Removing rows that became empty after cleaning...")
        
        initial_count = len(self.df)
        
        # Remove rows where cleaned text is empty or too short
        empty_mask = (
            (self.df['text'].str.strip() == '') |
            (self.df['combined_text'].str.strip() == '') |
            (self.df['text'].str.len() < 3) |
            (self.df['combined_text'].str.len() < 3)
        )
        
        self.df = self.df[~empty_mask].copy()
        
        removed_count = initial_count - len(self.df)
        self.cleaning_stats['emoji_only_removed'] += removed_count
        
        self.logger.info(f"Removed {removed_count} rows that became empty after cleaning")
        return self.df
        
    def validate_cleaned_data(self) -> Dict[str, Any]:
        """
        Validate the cleaned data and return quality metrics.
        
        Returns:
            Dictionary with validation metrics
        """
        self.logger.info("Validating cleaned data...")
        
        if self.df is None or len(self.df) == 0:
            return {'error': 'No data to validate'}
        
        validation_results = {
            'total_rows': len(self.df),
            'empty_text_count': (self.df['text'].str.strip() == '').sum(),
            'empty_combined_text_count': (self.df['combined_text'].str.strip() == '').sum(),
            'duplicate_count': self.df.duplicated(subset=['combined_text']).sum(),
            'avg_text_length': self.df['combined_text'].str.len().mean(),
            'min_text_length': self.df['combined_text'].str.len().min(),
            'max_text_length': self.df['combined_text'].str.len().max(),
            'avg_word_count': self.df['combined_text'].str.split().str.len().mean(),
            'contains_urls': self.df['combined_text'].str.contains(r'http|www|\.com|\.org').sum(),
            'contains_numbers': self.df['combined_text'].str.contains(r'\d').sum(),
            'contains_uppercase': self.df['combined_text'].str.contains(r'[A-Z]').sum(),
        }
        
        # Add content type and subreddit distribution if columns exist
        if 'content_type' in self.df.columns:
            validation_results['content_types'] = self.df['content_type'].value_counts().to_dict()
        
        if 'subreddit' in self.df.columns:
            validation_results['subreddit_distribution'] = self.df['subreddit'].value_counts().to_dict()
        
        return validation_results
        
    def save_cleaned_data(self, filename: str = "usc_cleaned_data.csv") -> str:
        """
        Save the cleaned data to CSV file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        self.logger.info(f"Saving cleaned data to {output_path}")
        
        try:
            # Save with progress indication
            with tqdm(desc="Saving cleaned data", unit="rows", total=len(self.df)) as pbar:
                self.df.to_csv(output_path, index=False)
                pbar.update(len(self.df))
                
            self.cleaning_stats['final_rows'] = len(self.df)
            self.logger.info(f"Successfully saved {len(self.df)} cleaned rows to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving cleaned data: {e}")
            raise
            
    def clean_data(self, min_words: int = 3, min_length: int = 10) -> pd.DataFrame:
        """
        Execute the complete data cleaning pipeline.
        
        Args:
            min_words: Minimum number of words required after cleaning
            min_length: Minimum character length required after cleaning
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Starting enhanced data cleaning pipeline...")
        
        # Load data
        self.load_data()
        
        # Execute cleaning steps in order
        self.remove_missing_text()
        self.remove_duplicates()
        self.remove_low_quality_content(min_words, min_length)
        self.apply_text_cleaning()
        self.remove_empty_after_cleaning()
        
        # Final duplicate check after cleaning
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['combined_text'], keep='first')
        final_duplicates_removed = initial_count - len(self.df)
        
        if final_duplicates_removed > 0:
            self.logger.info(f"Removed {final_duplicates_removed} additional duplicates after cleaning")
            self.cleaning_stats['duplicates_removed'] += final_duplicates_removed
        
        self.logger.info("Enhanced data cleaning pipeline completed successfully")
        return self.df
        
    def get_cleaning_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive cleaning report.
        
        Returns:
            Dictionary with cleaning statistics and metrics
        """
        report = {
            'cleaning_stats': self.cleaning_stats,
            'data_reduction': {
                'original_rows': self.cleaning_stats['original_rows'],
                'final_rows': self.cleaning_stats['final_rows'],
                'reduction_percentage': round(
                    (1 - self.cleaning_stats['final_rows'] / self.cleaning_stats['original_rows']) * 100, 2
                ) if self.cleaning_stats['original_rows'] > 0 else 0
            },
            'validation_results': self.validate_cleaned_data() if self.df is not None else {}
        }
        
        return report
        
    def print_cleaning_summary(self) -> None:
        """Print a formatted summary of the cleaning process."""
        report = self.get_cleaning_report()
        
        print("\n" + "="*60)
        print("ENHANCED USC REDDIT DATA CLEANING SUMMARY")
        print("="*60)
        print(f"Original rows: {report['cleaning_stats']['original_rows']:,}")
        print(f"Missing text removed: {report['cleaning_stats']['missing_text_removed']:,}")
        print(f"Duplicates removed: {report['cleaning_stats']['duplicates_removed']:,}")
        print(f"Low-quality content removed: {report['cleaning_stats']['short_content_removed']:,}")
        print(f"Empty after cleaning removed: {report['cleaning_stats']['emoji_only_removed']:,}")
        print(f"Final rows: {report['cleaning_stats']['final_rows']:,}")
        print(f"Data reduction: {report['data_reduction']['reduction_percentage']}%")
        print("-"*60)
        
        if report['validation_results'] and 'error' not in report['validation_results']:
            val = report['validation_results']
            print(f"Average text length: {val['avg_text_length']:.1f} characters")
            print(f"Average word count: {val['avg_word_count']:.1f} words")
            print(f"Text length range: {val['min_text_length']}-{val['max_text_length']} characters")
            print(f"Remaining URLs: {val['contains_urls']}")
            print(f"Remaining numbers: {val['contains_numbers']}")
            print(f"Remaining uppercase: {val['contains_uppercase']}")
            
            if 'content_types' in val:
                print(f"Content types: {val['content_types']}")
            if 'subreddit_distribution' in val:
                print(f"Top subreddits: {dict(list(val['subreddit_distribution'].items())[:3])}")
            
        print("="*60)


def main():
    """Main function to run the enhanced data cleaning process."""
    
    # You can modify these paths as needed
    input_file = "data/raw/usc_reddit_data_20250718_212102.csv"  # Update with your actual file path
    output_dir = "data/processed"
    
    try:
        # Initialize cleaner
        cleaner = USCDataCleaner(input_file, output_dir)
        
        # Clean the data with custom parameters
        cleaned_df = cleaner.clean_data(
            min_words=3,      # Minimum 3 meaningful words
            min_length=10     # Minimum 10 characters after cleaning
        )
        
        # Save cleaned data
        output_path = cleaner.save_cleaned_data("usc_enhanced_cleaned_data.csv")
        
        # Print summary
        cleaner.print_cleaning_summary()
        
        print(f"\n✅ Enhanced cleaned data saved to: {output_path}")
        print("\nCleaning improvements applied:")
        print("• Comprehensive emoji removal")
        print("• URL and link removal")
        print("• Number removal")
        print("• Superscript removal (^2, ^3, etc.)")
        print("• Lowercase conversion")
        print("• Reddit-specific formatting removal")
        print("• Enhanced duplicate detection")
        print("• Improved content quality filtering")
        
    except Exception as e:
        logging.error(f"Enhanced data cleaning failed: {e}")
        raise


if __name__ == "__main__":
    main()