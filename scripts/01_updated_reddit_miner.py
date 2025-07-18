"""
USC Reddit Scraper
A professional Reddit scraping tool for collecting University of San Carlos-related content.
"""

import praw
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path
import os
from dataclasses import dataclass, asdict
from tqdm import tqdm


@dataclass
class ScrapingConfig:
    """Configuration class for scraping parameters."""
    keywords: List[str]
    subreddits: List[str]
    start_year: int
    limit_per_subreddit: int
    output_dir: str


class USCRedditScraper:
    """
    A professional Reddit scraper for collecting University of San Carlos-related content.
    
    This scraper searches for USC-related posts and comments across specified subreddits,
    collecting structured data for research purposes.
    """
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        """
        Initialize the USC Reddit Scraper.
        
        Args:
            config: Optional configuration object. If None, uses default configuration.
        """
        self._setup_logging()
        self._initialize_reddit_client()
        self._setup_configuration(config)
        self.results: List[Dict[str, Any]] = []
        
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_reddit_client(self) -> None:
        """Initialize the Reddit client with credentials."""
        try:
            self.reddit = praw.Reddit(
                client_id='TG2zWWhi5QJNS8kIT5HgRQ',
                client_secret='Pk8S4MauTq0lQtg5IXfrnpIvRi_Rrw',
                user_agent='USC_Thesis_Research_v1.0 by u/Dizzy-Language-6383'
            )
            # Test the connection
            self.reddit.user.me()
            self.logger.info("Reddit client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Reddit client: {e}")
            raise
            
    def _setup_configuration(self, config: Optional[ScrapingConfig]) -> None:
        """Set up scraping configuration."""
        if config is None:
            config = ScrapingConfig(
                keywords=[
                    'university of san carlos', 'usc', 'san carlos university',
                    'carolinian', 'carolinians', 'usc talamban', 'usc downtown',
                    'usc main', 'usc tc', 'usc dc', 'usc cebu', 'usc philippines',
                    'university of san carlos talamban', 'university of san carlos main'
                ],
                subreddits=['Carolinian', 'Philippines', 'Cebu', 'studentsph', 'LawStudentsPH'],
                start_year=2020,
                limit_per_subreddit=1000,
                output_dir='data/raw'
            )
        
        self.config = config
        self.keywords_lower = [kw.lower() for kw in config.keywords]
        
        # Ensure output directory exists
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def is_relevant(self, text: str) -> bool:
        """
        Check if text contains any of the USC-related keywords.
        
        Args:
            text: Text to check for relevance
            
        Returns:
            True if text contains USC-related keywords, False otherwise
        """
        if not text:
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.keywords_lower)
    
    def is_recent(self, timestamp: float) -> bool:
        """
        Check if timestamp is within the specified year range.
        
        Args:
            timestamp: Unix timestamp to check
            
        Returns:
            True if timestamp is recent enough, False otherwise
        """
        return datetime.fromtimestamp(timestamp, timezone.utc).year >= self.config.start_year
    
    def format_timestamp(self, timestamp: float) -> str:
        """
        Format Unix timestamp to readable date string.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Formatted date string
        """
        return datetime.fromtimestamp(timestamp, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    
    def _create_base_entry(self, post: praw.models.Submission) -> Dict[str, Any]:
        """
        Create base entry dictionary with common fields.
        
        Args:
            post: Reddit post object
            
        Returns:
            Dictionary with base entry fields
        """
        return {
            'post_id': post.id,
            'subreddit': post.subreddit.display_name,
            'title': post.title,
            'score': post.score,
            'upvote_ratio': getattr(post, 'upvote_ratio', None),
            'num_comments': post.num_comments,
            'created_utc': post.created_utc,
            'created_date': self.format_timestamp(post.created_utc),
            'author': str(post.author),
            'url': post.url,
            'permalink': post.permalink,
            # Analysis fields (empty by default)
            'has_code_switching': '',
            'bisaya_count': '',
            'tagalog_count': '',
            'conyo_count': '',
            'mixed_patterns_found': '',
            'total_filipino_words': '',
            'language_mix': '',
            'usc_relevance_score': '',
            'code_switching_score': '',
            'polarity': '',
            'subjectivity': '',
            'sentiment_category': '',
            'filipino_positive_words': '',
            'filipino_negative_words': '',
        }
    
    def collect_post(self, post: praw.models.Submission, keyword_type: str) -> Dict[str, Any]:
        """
        Collect post data into structured format.
        
        Args:
            post: Reddit post object
            keyword_type: Type of keyword match
            
        Returns:
            Dictionary with post data
        """
        entry = self._create_base_entry(post)
        entry.update({
            'text': post.selftext if keyword_type == 'matched_post' else '',
            'combined_text': f"{post.title} {post.selftext}".strip(),
            'search_keyword': keyword_type,
            'content_type': 'post',
            'comment_id': '',
            'parent_id': '',
            'is_submitter': ''
        })
        return entry
    
    def collect_comment(self, comment: praw.models.Comment, post: praw.models.Submission) -> Dict[str, Any]:
        """
        Collect comment data into structured format.
        
        Args:
            comment: Reddit comment object
            post: Parent post object
            
        Returns:
            Dictionary with comment data
        """
        entry = self._create_base_entry(post)
        entry.update({
            'text': comment.body,
            'combined_text': comment.body,
            'score': comment.score,
            'upvote_ratio': '',
            'num_comments': '',
            'created_utc': comment.created_utc,
            'created_date': self.format_timestamp(comment.created_utc),
            'author': str(comment.author),
            'permalink': comment.permalink,
            'search_keyword': 'matched_comment',
            'content_type': 'comment',
            'comment_id': comment.id,
            'parent_id': comment.parent_id,
            'is_submitter': comment.is_submitter
        })
        return entry
    
    def scrape_subreddit(self, subreddit_name: str) -> None:
        """
        Scrape a single subreddit for USC-related content.
        
        Args:
            subreddit_name: Name of the subreddit to scrape
        """
        self.logger.info(f"Scraping /r/{subreddit_name}...")
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            post_count = 0
            comment_count = 0
            
            # Create progress bar for posts
            posts_iterator = subreddit.new(limit=self.config.limit_per_subreddit)
            
            with tqdm(
                desc=f"r/{subreddit_name} posts", 
                unit="posts",
                colour="blue",
                leave=False
            ) as pbar:
                
                for post in posts_iterator:
                    if not self.is_recent(post.created_utc):
                        pbar.update(1)
                        continue
                    
                    # Check if post is relevant
                    if self.is_relevant(f"{post.title} {post.selftext}"):
                        self.results.append(self.collect_post(post, 'matched_post'))
                        post_count += 1
                        pbar.set_postfix({"Found": post_count})
                    
                    # Check comments with nested progress bar
                    try:
                        post.comments.replace_more(limit=0)
                        comments_list = post.comments.list()
                        
                        if comments_list:
                            for comment in tqdm(
                                comments_list,
                                desc=f"Comments in {post.id}",
                                unit="comments",
                                colour="green",
                                leave=False
                            ):
                                if (self.is_recent(comment.created_utc) and 
                                    self.is_relevant(comment.body)):
                                    self.results.append(self.collect_comment(comment, post))
                                    comment_count += 1
                                    
                    except Exception as e:
                        self.logger.warning(f"Error processing comments for post {post.id}: {e}")
                        continue
                    
                    pbar.update(1)
                    
            self.logger.info(f"Collected {post_count} posts and {comment_count} comments from /r/{subreddit_name}")
            
        except Exception as e:
            self.logger.error(f"Error scraping /r/{subreddit_name}: {e}")
    
    def scrape(self) -> None:
        """
        Main scraping method that processes all configured subreddits.
        """
        self.logger.info("Starting USC Reddit scraping...")
        self.logger.info(f"Target subreddits: {', '.join(self.config.subreddits)}")
        self.logger.info(f"Keyword count: {len(self.config.keywords)}")
        
        # Create main progress bar for subreddits
        with tqdm(
            total=len(self.config.subreddits),
            desc="Overall Progress",
            unit="subreddit",
            colour="magenta",
            position=0
        ) as main_pbar:
            
            for subreddit_name in self.config.subreddits:
                main_pbar.set_description(f"Scraping r/{subreddit_name}")
                self.scrape_subreddit(subreddit_name)
                main_pbar.set_postfix({"Total Found": len(self.results)})
                main_pbar.update(1)
                
        self.logger.info(f"Scraping completed. Total entries collected: {len(self.results)}")
    
    def save_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Save collected data to CSV file.
        
        Args:
            filename: Optional custom filename. If None, uses default naming.
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"usc_reddit_data_{timestamp}.csv"
        
        filepath = Path(self.config.output_dir) / filename
        
        if not self.results:
            self.logger.warning("No data to save!")
            return str(filepath)
        
        try:
            df = pd.DataFrame(self.results)
            
            # Add progress bar for saving
            with tqdm(desc="Saving to CSV", unit="rows", total=len(df)) as pbar:
                df.to_csv(filepath, index=False)
                pbar.update(len(df))
                
            self.logger.info(f"Successfully saved {len(self.results)} entries to {filepath}")
            return str(filepath)
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the collected data.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.results:
            return {"total_entries": 0}
        
        df = pd.DataFrame(self.results)
        
        return {
            "total_entries": len(df),
            "posts": len(df[df['content_type'] == 'post']),
            "comments": len(df[df['content_type'] == 'comment']),
            "subreddits": df['subreddit'].unique().tolist(),
            "date_range": {
                "earliest": df['created_date'].min(),
                "latest": df['created_date'].max()
            },
            "top_authors": df['author'].value_counts().head(5).to_dict()
        }


def main():
    """Main function to run the scraper."""
    try:
        # Initialize scraper with default configuration
        scraper = USCRedditScraper()
        
        # Run the scraping process
        scraper.scrape()
        
        # Save results
        filepath = scraper.save_to_csv()
        
        # Print summary
        summary = scraper.get_summary()
        print("\n" + "="*50)
        print("SCRAPING SUMMARY")
        print("="*50)
        print(f"Total entries collected: {summary['total_entries']}")
        print(f"Posts: {summary['posts']}")
        print(f"Comments: {summary['comments']}")
        print(f"Subreddits: {', '.join(summary['subreddits'])}")
        print(f"Data saved to: {filepath}")
        
    except Exception as e:
        logging.error(f"Scraping failed: {e}")
        raise


if __name__ == "__main__":
    main()