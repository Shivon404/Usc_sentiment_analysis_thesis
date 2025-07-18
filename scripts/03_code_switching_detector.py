"""
USC Reddit Code-Switching Detection Script
Detects code-switching between English and Filipino languages (Tagalog/Bisaya) in text data.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Set, Tuple, Dict, Any
import logging
from tqdm import tqdm


class CodeSwitchingDetector:
    """
    A code-switching detector for English and Filipino languages (Tagalog/Bisaya).
    
    This detector identifies text that contains both English and Filipino words,
    indicating code-switching behavior in social media posts and comments.
    """
    
    def __init__(self):
        """Initialize the Code-Switching Detector with language word lists."""
        self._setup_logging()
        self.tagalog_words = self._get_tagalog_words()
        self.bisaya_words = self._get_bisaya_words()
        self.english_words = self._get_common_english_words()
        
        # Combine all Filipino words for easier checking
        self.filipino_words = self.tagalog_words.union(self.bisaya_words)
        
        # Statistics tracking
        self.detection_stats = {
            'total_processed': 0,
            'code_switching_found': 0,
            'english_only': 0,
            'filipino_only': 0,
            'neither_detected': 0
        }
        
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _get_tagalog_words(self) -> Set[str]:
        """
        Get a comprehensive set of common Tagalog words.
        
        Returns:
            Set of Tagalog words in lowercase
        """
        tagalog_words = {
            # Pronouns and basic words
            'ako', 'ikaw', 'siya', 'kami', 'kayo', 'sila', 'nila', 'namin', 'ninyo',
            'ko', 'mo', 'niya', 'natin', 'ninyong', 'kanila',
            
            # Particles and connectors
            'ng', 'na', 'ang', 'sa', 'para', 'pero', 'kasi', 'lang', 'din', 'rin',
            'naman', 'talaga', 'sana', 'kaya', 'ba', 'pa', 'daw', 'raw',
            
            # Common verbs
            'kumain', 'uminom', 'natulog', 'gumising', 'pumunta', 'umuwi', 'nagawa',
            'ginawa', 'magawa', 'gagawa', 'gawin', 'gawa', 'kain', 'inom', 'tulog',
            
            # Common adjectives
            'maganda', 'pangit', 'mabait', 'masama', 'malaki', 'maliit', 'mahal',
            'mura', 'mainit', 'malamig', 'masarap', 'masakit',
            
            # Common nouns
            'bahay', 'kotse', 'pera', 'trabaho', 'pamilya', 'kaibigan', 'pagkain',
            'tubig', 'damit', 'sapatos', 'libro', 'cellphone', 'kompyuter',
            
            # Time and numbers
            'araw', 'gabi', 'umaga', 'hapon', 'linggo', 'buwan', 'taon', 'oras',
            'isa', 'dalawa', 'tatlo', 'apat', 'lima', 'anim', 'pito', 'walo', 'siyam', 'sampu',
            
            # Common expressions
            'oo', 'hindi', 'salamat', 'pasensya', 'tara', 'hala', 'grabe', 'hay',
            'naku', 'sus', 'ano', 'bakit', 'saan', 'paano', 'kelan', 'sino',
            
            # Casual/slang
            'tol', 'kuya', 'ate', 'tita', 'tito', 'lola', 'lolo', 'nanay', 'tatay',
            'bro', 'sis', 'mare', 'pare', 'tsong', 'girl', 'boy'
        }
        
        return tagalog_words
        
    def _get_bisaya_words(self) -> Set[str]:
        """
        Get a comprehensive set of common Bisaya/Cebuano words.
        
        Returns:
            Set of Bisaya words in lowercase
        """
        bisaya_words = {
            # Common particles and connectors
            'walay', 'kaayo', 'jud', 'gyud', 'murag', 'unsa', 'ug', 'ra', 'man',
            'gani', 'sad', 'pud', 'pod', 'lagi', 'bitaw', 'diay', 'kay', 'oy',
            
            # Pronouns
            'ako', 'ikaw', 'siya', 'kami', 'kamo', 'sila', 'nako', 'nimo', 'niya',
            'nato', 'ninyo', 'nila', 'ko', 'mo', 'nya',
            
            # Common verbs
            'kaon', 'inom', 'tulog', 'mata', 'adto', 'balik', 'uli', 'lakaw',
            'lingkod', 'barog', 'higa', 'sulti', 'tingog', 'tan-aw', 'basag',
            
            # Common adjectives
            'gwapa', 'gwapo', 'bati', 'maayo', 'dako', 'gamay', 'mahal', 'barato',
            'init', 'bugnaw', 'lami', 'sakit', 'kusog', 'hinay', 'taas', 'mubo',
            
            # Common nouns
            'balay', 'sakyanan', 'kwarta', 'trabaho', 'pamilya', 'higala', 'pagkaon',
            'tubig', 'sinina', 'sapatos', 'libro', 'cellphone', 'kompyuter',
            
            # Time expressions
            'adlaw', 'gabii', 'buntag', 'hapon', 'dominggo', 'buwan', 'tuig', 'takna',
            'karon', 'ugma', 'kagahapon', 'kaniadto', 'karong adlawa',
            
            # Numbers
            'usa', 'duha', 'tulo', 'upat', 'lima', 'unom', 'pito', 'walo', 'siyam', 'napulo',
            
            # Common expressions
            'oo', 'dili', 'salamat', 'pasaylo', 'tara', 'hala', 'grabe', 'ay',
            'sus', 'naku', 'asa', 'ngano', 'asa', 'unsaon', 'kanus-a', 'kinsa',
            
            # Casual/terms of address  
            'dong', 'day', 'manong', 'manang', 'tita', 'tito', 'lola', 'lolo',
            'mama', 'papa', 'kuya', 'ate', 'bai', 'bro', 'sis'
        }
        
        return bisaya_words
        
    def _get_common_english_words(self) -> Set[str]:
        """
        Get a set of common English words to help identify English content.
        
        Returns:
            Set of common English words in lowercase
        """
        english_words = {
            # Common pronouns
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that', 'these', 'those',
            
            # Common verbs
            'am', 'is', 'are', 'was', 'were', 'be', 'being', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might',
            'go', 'went', 'going', 'come', 'came', 'get', 'got', 'make', 'made', 'take', 'took',
            'see', 'saw', 'know', 'knew', 'think', 'thought', 'want', 'like', 'love', 'need',
            
            # Common adjectives
            'good', 'bad', 'big', 'small', 'great', 'little', 'new', 'old', 'first', 'last',
            'long', 'short', 'high', 'low', 'right', 'wrong', 'same', 'different', 'important',
            'easy', 'hard', 'nice', 'beautiful', 'pretty', 'ugly', 'happy', 'sad', 'angry',
            
            # Common nouns
            'time', 'person', 'year', 'way', 'day', 'thing', 'man', 'world', 'life', 'hand',
            'part', 'child', 'eye', 'woman', 'place', 'work', 'week', 'case', 'point', 'government',
            'school', 'student', 'teacher', 'class', 'book', 'money', 'food', 'water', 'house',
            
            # Common prepositions and conjunctions
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'among', 'under', 'over', 'so', 'because', 'if', 'when', 'while',
            
            # Common adverbs
            'not', 'now', 'here', 'there', 'then', 'very', 'well', 'still', 'just', 'more',
            'also', 'how', 'where', 'why', 'what', 'when', 'who', 'which', 'really', 'again',
            
            # Common internet/social media terms
            'lol', 'omg', 'wtf', 'btw', 'tbh', 'imo', 'imho', 'fyi', 'aka', 'etc'
        }
        
        return english_words
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis by cleaning and normalizing.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags but keep the text
        text = re.sub(r'[@#]\w+', '', text)
        
        # Remove excessive punctuation but keep some for word boundaries
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
        
    def extract_words(self, text: str) -> Set[str]:
        """
        Extract individual words from preprocessed text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Set of words found in the text
        """
        if not text:
            return set()
            
        # Split into words and filter out very short words and numbers
        words = set()
        for word in text.split():
            word = word.strip("'")  # Remove quotes but keep contractions
            if len(word) >= 2 and not word.isdigit():
                words.add(word)
                
        return words
        
    def detect_languages(self, text: str) -> Tuple[bool, bool]:
        """
        Detect presence of English and Filipino languages in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (has_english, has_filipino)
        """
        if not text:
            return False, False
            
        # Preprocess text
        processed_text = self.preprocess_text(text)
        words = self.extract_words(processed_text)
        
        if not words:
            return False, False
            
        # Check for English words
        has_english = len(words.intersection(self.english_words)) > 0
        
        # Check for Filipino words (Tagalog or Bisaya)
        has_filipino = len(words.intersection(self.filipino_words)) > 0
        
        return has_english, has_filipino
        
    def detect_code_switching(self, text: str) -> Dict[str, Any]:
        """
        Detect code-switching in the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detection results
        """
        has_english, has_filipino = self.detect_languages(text)
        
        # Code-switching occurs when both languages are present
        has_code_switching = has_english and has_filipino
        
        # Get detailed word analysis
        processed_text = self.preprocess_text(text)
        words = self.extract_words(processed_text)
        
        english_words_found = words.intersection(self.english_words)
        filipino_words_found = words.intersection(self.filipino_words)
        tagalog_words_found = words.intersection(self.tagalog_words)
        bisaya_words_found = words.intersection(self.bisaya_words)
        
        return {
            'has_code_switching': has_code_switching,
            'has_english': has_english,
            'has_filipino': has_filipino,
            'english_words_found': list(english_words_found),
            'filipino_words_found': list(filipino_words_found),
            'tagalog_words_found': list(tagalog_words_found),
            'bisaya_words_found': list(bisaya_words_found),
            'english_word_count': len(english_words_found),
            'filipino_word_count': len(filipino_words_found),
            'tagalog_word_count': len(tagalog_words_found),
            'bisaya_word_count': len(bisaya_words_found)
        }
        
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the entire DataFrame to detect code-switching.
        
        Args:
            df: Input DataFrame with combined_text column
            
        Returns:
            DataFrame with updated code-switching information
        """
        self.logger.info(f"Processing {len(df)} rows for code-switching detection...")
        
        # Reset statistics
        self.detection_stats = {
            'total_processed': 0,
            'code_switching_found': 0,
            'english_only': 0,
            'filipino_only': 0,
            'neither_detected': 0
        }
        
        # Initialize results lists
        has_code_switching_list = []
        detailed_results = []
        
        # Process each row with progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Detecting code-switching"):
            text = row.get('combined_text', '')
            
            # Detect code-switching
            result = self.detect_code_switching(text)
            
            # Store main result
            has_code_switching_list.append(result['has_code_switching'])
            detailed_results.append(result)
            
            # Update statistics
            self.detection_stats['total_processed'] += 1
            
            if result['has_code_switching']:
                self.detection_stats['code_switching_found'] += 1
            elif result['has_english'] and not result['has_filipino']:
                self.detection_stats['english_only'] += 1
            elif result['has_filipino'] and not result['has_english']:
                self.detection_stats['filipino_only'] += 1
            else:
                self.detection_stats['neither_detected'] += 1
                
        # Update DataFrame
        df_result = df.copy()
        df_result['has_code_switching'] = has_code_switching_list
        
        # Optionally add detailed columns for analysis
        df_result['english_word_count'] = [r['english_word_count'] for r in detailed_results]
        df_result['filipino_word_count'] = [r['filipino_word_count'] for r in detailed_results]
        df_result['tagalog_word_count'] = [r['tagalog_word_count'] for r in detailed_results]
        df_result['bisaya_word_count'] = [r['bisaya_word_count'] for r in detailed_results]
        
        self.logger.info("Code-switching detection completed")
        return df_result
        
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get detection statistics.
        
        Returns:
            Dictionary with detection statistics
        """
        total = self.detection_stats['total_processed']
        if total == 0:
            return self.detection_stats
            
        stats = self.detection_stats.copy()
        stats['percentages'] = {
            'code_switching_percentage': round((stats['code_switching_found'] / total) * 100, 2),
            'english_only_percentage': round((stats['english_only'] / total) * 100, 2),
            'filipino_only_percentage': round((stats['filipino_only'] / total) * 100, 2),
            'neither_detected_percentage': round((stats['neither_detected'] / total) * 100, 2)
        }
        
        return stats
        
    def print_detection_summary(self) -> None:
        """Print a formatted summary of the detection results."""
        stats = self.get_detection_stats()
        
        print("\n" + "="*60)
        print("CODE-SWITCHING DETECTION SUMMARY")
        print("="*60)
        print(f"Total texts processed: {stats['total_processed']:,}")
        print(f"Code-switching detected: {stats['code_switching_found']:,} ({stats['percentages']['code_switching_percentage']}%)")
        print(f"English only: {stats['english_only']:,} ({stats['percentages']['english_only_percentage']}%)")
        print(f"Filipino only: {stats['filipino_only']:,} ({stats['percentages']['filipino_only_percentage']}%)")
        print(f"Neither detected: {stats['neither_detected']:,} ({stats['percentages']['neither_detected_percentage']}%)")
        print("="*60)


def main():
    """Main function to run the code-switching detection process."""
    
    # File paths
    input_file = "data/processed/01_usc_enhanced_cleaned_data.csv"
    output_file = "data/processed/02_usc_cleaned_with_code_switch.csv"
    
    try:
        # Initialize detector
        detector = CodeSwitchingDetector()
        
        # Load data
        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} rows")
        
        # Process data for code-switching detection
        df_processed = detector.process_dataframe(df)
        
        # Save results
        print(f"\nSaving results to {output_file}...")
        df_processed.to_csv(output_file, index=False)
        print(f"Successfully saved {len(df_processed)} rows with code-switching detection")
        
        # Print summary
        detector.print_detection_summary()
        
        # Show some examples
        print("\nSample results:")
        print("-" * 60)
        code_switch_examples = df_processed[df_processed['has_code_switching'] == True].head(3)
        
        for idx, row in code_switch_examples.iterrows():
            print(f"Text: {row['combined_text'][:100]}...")
            print(f"Code-switching: {row['has_code_switching']}")
            print(f"English words: {row['english_word_count']}, Filipino words: {row['filipino_word_count']}")
            print("-" * 60)
            
        print(f"\n✅ Code-switching detection completed successfully!")
        print(f"📄 Results saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found.")
        print("Please make sure the cleaned data file exists.")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()