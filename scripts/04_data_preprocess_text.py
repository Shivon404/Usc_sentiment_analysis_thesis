import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading required NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("NLTK resources downloaded successfully!")

def preprocess_text(text):
    """
    Comprehensive text preprocessing function
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string to handle any non-string values
    text = str(text)
    
    # 1. Lowercase all text
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove mentions (@username) and hashtags (#hashtag)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # 4. Remove emojis and special characters (keep only letters, numbers, and spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # 7. Tokenize the text
    tokens = word_tokenize(text)
    
    # 8. Remove stopwords (English + Filipino)
    english_stopwords = set(stopwords.words('english'))
    
    # Common Filipino/Tagalog/Bisaya stopwords
    filipino_stopwords = {
        'ako', 'siya', 'na', 'lang', 'ba', 'sa', 'ng', 'mga', 'ang', 'si', 
        'ay', 'para', 'hindi', 'kasi', 'yung', 'yan', 'yun', 'din', 'rin',
        'naman', 'pala', 'talaga', 'sobra', 'masyado', 'daw', 'raw', 'kaya',
        'pero', 'tapos', 'sana', 'nila', 'natin', 'namin', 'kami', 'tayo',
        'kayo', 'sila', 'mo', 'ko', 'ka', 'niya', 'nila', 'atin', 'amin',
        'inyo', 'kanila', 'dito', 'diyan', 'doon', 'rito', 'riyan', 'roon',
        'ganito', 'ganyan', 'ganoon', 'paano', 'ano', 'sino', 'saan', 'kailan',
        'bakit', 'alin', 'ilan', 'may', 'meron', 'wala', 'walang', 'mayroong',
        'kung', 'kapag', 'pag', 'habang', 'dahil', 'kaya', 'upang', 'gayundin',
        'gayunpaman', 'subalit', 'ngunit', 'datapwat', 'o', 'ni', 'kay', 'nang',
        'pa', 'man', 'lamang', 'dapat', 'kailangan', 'pwede', 'puwede', 'maaari',
        'gusto', 'ayaw', 'ibig', 'nais', 'mahal', 'libre', 'libre', 'salamat',
        'pasensya', 'excuse', 'sorry', 'ok', 'okay', 'oo', 'hindi', 'opo', 'oho',
        'uu', 'ee', 'ah', 'oh', 'eh', 'ha', 'huh', 'ano', 'haha', 'hehe',
        'wag', 'huwag', 'ayoko', 'ayaw', 'gusto', 'trip', 'type', 'bet',
        'uy', 'ui', 'oi', 'pre', 'bro', 'sis', 'kuya', 'ate', 'tita', 'tito',
        'lola', 'lolo', 'mama', 'papa', 'nanay', 'tatay', 'inay', 'itay'
    }
    
    # Greeting words to remove (English + Filipino)
    greeting_words = {
        'hi', 'hello', 'hey', 'greetings', 'good', 'morning', 'afternoon', 
        'evening', 'night', 'goodmorning', 'goodafternoon', 'goodevening', 
        'goodnight', 'kumusta', 'kamusta', 'musta', 'maayong', 'magandang',
        'umaga', 'hapon', 'gabi', 'tanghali', 'aga', 'adlaw', 'buntag',
        'maayong', 'gabii', 'gdmorning', 'gdafternoon', 'gdevening',
        'gdnight', 'gm', 'gn', 'ge', 'ga'
    }
    
    # Combine all stopwords (including greetings)
    all_stopwords = english_stopwords.union(filipino_stopwords).union(greeting_words)
    
    # Filter out stopwords
    tokens = [token for token in tokens if token not in all_stopwords]
    
    # 9. Remove very short words (less than 2 characters) and meaningless words
    meaningless_words = {'aa', 'yo', 'ur', 'nd', 'st', 'rd', 'th', 'lol', 'omg', 'wtf', 'idk', 'tbh', 'imo', 'btw'}
    tokens = [token for token in tokens if len(token) >= 2 and token not in meaningless_words]
    
    # 10. Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # 11. Remove any remaining empty tokens
    tokens = [token for token in tokens if token.strip()]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def main():
    """Main function to run the text preprocessing pipeline"""
    
    print("Starting text preprocessing pipeline...")
    
    # Read the CSV file
    input_file = 'data/processed/03_usc_cleaned_data_translation.csv'
    output_file = 'data/processed/04_usc_data_preprocessed.csv'
    
    try:
        print(f"Reading file: {input_file}")
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} rows")
        
        # Display basic info about the dataset
        print(f"\nDataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if manual_translation column exists
        if 'manual_translation' not in df.columns:
            print("Error: 'manual_translation' column not found in the dataset!")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Display sample of original text
        print("\nSample original text:")
        sample_texts = df['manual_translation'].dropna().head(3)
        for i, text in enumerate(sample_texts, 1):
            print(f"{i}. {text[:100]}...")
        
        # Apply preprocessing
        print("\nApplying text preprocessing...")
        print("This may take a few minutes depending on the dataset size...")
        
        # Apply preprocessing function to the manual_translation column
        df['preprocessed_text'] = df['manual_translation'].apply(preprocess_text)
        
        # Display sample of preprocessed text
        print("\nSample preprocessed text:")
        sample_preprocessed = df['preprocessed_text'].head(3)
        for i, text in enumerate(sample_preprocessed, 1):
            print(f"{i}. {text}")
        
        # Calculate some statistics
        original_lengths = df['manual_translation'].fillna('').str.len()
        preprocessed_lengths = df['preprocessed_text'].str.len()
        
        print(f"\nPreprocessing Statistics:")
        print(f"Average original text length: {original_lengths.mean():.2f} characters")
        print(f"Average preprocessed text length: {preprocessed_lengths.mean():.2f} characters")
        print(f"Text reduction: {((original_lengths.mean() - preprocessed_lengths.mean()) / original_lengths.mean() * 100):.2f}%")
        
        # Count empty preprocessed texts
        empty_count = (df['preprocessed_text'] == '').sum()
        print(f"Empty preprocessed texts: {empty_count} ({empty_count/len(df)*100:.2f}%)")
        
        # Save the result
        print(f"\nSaving preprocessed data to: {output_file}")
        df.to_csv(output_file, index=False)
        print("File saved successfully!")
        
        # Display final dataset info
        print(f"\nFinal dataset shape: {df.shape}")
        print("Preprocessing pipeline completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found!")
        print("Please make sure the file exists in the specified path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()