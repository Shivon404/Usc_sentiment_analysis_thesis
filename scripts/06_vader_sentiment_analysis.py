import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load the CSV file and return DataFrame"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded {len(df)} rows from {file_path}")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def initialize_vader():
    """Initialize VADER sentiment analyzer"""
    try:
        analyzer = SentimentIntensityAnalyzer()
        print("✅ VADER sentiment analyzer initialized successfully")
        return analyzer
    except Exception as e:
        print(f"❌ Error initializing VADER: {e}")
        print("💡 Try installing vaderSentiment: pip install vaderSentiment")
        return None

def get_sentiment_scores(text, analyzer):
    """Get VADER sentiment scores for a single text"""
    if pd.isna(text) or text.strip() == '':
        return {
            'compound': 0.0,
            'pos': 0.0,
            'neu': 1.0,
            'neg': 0.0
        }
    
    try:
        scores = analyzer.polarity_scores(str(text))
        return scores
    except Exception as e:
        print(f"⚠️ Error analyzing text: {e}")
        return {
            'compound': 0.0,
            'pos': 0.0,
            'neu': 1.0,
            'neg': 0.0
        }

def classify_sentiment(compound_score):
    """Classify sentiment based on compound score"""
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_sentiment(df, text_column='manual_translation'):
    """Perform VADER sentiment analysis on the DataFrame"""
    if text_column not in df.columns:
        print(f"❌ Error: Column '{text_column}' not found in DataFrame")
        print(f"📋 Available columns: {df.columns.tolist()}")
        return None
    
    print(f"🔍 Analyzing sentiment for {len(df)} entries...")
    print(f"📝 Using column: '{text_column}'")
    
    # Initialize VADER analyzer
    analyzer = initialize_vader()
    if analyzer is None:
        return None
    
    # Check for missing values
    missing_count = df[text_column].isna().sum()
    empty_count = (df[text_column].str.strip() == '').sum()
    print(f"📊 Missing values: {missing_count}")
    print(f"📊 Empty values: {empty_count}")
    
    # Apply sentiment analysis
    print("🔄 Processing sentiment analysis...")
    
    # Get sentiment scores for each text
    sentiment_scores = df[text_column].apply(lambda x: get_sentiment_scores(x, analyzer))
    
    # Extract individual scores
    df['polarity'] = sentiment_scores.apply(lambda x: x['compound'])
    df['positive_score'] = sentiment_scores.apply(lambda x: x['pos'])
    df['neutral_score'] = sentiment_scores.apply(lambda x: x['neu'])
    df['negative_score'] = sentiment_scores.apply(lambda x: x['neg'])
    
    # Classify sentiment labels
    df['sentiment_label'] = df['polarity'].apply(classify_sentiment)
    
    print("✅ Sentiment analysis completed!")
    
    return df

def display_sentiment_statistics(df):
    """Display detailed sentiment statistics"""
    print("\n" + "="*60)
    print("📊 SENTIMENT ANALYSIS RESULTS")
    print("="*60)
    
    # Basic statistics
    print(f"📈 Polarity Score Statistics:")
    print(f"   Mean: {df['polarity'].mean():.4f}")
    print(f"   Median: {df['polarity'].median():.4f}")
    print(f"   Std Dev: {df['polarity'].std():.4f}")
    print(f"   Min: {df['polarity'].min():.4f}")
    print(f"   Max: {df['polarity'].max():.4f}")
    
    # Sentiment distribution
    sentiment_counts = df['sentiment_label'].value_counts()
    print(f"\n📊 Sentiment Distribution:")
    print(sentiment_counts)
    
    print(f"\n📊 Sentiment Percentages:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment}: {count} ({percentage:.1f}%)")
    
    # Category-wise sentiment (if category column exists)
    if 'category' in df.columns:
        print(f"\n📊 Sentiment by Category:")
        category_sentiment = pd.crosstab(df['category'], df['sentiment_label'], margins=True)
        print(category_sentiment)
    
    return sentiment_counts

def create_sentiment_visualization(df, save_path='sentiment_distribution.png'):
    """Create comprehensive sentiment visualization"""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🧠 VADER Sentiment Analysis Results - USC Reddit Dataset', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Sentiment Distribution Bar Chart
    sentiment_counts = df['sentiment_label'].value_counts()
    colors = ['#2E8B57', '#FFD700', '#DC143C']  # Green, Gold, Red
    
    bars = axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values, 
                         color=colors, edgecolor='black', linewidth=1)
    axes[0, 0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Sentiment Label', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Posts/Comments', fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Polarity Score Distribution (Histogram)
    axes[0, 1].hist(df['polarity'], bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(df['polarity'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {df["polarity"].mean():.3f}')
    axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.5, label='Neutral')
    axes[0, 1].set_title('Polarity Score Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Polarity Score (Compound)', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].legend()
    
    # 3. Sentiment by Category (if category column exists)
    if 'category' in df.columns:
        category_sentiment = pd.crosstab(df['category'], df['sentiment_label'])
        category_sentiment.plot(kind='bar', ax=axes[1, 0], color=colors)
        axes[1, 0].set_title('Sentiment by Category', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Category', fontweight='bold')
        axes[1, 0].set_ylabel('Count', fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(title='Sentiment')
    else:
        # Alternative: Box plot of polarity scores
        df.boxplot(column='polarity', ax=axes[1, 0])
        axes[1, 0].set_title('Polarity Score Box Plot', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Polarity Score', fontweight='bold')
    
    # 4. Sentiment Scores Breakdown
    sentiment_scores = df[['positive_score', 'neutral_score', 'negative_score']].mean()
    axes[1, 1].pie(sentiment_scores.values, labels=sentiment_scores.index, autopct='%1.1f%%',
                   colors=['#90EE90', '#FFD700', '#FFB6C1'])
    axes[1, 1].set_title('Average Sentiment Scores Breakdown', fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved as {save_path}")
    
    # Show the plot
    plt.show()
    
    return sentiment_counts

def save_results(df, output_file):
    """Save the results to CSV file"""
    try:
        # Create output directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"✅ Results saved to {output_file}")
        
        # Display sample results
        print(f"\n📋 Sample Results:")
        sample_cols = ['manual_translation', 'polarity', 'sentiment_label', 
                      'positive_score', 'neutral_score', 'negative_score']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head())
        
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False

def main():
    """Main function to run the sentiment analysis pipeline"""
    print("🚀 Starting VADER Sentiment Analysis for USC Reddit Dataset")
    print("="*60)
    
    # File paths
    input_file = 'data/processed/05_usc_data_classified.csv'
    output_file = 'data/processed/06_usc_vader_sentiment_labeled.csv'
    
    # Load data
    df = load_data(input_file)
    if df is None:
        return
    
    # Check if manual_translation column exists
    if 'manual_translation' not in df.columns:
        print("❌ 'manual_translation' column not found!")
        print(f"📋 Available columns: {df.columns.tolist()}")
        return
    
    # Display basic info about the dataset
    print(f"\n📊 Dataset Info:")
    print(f"   Shape: {df.shape}")
    print(f"   Missing values in manual_translation: {df['manual_translation'].isna().sum()}")
    
    # Show a sample of the data
    print(f"\n📋 Sample data:")
    print(df[['manual_translation']].head(3))
    
    # Perform sentiment analysis
    df_with_sentiment = analyze_sentiment(df)
    if df_with_sentiment is None:
        return
    
    # Display statistics
    sentiment_counts = display_sentiment_statistics(df_with_sentiment)
    
    # Create visualization
    create_sentiment_visualization(df_with_sentiment, 'usc_sentiment_analysis.png')
    
    # Save results
    success = save_results(df_with_sentiment, output_file)
    
    if success:
        print("\n🎉 Sentiment analysis completed successfully!")
        print(f"📂 Input file: {input_file}")
        print(f"📂 Output file: {output_file}")
        print(f"📊 Visualization: usc_sentiment_analysis.png")
        
        # Final summary
        print(f"\n📊 Final Summary:")
        print(f"   Total entries analyzed: {len(df_with_sentiment)}")
        print(f"   Positive sentiment: {sentiment_counts.get('Positive', 0)} posts/comments")
        print(f"   Neutral sentiment: {sentiment_counts.get('Neutral', 0)} posts/comments")
        print(f"   Negative sentiment: {sentiment_counts.get('Negative', 0)} posts/comments")
        print(f"   Average polarity: {df_with_sentiment['polarity'].mean():.4f}")

if __name__ == "__main__":
    main()