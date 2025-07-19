import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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

def merge_sentiment_labels(df):
    """Merge final_sentiment_labeling with sentiment_label"""
    print("🔄 Merging manual corrections with VADER sentiment labels...")
    
    # Check if required columns exist
    required_cols = ['sentiment_label', 'final_sentiment_labeling']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        print(f"📋 Available columns: {df.columns.tolist()}")
        return None
    
    # Create a copy for processing
    df_merged = df.copy()
    
    # Count manual corrections
    manual_corrections = df_merged['final_sentiment_labeling'].notna().sum()
    vader_kept = df_merged['final_sentiment_labeling'].isna().sum()
    
    print(f"📊 Manual corrections: {manual_corrections}")
    print(f"📊 VADER labels kept: {vader_kept}")
    
    # Create the combined sentiment label
    # If final_sentiment_labeling is not empty/null, use it; otherwise use sentiment_label
    df_merged['combined_sentiment_label'] = df_merged['final_sentiment_labeling'].fillna(df_merged['sentiment_label'])
    
    # Display the merging results
    print(f"\n📊 Sentiment Label Comparison:")
    print("Original VADER vs Final Combined:")
    
    # Compare original vs final
    comparison = pd.crosstab(df_merged['sentiment_label'], 
                           df_merged['combined_sentiment_label'], 
                           margins=True, margins_name="Total")
    print(comparison)
    
    # Show which ones were manually corrected
    if manual_corrections > 0:
        print(f"\n🔍 Manual Corrections Summary:")
        corrections_df = df_merged[df_merged['final_sentiment_labeling'].notna()][
            ['sentiment_label', 'final_sentiment_labeling', 'combined_sentiment_label']
        ].copy()
        
        # Count correction types
        correction_summary = corrections_df.groupby(['sentiment_label', 'final_sentiment_labeling']).size().reset_index(name='count')
        print("Changes made:")
        for _, row in correction_summary.iterrows():
            print(f"   {row['sentiment_label']} → {row['final_sentiment_labeling']}: {row['count']} cases")
    
    return df_merged

def display_sentiment_statistics(df):
    """Display detailed sentiment statistics for both original and combined labels"""
    print("\n" + "="*70)
    print("📊 SENTIMENT ANALYSIS RESULTS - BEFORE & AFTER MANUAL CORRECTIONS")
    print("="*70)
    
    # Original VADER statistics
    print(f"\n🤖 Original VADER Results:")
    vader_counts = df['sentiment_label'].value_counts()
    print(vader_counts)
    
    print(f"\n📊 VADER Percentages:")
    for sentiment, count in vader_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment}: {count} ({percentage:.1f}%)")
    
    # Combined (Final) statistics
    print(f"\n✨ Final Combined Results:")
    final_counts = df['combined_sentiment_label'].value_counts()
    print(final_counts)
    
    print(f"\n📊 Final Percentages:")
    for sentiment, count in final_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {sentiment}: {count} ({percentage:.1f}%)")
    
    # Show the changes
    print(f"\n📈 Changes Summary:")
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        original = vader_counts.get(sentiment, 0)
        final = final_counts.get(sentiment, 0)
        change = final - original
        if change != 0:
            print(f"   {sentiment}: {original} → {final} (change: {change:+d})")
        else:
            print(f"   {sentiment}: {original} → {final} (no change)")
    
    # Basic statistics for polarity scores
    if 'polarity' in df.columns:
        print(f"\n📈 Polarity Score Statistics:")
        print(f"   Mean: {df['polarity'].mean():.4f}")
        print(f"   Median: {df['polarity'].median():.4f}")
        print(f"   Std Dev: {df['polarity'].std():.4f}")
        print(f"   Min: {df['polarity'].min():.4f}")
        print(f"   Max: {df['polarity'].max():.4f}")
    
    # Category-wise sentiment (if category column exists)
    if 'category' in df.columns:
        print(f"\n📊 Final Sentiment by Category:")
        category_sentiment = pd.crosstab(df['category'], df['combined_sentiment_label'], margins=True)
        print(category_sentiment)
    
    return vader_counts, final_counts

def create_comparison_visualization(df, save_path='sentiment_comparison.png'):
    """Create comprehensive before/after sentiment visualization"""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Sentiment Analysis Results - USC Reddit Dataset', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Colors for consistency
    colors = ['#2E8B57', '#FFD700', '#DC143C']  # Green, Gold, Red
    
    # 1. Original VADER Sentiment Distribution
    vader_counts = df['sentiment_label'].value_counts()
    bars1 = axes[0, 0].bar(vader_counts.index, vader_counts.values, 
                          color=colors, edgecolor='black', linewidth=1)
    axes[0, 0].set_title('Original VADER Sentiment', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Sentiment Label', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Posts/Comments', fontweight='bold')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Final Combined Sentiment Distribution
    final_counts = df['combined_sentiment_label'].value_counts()
    bars2 = axes[0, 1].bar(final_counts.index, final_counts.values, 
                          color=colors, edgecolor='black', linewidth=1)
    axes[0, 1].set_title('Final (Manual-Corrected) Sentiment', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Sentiment Label', fontweight='bold')
    axes[0, 1].set_ylabel('Number of Posts/Comments', fontweight='bold')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Side-by-Side Comparison
    comparison_data = pd.DataFrame({
        'VADER': vader_counts,
        'Manual-Corrected': final_counts
    }).fillna(0)
    
    x_pos = np.arange(len(comparison_data.index))
    width = 0.35
    
    bars3 = axes[1, 0].bar(x_pos - width/2, comparison_data['VADER'], width, 
                          label='VADER', color='lightblue', edgecolor='black')
    bars4 = axes[1, 0].bar(x_pos + width/2, comparison_data['Manual-Corrected'], width,
                          label='Manual-Corrected', color='lightcoral', edgecolor='black')
    
    axes[1, 0].set_title('VADER vs Manual-Corrected Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Sentiment Label', fontweight='bold')
    axes[1, 0].set_ylabel('Number of Posts/Comments', fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(comparison_data.index)
    axes[1, 0].legend()
    
    # Add value labels for comparison chart
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # 4. Manual Corrections Breakdown
    manual_corrections = df[df['final_sentiment_labeling'].notna()]
    if len(manual_corrections) > 0:
        correction_changes = pd.crosstab(manual_corrections['sentiment_label'], 
                                       manual_corrections['final_sentiment_labeling'])
        
        # Create heatmap for corrections
        sns.heatmap(correction_changes, annot=True, fmt='d', cmap='YlOrRd', 
                   ax=axes[1, 1], cbar_kws={'label': 'Number of Changes'})
        axes[1, 1].set_title('Manual Corrections Heatmap', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Corrected To', fontweight='bold')
        axes[1, 1].set_ylabel('Original VADER Label', fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Manual\nCorrections Made', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=14, fontweight='bold')
        axes[1, 1].set_title('Manual Corrections', fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison visualization saved as {save_path}")
    
    # Show the plot
    plt.show()
    
    return vader_counts, final_counts

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
        sample_cols = ['manual_translation', 'sentiment_label', 'final_sentiment_labeling', 
                      'combined_sentiment_label', 'polarity']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head())
        
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False

def main():
    """Main function to merge sentiment labels and create visualizations"""
    print("🚀 Merging Manual Corrections with VADER Sentiment Labels")
    print("="*65)
    
    # File paths
    input_file = 'data/processed/07_usc_manual_sentiment_labeled.csv'
    output_file = 'data/processed/08_usc_final_vader_sentiment_labeled.csv'
    
    # Load data
    df = load_data(input_file)
    if df is None:
        return
    
    # Display basic info about the dataset
    print(f"\n📊 Dataset Info:")
    print(f"   Shape: {df.shape}")
    if 'final_sentiment_labeling' in df.columns:
        manual_corrections = df['final_sentiment_labeling'].notna().sum()
        print(f"   Manual corrections found: {manual_corrections}")
    else:
        print("❌ 'final_sentiment_labeling' column not found!")
        return
    
    # Merge sentiment labels
    df_merged = merge_sentiment_labels(df)
    if df_merged is None:
        return
    
    # Display statistics
    vader_counts, final_counts = display_sentiment_statistics(df_merged)
    
    # Create comparison visualization
    create_comparison_visualization(df_merged, 'usc_sentiment_comparison.png')
    
    # Save results
    success = save_results(df_merged, output_file)
    
    if success:
        print("\n🎉 Sentiment label merging completed successfully!")
        print(f"📂 Input file: {input_file}")
        print(f"📂 Output file: {output_file}")
        print(f"📊 Visualization: usc_sentiment_comparison.png")
        
        # Final summary
        manual_corrections = df_merged['final_sentiment_labeling'].notna().sum()
        print(f"\n📊 Final Summary:")
        print(f"   Total entries processed: {len(df_merged)}")
        print(f"   Manual corrections applied: {manual_corrections}")
        print(f"   VADER labels kept: {len(df_merged) - manual_corrections}")
        print(f"\n📈 Final Sentiment Distribution:")
        for sentiment, count in final_counts.items():
            percentage = (count / len(df_merged)) * 100
            print(f"   {sentiment}: {count} ({percentage:.1f}%)")
        
        if 'polarity' in df_merged.columns:
            print(f"   Average polarity: {df_merged['polarity'].mean():.4f}")

if __name__ == "__main__":
    main()