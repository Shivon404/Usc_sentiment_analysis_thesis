import pandas as pd

# Load the dataset
df = pd.read_csv("data/processed/02_usc_cleaned_with_code_switch.csv")

# Columns to drop
columns_to_remove = [
    'bisaya_count', 'tagalog_count', 'conyo_count', 'mixed_patterns_found',
    'total_filipino_words', 'language_mix', 'usc_relevance_score',
    'code_switching_score', 'polarity', 'subjectivity', 'sentiment_category',
    'filipino_positive_words', 'filipino_negative_words'
]

# Drop the unwanted columns if they exist
df_filtered = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

# Add an empty column for manual translation
df_filtered['manual_translation'] = ''

# Save to new CSV
df_filtered.to_csv("data/processed/03_usc_cleaned_data_translation.csv", index=False)

print("✅ Cleaned translation sheet created: 'data/processed/02_usc_cleaned_for_translation.csv'")
