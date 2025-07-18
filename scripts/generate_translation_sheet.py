import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/processed/02_usc_cleaned_with_code_switch.csv")

# Select only the relevant columns for all entries (regardless of code-switching)
translation_df = df[['title', 'text', 'has_code_switching', 'content_type']].copy()

# Add an empty column for manual translation
translation_df['manual_translation'] = ''

# Save to new CSV
translation_df.to_csv("usc_manual_translation.csv", index=False)

print("✅ Translation sheet created: 'usc_manual_translation.csv'")
