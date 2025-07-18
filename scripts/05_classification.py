import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from pathlib import Path

def load_data(file_path):
    """Load the CSV file and return DataFrame"""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def define_keywords():
    """Define keyword dictionaries for each category"""
    keywords = {
        'faculty_evaluation': [
            # Teaching and instruction
            'professor', 'prof', 'instructor', 'teacher', 'faculty', 'teaching',
            'lecture', 'lecturer', 'class', 'course', 'subject', 'lesson',
            'exam', 'test', 'quiz', 'assignment', 'homework', 'project',
            'grade', 'grading', 'graded', 'marks', 'scoring', 'evaluation',
            # Teaching quality
            'explain', 'explanation', 'clear', 'confusing', 'understand',
            'difficult', 'easy', 'hard', 'challenging', 'boring', 'interesting',
            'engaging', 'passionate', 'knowledgeable', 'helpful', 'supportive',
            # Course-related
            'syllabus', 'curriculum', 'textbook', 'material', 'content',
            'attendance', 'participation', 'discussion', 'presentation',
            'feedback', 'office hours', 'consultation', 'review', 'rating'
        ],
        
        'admission': [
            # Application process
            'admission', 'application', 'apply', 'applying', 'applicant',
            'entrance', 'entry', 'requirements', 'requirement', 'qualify',
            'qualification', 'eligible', 'eligibility', 'criteria',
            # Tests and exams
            'entrance exam', 'admission test', 'qualifying exam', 'upcat',
            'nmat', 'lat', 'sat', 'act', 'gre', 'gmat', 'toefl', 'ielts',
            # Transfer and programs
            'transfer', 'transferring', 'transferee', 'shiftee', 'shifting',
            'program', 'course', 'major', 'degree', 'bachelor', 'master',
            'doctoral', 'phd', 'graduate', 'undergraduate', 'freshmen',
            # Application components
            'transcript', 'gpa', 'gwa', 'grades', 'documents', 'requirements',
            'interview', 'portfolio', 'essay', 'personal statement',
            'recommendation', 'reference', 'scholarship', 'financial aid',
            # Status and results
            'accepted', 'rejected', 'waitlist', 'pending', 'result',
            'notification', 'deadline', 'cutoff', 'quota', 'slots'
        ],
        
        'student_experience': [
            # Academic life
            'student', 'academic', 'academics', 'study', 'studying', 'studies',
            'semester', 'trimester', 'quarter', 'midterm', 'finals', 'period',
            'workload', 'stress', 'pressure', 'deadline', 'busy', 'hectic',
            # Grades and performance
            'grade', 'gpa', 'gwa', 'dean\'s list', 'honor', 'magna cum laude',
            'cum laude', 'latin honor', 'scholarship', 'academic performance',
            # Student life
            'college life', 'university life', 'campus life', 'social life',
            'friends', 'classmates', 'groupmates', 'blockmates', 'barkada',
            'relationship', 'dating', 'party', 'event', 'celebration',
            # Organizations and activities
            'org', 'organization', 'club', 'society', 'fraternity', 'sorority',
            'student council', 'student government', 'leadership', 'officer',
            'member', 'join', 'participate', 'involvement', 'extracurricular',
            'volunteer', 'community service', 'outreach', 'competition',
            # Daily life
            'schedule', 'routine', 'balance', 'time management', 'lifestyle',
            'health', 'mental health', 'wellness', 'burnout', 'exhaustion',
            'motivation', 'inspiration', 'goal', 'achievement', 'success',
            # Curriculum and subjects
            'curriculum', 'subject', 'course', 'elective', 'major', 'minor',
            'specialization', 'track', 'strand', 'program', 'degree',
            'internship', 'practicum', 'thesis', 'capstone', 'research'
        ],
        
        'school_environment': [
            # Campus and facilities
            'campus', 'facility', 'facilities', 'building', 'classroom',
            'laboratory', 'library', 'gym', 'gymnasium', 'auditorium',
            'cafeteria', 'canteen', 'food court', 'dining', 'restaurant',
            'dormitory', 'residence hall', 'housing', 'accommodation',
            # Infrastructure
            'wifi', 'internet', 'connection', 'network', 'computer', 'laptop',
            'projector', 'equipment', 'technology', 'aircon', 'air conditioning',
            'ventilation', 'lighting', 'electricity', 'power', 'water',
            # Safety and security
            'security', 'safety', 'guard', 'cctv', 'surveillance', 'lighting',
            'emergency', 'evacuation', 'drill', 'protocol', 'incident',
            'crime', 'theft', 'harassment', 'bullying', 'violence',
            # Accessibility and transportation
            'parking', 'transportation', 'shuttle', 'jeepney', 'bus', 'mrt',
            'lrt', 'commute', 'traffic', 'accessibility', 'wheelchair',
            'ramp', 'elevator', 'escalator', 'stairs', 'entrance', 'exit',
            # Environment and atmosphere
            'environment', 'atmosphere', 'ambiance', 'cleanliness', 'hygiene',
            'maintenance', 'renovation', 'construction', 'noise', 'quiet',
            'peaceful', 'crowded', 'spacious', 'comfortable', 'convenient',
            # Services
            'registrar', 'cashier', 'accounting', 'finance', 'clinic',
            'health center', 'medical', 'pharmacy', 'bookstore', 'supplies',
            'printing', 'photocopy', 'scanner', 'service', 'staff', 'personnel'
        ]
    }
    
    return keywords

def classify_text(text, keywords):
    """Classify a single text entry based on keyword matching"""
    if pd.isna(text) or text.strip() == '':
        return 'others'
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Count keyword matches for each category
    category_scores = {}
    
    for category, category_keywords in keywords.items():
        score = 0
        for keyword in category_keywords:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            score += matches
        
        category_scores[category] = score
    
    # Return the category with the highest score
    max_score = max(category_scores.values())
    
    if max_score == 0:
        return 'others'
    
    # Return the category with the highest score
    best_category = max(category_scores, key=category_scores.get)
    return best_category

def classify_dataframe(df, text_column='preprocessed_text'):
    """Classify all entries in the DataFrame"""
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in DataFrame")
        return None
    
    print(f"Classifying {len(df)} entries...")
    
    # Define keywords
    keywords = define_keywords()
    
    # Apply classification
    df['category'] = df[text_column].apply(lambda x: classify_text(x, keywords))
    
    # Print classification summary
    category_counts = df['category'].value_counts()
    print("\nClassification Summary:")
    print(category_counts)
    
    return df

def create_visualization(df, save_path='classification_results.png'):
    """Create and save a bar chart of classification results"""
    category_counts = df['category'].value_counts()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create bar plot
    bars = plt.bar(category_counts.index, category_counts.values, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
                   edgecolor='black', linewidth=0.7)
    
    # Customize the plot
    plt.title('Distribution of Reddit Posts/Comments by Category', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Category', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Posts/Comments', fontsize=12, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as {save_path}")
    
    # Show the plot
    plt.show()
    
    return category_counts

def main():
    """Main function to run the classification pipeline"""
    # File paths
    input_file = 'data/processed/04_usc_data_preprocessed.csv'
    output_file = 'data/processed/05_usc_data_classified.csv'
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(input_file)
    if df is None:
        return
    
    # Check if preprocessed_text column exists
    if 'preprocessed_text' not in df.columns:
        print("Available columns:", df.columns.tolist())
        return
    
    # Display basic info about the dataset
    print(f"\nDataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing values in preprocessed_text: {df['preprocessed_text'].isna().sum()}")
    
    # Classify the data
    df_classified = classify_dataframe(df)
    if df_classified is None:
        return
    
    # Save results
    try:
        df_classified.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return
    
    # Create visualization
    category_counts = create_visualization(df_classified, 'classification_visualization.png')
    
    # Display detailed statistics
    print(f"\nDetailed Statistics:")
    print(f"Total entries: {len(df_classified)}")
    print(f"Categories distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(df_classified)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    # Show sample classifications
    print(f"\nSample Classifications:")
    for category in category_counts.index:
        sample = df_classified[df_classified['category'] == category]['preprocessed_text'].iloc[0]
        print(f"\n{category.upper()}:")
        print(f"  Sample: {sample[:100]}...")

if __name__ == "__main__":
    main()