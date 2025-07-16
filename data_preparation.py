import pandas as pd
from sklearn.model_selection import train_test_split

# Load datasets
imdb_df = pd.read_csv('data/imdb_review.csv')[['review', 'sentiment']]
#emo_df = pd.read_csv('data/emotion_dataset.csv')[['text', 'label']]
#emo_df.columns = ['review', 'sentiment']
#emo_df = pd.read_csv('data/emotion_dataset.csv')[['text', 'emotion']]
emo_df = pd.read_csv('data/emotion_dataset.csv', header=None, sep=';')
emo_df.columns = ['review', 'sentiment']  # Rename columns in code


# Normalize emotion labels
emo_df['sentiment'] = emo_df['sentiment'].apply(lambda x: 'positive' if x in ['joy', 'love'] else 'negative')

# Combine and shuffle
combined = pd.concat([imdb_df, emo_df], ignore_index=True)
combined.dropna(subset=['review', 'sentiment'], inplace=True)
combined['sentiment'] = combined['sentiment'].map({'positive': 1, 'negative': 0})
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Train-test split
train_df, test_df = train_test_split(combined, test_size=0.2, random_state=42)
train_df.to_csv('data/train_combined.csv', index=False)
test_df.to_csv('data/test_combined.csv', index=False)
