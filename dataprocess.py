import pandas as pd
from sklearn.model_selection import train_test_split

# Load IMDb dataset
imdb = pd.read_csv('data/imdb_reviews.csv')

# Assuming the IMDb dataset has 'review' and 'sentiment' columns
imdb = imdb[['review', 'sentiment']]
imdb['source'] = 'imdb'

# Load DailyDialog dataset
dailydialog = pd.read_csv('data/dailydialog.csv')

# Assuming DailyDialog has 'dialogue' and 'emotion' columns
dailydialog = dailydialog[['dialogue', 'emotion']]
dailydialog.rename(columns={'dialogue': 'review', 'emotion': 'sentiment'}, inplace=True)
dailydialog['source'] = 'dailydialog'

# Load Emotion Detection dataset
emotion = pd.read_csv('data/emotion_dataset.csv')

# Assuming Emotion Detection has 'text' and 'label' columns
emotion = emotion[['text', 'label']]
emotion.rename(columns={'text': 'review', 'label': 'sentiment'}, inplace=True)
emotion['source'] = 'emotion_detection'

# Combine datasets
combined = pd.concat([imdb, dailydialog, emotion], ignore_index=True)

# Save combined dataset
combined.to_csv('data/combined_dataset.csv', index=False)

# Split combined dataset for training and testing
train_df, test_df = train_test_split(combined, test_size=0.2, random_state=42)

train_df.to_csv('data/train_combined.csv', index=False)
test_df.to_csv('data/test_combined.csv', index=False)
