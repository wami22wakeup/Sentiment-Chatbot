import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import pandas as pd

# Load preprocessed data
df = pd.read_csv('data/train_combined.csv')
df = df.dropna(subset=['review', 'sentiment'])
df['sentiment'] = df['sentiment'].astype(int)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class CustomDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(df['review'].tolist(), truncation=True, padding=True, max_length=128)
        self.labels = torch.tensor(df['sentiment'].values, dtype=torch.long)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

dataset = CustomDataset(df)

# Training args
training_args = TrainingArguments(
    output_dir='./models/sentiment_model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="no",
    save_total_limit=2,
)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained('./models/sentiment_model')
tokenizer.save_pretrained('./models/sentiment_model')
