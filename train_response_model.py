import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load dataset (expects a CSV with 'input' and 'response' columns)
import pandas as pd

df = pd.read_csv("data/chatbot_dataset.csv")  # Your conversation dataset
df['text'] = df['input'] + tokenizer.eos_token + df['response']

dataset = Dataset.from_pandas(df[['text']])

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training configuration
training_args = TrainingArguments(
    output_dir="./models/response_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs_response",
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    evaluation_strategy="no",
    fp16=torch.cuda.is_available(),
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save
model.save_pretrained("./models/response_model")
tokenizer.save_pretrained("./models/response_model")
