# app.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer

# Load Sentiment Classifier
sentiment_tokenizer = BertTokenizer.from_pretrained('./models/sentiment_model')
sentiment_model = BertForSequenceClassification.from_pretrained('./models/sentiment_model')
sentiment_model.eval()

# Load Response Generator (GPT-2)
response_tokenizer = GPT2Tokenizer.from_pretrained('./models/response_model')
response_model = GPT2LMHeadModel.from_pretrained('./models/response_model')
response_model.eval()

# Predict sentiment (positive/negative)
def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return "positive" if pred == 1 else "negative"

# Generate chatbot response based on text and sentiment
def generate_response(text, sentiment):
    prompt = f"[{sentiment}] {text}"
    input_ids = response_tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output_ids = response_model.generate(
            input_ids,
            max_length=100,
            pad_token_id=response_tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
    response = response_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()
