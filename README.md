
# 💬 Sentiment-Aware Chatbot 🤖🧠

A conversational AI chatbot that **detects user sentiment using BERT** and generates **context-aware responses using GPT-2**, built with PyTorch and HuggingFace Transformers. The app runs via a **Streamlit interface**.

---

## 📂 Project Structure

```
Sentiment-Chatbot/
│
├── data_preparation.py            # Combines and preprocesses IMDb & Emotion datasets
├── train_sentiment_model.py       # Fine-tunes BERT for sentiment classification
├── train_response_model.py        # Fine-tunes GPT-2 for chatbot responses
├── streamlit_app.py               # Streamlit web UI
├── app.py                         # Optional: backend logic helper
│
├── models/
│   ├── sentiment_model/           # Saved BERT model
│   └── response_model/            # Saved GPT-2 model
│
└── data/
    ├── imdb_reviews.csv
    ├── emotion_dataset.csv
    └── combined_dataset.csv
```

---

## 🛠️ Tech Stack

* **Python 3.12**
* **PyTorch**
* **Transformers (HuggingFace)**
* **Streamlit**
* **Pandas & Sklearn**

---

## 📥 Datasets Used

* **IMDb Reviews** – Binary sentiment classification (`positive`/`negative`)
* **Emotion Dataset** – Multiclass emotion labels (joy, anger, sadness, etc.)

> Ensure these are saved in the `data/` folder with correct column names before running.

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/<your-username>/Sentiment-Chatbot.git
cd Sentiment-Chatbot
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv newenv
newenv\Scripts\activate      # Windows
# source newenv/bin/activate # Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline

```bash
python data_preparation.py
python train_sentiment_model.py
python train_response_model.py
streamlit run streamlit_app.py
```



## 📸 Preview

The chatbot interface takes a user message, detects the sentiment/emotion, and generates a GPT-2-based response accordingly.



## ✨ Features

* 🎯 Fine-tuned **BERT** for accurate sentiment detection.
* 💬 Natural **GPT-2** based dialogue generation.
* 🌐 **Streamlit** interface for interactive usage.
* 🔄 Modular code: easy to expand with more models/datasets.



## 🤝 Contributing

Pull requests and suggestions are welcome! Make sure to:

* Fork the repository
* Create a new branch
* Push changes and create a PR

## 📄 License

MIT License
