# streamlit_app.py
import streamlit as st
from app import predict_sentiment, generate_response

st.title("🎯 Sentiment-Aware Chatbot")

user_input = st.text_input("You:", "")

if user_input:
    sentiment = predict_sentiment(user_input)
    response = generate_response(user_input, sentiment)

    st.markdown(f"**Sentiment:** {sentiment}")
    st.markdown(f"🤖 **Bot:** {response}")
