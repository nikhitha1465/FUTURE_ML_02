import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

# Load your saved models
category_model = joblib.load('models/category_model.pkl')
category_vectorizer = joblib.load('models/category_vectorizer.pkl')
priority_model = joblib.load('models/priority_model.pkl')
priority_vectorizer = joblib.load('models/priority_vectorizer.pkl')

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def predict_ticket(text):
    cleaned = clean_text(text)
    cleaned = lemmatize(cleaned)

    category = category_model.predict(category_vectorizer.transform([cleaned]))[0]
    priority = priority_model.predict(priority_vectorizer.transform([cleaned]))[0]

    return category, priority

# --- UI ---
st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫")
st.title("🎫 Support Ticket Classifier")
st.write("Paste a support ticket below to auto-classify its category and priority.")

ticket_text = st.text_area("Ticket text", height=150, placeholder="e.g. My laptop won't turn on and I need it urgently...")

if st.button("Classify Ticket"):
    if ticket_text.strip():
        category, priority = predict_ticket(ticket_text)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Category", category)
        with col2:
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            st.metric("Priority", f"{priority_color.get(priority, '')} {priority.upper()}")
    else:
        st.warning("Please enter some ticket text first.")

with st.expander("ℹ️ About this model"):
    st.write("""
    - **Category model**: Linear SVM on TF-IDF features, 84.8% accuracy
    - **Priority model**: Linear SVM on TF-IDF features, 95.8% accuracy (priority derived from urgency keywords in text)
    """)