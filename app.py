import streamlit as st
import pickle
import re, string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# ── Load Model ────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words("english"))

# ── Clean Text ────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detector")
st.caption("Enter a news headline or article to check if it's real or fake.")

# ── Input ─────────────────────────────────────────────────────
news_input = st.text_area("📝 Paste your news text here:", height=200)

if st.button("🔍 Analyze"):
    if news_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        confidence = model.predict_proba(vectorized)[0]

        st.divider()

        if prediction == 1:
            st.success("✅ This news appears to be **REAL**")
        else:
            st.error("🚨 This news appears to be **FAKE**")

        real_conf = confidence[1] * 100
        fake_conf = confidence[0] * 100

        st.metric("Real Confidence", f"{real_conf:.1f}%")
        st.metric("Fake Confidence", f"{fake_conf:.1f}%")

        st.progress(int(real_conf))