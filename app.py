import streamlit as st
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

st.set_page_config(page_title="Transformer Topic Modeling", layout="wide")
st.title("🔍 BERTopic Transformer Topic Modeling")

st.markdown("""
This app uses **BERTopic** to detect topics in a collection of short documents.
Paste or upload your own dataset below (1 text per line).
""")

# User input
text_input = st.text_area("Enter your texts (one per line):", height=200)

if st.button("Run Topic Modeling"):
    with st.spinner("Processing texts..."):
        # Preprocess function
        def preprocess(texts):
            stop_words = set(stopwords.words('english'))
            cleaned = []
            for text in texts:
                words = word_tokenize(re.sub(r"[^a-zA-Z\s]", "", text.lower()))
                filtered = [w for w in words if w not in stop_words]
                cleaned.append(" ".join(filtered))
            return cleaned

        # Step 1: Parse user input
        docs = [line.strip() for line in text_input.split("\n") if line.strip()]

        # Step 2: Preprocess text
        cleaned_docs = preprocess(docs)

        # Step 3: Embedding + BERTopic
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(cleaned_docs, show_progress_bar=False)

        topic_model = BERTopic(verbose=False)
        topics, probs = topic_model.fit_transform(cleaned_docs, embeddings)

        # Step 4: Show results
        topic_info = topic_model.get_topic_info()
        st.subheader("📌 Detected Topics")
        st.dataframe(topic_info)

        # Optional: Word cloud
        st.subheader("☁️ Word Cloud of a Topic")
        selected_topic = st.number_input("Select Topic Number", min_value=0, max_value=len(topic_info)-1, value=0)

        topic_words = dict(topic_model.get_topic(selected_topic))
        if topic_words:
            wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(topic_words)
            st.pyplot(plt.imshow(wc, interpolation='bilinear'))
            plt.axis("off")
            plt.close()
        else:
            st.write("No words found for the selected topic.")
