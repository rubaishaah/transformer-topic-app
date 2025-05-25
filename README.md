# Transformer Topic Modeling App

This is a simple web app built with **Streamlit** that demonstrates Transformer-based topic modeling using **BERTopic** and **SentenceTransformers**.

---

## 🚀 Features

* Input multiple texts (1 per line)
* Clean and preprocess texts
* Generate embeddings using `all-MiniLM-L6-v2`
* Extract topics using BERTopic
* View topics and their keywords
* Visualize topic keywords via WordClouds

---

## 🛠 Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/transformer-topic-app.git
cd transformer-topic-app
pip install -r requirements.txt
```

---

## ▶️ Running the App Locally

```bash
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Visit [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repo
4. Select `app.py` as the entry point
5. Deploy and share your app!

---

## 📋 Example Input

```
Deep learning models are revolutionizing NLP.
CRISPR is a new tool in biotechnology.
Transformers outperform RNNs in language modeling.
Genomics helps understand disease mechanisms.
```

---

## 📦 Dependencies

* streamlit
* bertopic
* sentence-transformers
* nltk
* pandas
* wordcloud
* matplotlib
* scikit-learn
* umap-learn
* hdbscan

---

## 📜 License

MIT License
