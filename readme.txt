# 🤖 AI Virtual Assistant (RAG App)

An intelligent **Retrieval-Augmented Generation (RAG)** powered Virtual Assistant that can read content from URLs and answer user questions with accurate, context-aware responses.

---

## 🚀 Features

* 🔗 Extracts knowledge from multiple URLs
* ✂️ Processes and chunks large text efficiently
* 🧠 Uses embeddings for semantic understanding
* 🤖 Answers queries using LLM (Groq - LLaMA 3)
* 📚 Provides source-backed responses
* ⚡ Interactive UI built with Streamlit

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Groq (LLaMA 3.3 70B)
* **Framework:** LangChain
* **Vector DB:** ChromaDB
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Web Scraping:** Unstructured + BeautifulSoup

---

## 📂 Project Structure

```
├── app.py              # Streamlit app (UI)
├── rag.py              # RAG pipeline logic
├── requirements.txt
├── .env                # API keys (not committed)
└── resources/
    └── vectorstore/    # Chroma DB storage
```
Link :- https://virtual-assistant-7.streamlit.app/
---

