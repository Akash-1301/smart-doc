# Expert Assistant — 

This  project is basically a natural language chatbot that can read and understand documents like PDFs and answer questions based on the content.It is a RAG Based Implementation.

---

## What It Does

- Loads PDFs and web content (Wikipedia, articles) about a topic.
- Creates semantic vector embeddings for efficient search.
- Uses a local language model (Ollama) to generate helpful answers.
- Provides a simple Streamlit web app interface where users can ask questions in plain English.
- Filters out profanity to keep conversations respectful.

---

## How To Use

1. **Ingest data** — Run the `ingest.py` script to load documents and build the vector store.
   python ingest.py
2.Start the app — Run the Streamlit app to chat with the assistant.
   streamlit run streamlit_app.py
3.Ask questions — Type your question in the text box and get answers based on your documents.
## Technologies Used
Python

Streamlit for UI

LangChain + HuggingFace embeddings for document search

FAISS vector store for fast similarity search

Ollama for local LLM-based answer generation

Web scraping with BeautifulSoup and Wikipedia API

## Why I Built This
I wanted to learn how RAG Works and to  combine vector search with language models and build a useful chatbot that can answer real questions from documents, without relying on cloud APIs. This project helped me understand LLM integration, data ingestion, and building interactive apps.

## About Me
Akash Kumar — Passionate about AI and building smart applications.
GitHub: Akash-1301

## License

This project is made for personal learning and demonstration purposes.  
Feel free to explore, learn, and get inspired — but **please do not copy, reuse, or distribute any part of this project without my permission**.

If you'd like to use or build upon this work, just reach out — I'm happy to collaborate or discuss.

© 2025 Akash Kumar(Akash-1301)
