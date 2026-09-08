# 🩺 MedBot • Intelligent Medical AI Assistant

An ultra-fast, evidence-grounded **Retrieval-Augmented Generation (RAG)** medical AI assistant built with **Streamlit, LangChain, Pinecone, Groq (`openai/gpt-oss-120b`), and Sentence Transformers**.

**MedBot** retrieves relevant medical excerpts from an indexed clinical knowledge base using **semantic search** and generates real-time, streamed answers with multi-turn conversation memory.

> ⚠️ **Important:** MedBot is designed to answer questions grounded in the indexed medical documents and conversational context. It is intended for educational and research purposes only and is not a substitute for professional medical advice.

---

## 🚀 Features

* 🩺 **Accurate Medical Question Answering** grounded in clinical reference literature.
* ⚡ **Ultra-Fast Groq Inference**: Powered by `openai/gpt-oss-120b` with real-time token streaming.
* 💬 **Multi-Turn Chat History**: Retains dialogue context for seamless follow-up questions (e.g., *"What are its symptoms?"*, *"How is it treated?"*).
* 🔍 **Source Transparency**: Collapsible citations drawer showing exact reference text chunks retrieved from Pinecone.
* 📌 **Pinecone Vector Database**: High-performance serverless vector index with `sentence-transformers/all-MiniLM-L6-v2`.
* 🎨 **Clinical Glassmorphic UI**: Sleek, modern interface with high-contrast medical theme and responsive layout.
* 🎚️ **Interactive Controls**: Adjustable Top-K retrieval chunks and temperature sliders.
* 📥 **Chat Export**: One-click Markdown conversation history export.
* 🧹 **Clear Chat**: Instant session reset.

---

## 🧠 Architecture & RAG Pipeline

```text
               User Question + Chat History
                            │
                            ▼
                 ┌────────────────────┐
                 │     Embedding      │
                 │   all-MiniLM-L6    │
                 └──────────┬─────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │  Pinecone Vector   │
                 │     Database       │
                 └──────────┬─────────┘
                            │
                      Retrieved Docs
                            │
                            ▼
                 ┌────────────────────┐
                 │  Context-Aware     │
                 │   RAG Prompt       │
                 └──────────┬─────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │        Groq        │
                 │ openai/gpt-oss-120b│
                 └──────────┬─────────┘
                            │
                            ▼
                   Live Streamed Answer
                + Source Citations Drawer
```

---

## 📚 Data Source

The chatbot's knowledge base is built from the following medical reference:

### **The Gale Encyclopedia of Medicine — Volume 1 (A–B)**

The source PDF contains medical reference information covering topics from **A to B** and is used as the primary knowledge source for the RAG system.

📄 **Source PDF:**
[Gale Encyclopedia of Medicine Vol. 1 (A–B) — Google Drive](https://docs.google.com/open?id=0B7HZIUBvCH1EbTZuejEwZ0s1R1k&utm_source=chatgpt.com)

The PDF content is processed, chunked, converted into embeddings, and indexed in **Pinecone** for semantic retrieval.

---

## 🧱 Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | 🐍 Python 3.10+ | Core application logic |
| **Frontend UI** | 🎨 Streamlit | Interactive web interface |
| **LLM Inference** | ⚡ Groq Cloud | Ultra-low latency `openai/gpt-oss-120b` |
| **RAG Orchestration** | 🦜 LangChain / `langchain-groq` | Prompt composition & pipeline |
| **Vector Database** | 📌 Pinecone | Serverless semantic search index |
| **Embeddings** | 🔤 Sentence-Transformers | 384-dimensional dense text vectors |

---

## 📁 Project Structure

```text
med-chatbot/
│
├── app.py              # Main Streamlit application with MedBot UI & Groq streaming
├── requirements.txt    # Project dependencies
├── .env                # API keys and environment variables
├── .gitignore          # Git ignore rules
└── Readme.md           # Project documentation
```

---

## 📦 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/milan-7417/med-chatbot.git
cd med-chatbot
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=med-chatbot
PINECONE_ENV=us-east-1

GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to:
```
http://localhost:8501
```

---

## 💡 Example Queries

* *"What is hypertension and what are its primary causes?"*
* Follow-up: *"What are common complications associated with it?"*
* *"What causes anemia and how is it clinically diagnosed?"*
* *"What are common triggers for asthma and how is an acute attack managed?"*

---

## ⚠️ Disclaimer

**MedBot** is an AI assistant intended strictly for educational, informational, and research purposes. It does not provide medical diagnoses or treatment recommendations. Always seek the advice of a qualified physician or healthcare provider with any medical questions.
