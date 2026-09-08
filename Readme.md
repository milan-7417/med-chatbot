# 🩺 Medical RAG Chatbot

An intelligent **Retrieval-Augmented Generation (RAG)** medical chatbot built with **Streamlit, LangChain, Pinecone, and Hugging Face Qwen2.5-3B-Instruct**.

The chatbot retrieves relevant information from an indexed medical knowledge base using **semantic search** and generates answers using a locally running Hugging Face LLM.

> ⚠️ **Important:** The chatbot is designed to answer questions **strictly from the indexed medical documents**. If the required information is not available in the retrieved context, it responds that it does not know based on the provided documents.

---

## 🚀 Features

* 🩺 **Medical Question Answering**
* 📚 **Retrieval-Augmented Generation (RAG)**
* 🔍 **Semantic Search with Pinecone**
* 🧠 **Hugging Face Qwen2.5-3B-Instruct LLM**
* 🤗 **Fully Local LLM Inference**
* 💬 **Interactive Chat Interface**
* 🎨 **Modern & Colorful Streamlit UI**
* 📖 **Medical Knowledge Base Integration**
* 🎚️ **Adjustable Top-K Document Retrieval**
* 🧹 **Clear Conversation Functionality**
* ⚡ **Cached Models and Resources**
* ☁️ **Deployable Streamlit Application**

---

## 🧠 How It Works

The application follows a Retrieval-Augmented Generation pipeline:

```text
                 User Question
                       │
                       ▼
              ┌─────────────────┐
              │    Embedding    │
              │  MiniLM Model   │
              └────────┬────────┘
                       │
                       ▼
                ┌─────────────┐
                │   Pinecone  │
                │ Vector Store│
                └──────┬──────┘
                       │
                 Relevant Docs
                       │
                       ▼
              ┌─────────────────┐
              │  RAG Prompt     │
              │ Context + Query │
              └────────┬────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │ Qwen2.5-3B-Instruct   │
          │ Hugging Face LLM      │
          └────────────┬───────────┘
                       │
                       ▼
                 Final Answer
```

---

## 📚 Data Source

The chatbot's knowledge base is built from the following medical reference:

### **The Gale Encyclopedia of Medicine — Volume 1 (A–B)**

The source PDF contains medical reference information covering topics from **A to B** and is used as the primary knowledge source for the RAG system.

📄 **Source PDF:**
[Gale Encyclopedia of Medicine Vol. 1 (A–B) — Google Drive](https://docs.google.com/open?id=0B7HZIUBvCH1EbTZuejEwZ0s1R1k&utm_source=chatgpt.com)

The PDF content is processed, chunked, converted into embeddings, and indexed in **Pinecone** for semantic retrieval.

### Data Processing Pipeline

```text
Gale Encyclopedia PDF
          │
          ▼
     Text Extraction
          │
          ▼
       Chunking
          │
          ▼
  Sentence Transformer
          │
          ▼
      Embeddings
          │
          ▼
       Pinecone
          │
          ▼
    Semantic Retrieval
```

---

## 🤖 LLM

The chatbot uses:

```text
Qwen/Qwen2.5-3B-Instruct
```

The model is loaded directly from Hugging Face using the Transformers library.

```python
model_id = "Qwen/Qwen2.5-3B-Instruct"
```

No OpenAI, Groq, or other external LLM API is required.

---

## 🔍 Retrieval System

The chatbot uses **Pinecone** as its vector database.

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model converts the user query into a vector representation.

Pinecone then retrieves the most semantically relevant medical document chunks.

The number of retrieved documents can be adjusted from the Streamlit sidebar using the **Top-K** slider.

---

## 🧱 Tech Stack

| Technology               | Purpose                           |
| ------------------------ | --------------------------------- |
| 🐍 Python                | Core programming language         |
| 🎨 Streamlit             | Web application & UI              |
| 🦜 LangChain             | RAG pipeline orchestration        |
| 📌 Pinecone              | Vector database & semantic search |
| 🤗 Hugging Face          | LLM ecosystem                     |
| 🧠 Qwen2.5-3B-Instruct   | Language model                    |
| 🔤 Sentence Transformers | Text embeddings                   |
| 🔥 PyTorch               | Model inference                   |
| 🤖 Transformers          | Hugging Face model loading        |

---

## 📁 Project Structure

```text
medical-rag-chatbot/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/medical-rag-chatbot.git
```

```bash
cd medical-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
```

> 🔒 Never commit your `.env` file or API keys to GitHub.

Add this to `.gitignore`:

```text
.env
venv/
__pycache__/
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🎨 User Interface

The application provides a modern interactive interface with:

* 🌈 Gradient-based visual design
* 💬 Chat-style conversation
* 🩺 Medical assistant branding
* 📚 Adjustable document retrieval
* 💡 Quick medical question suggestions
* 🧹 Clear conversation button
* 📱 Responsive Streamlit layout
* 🧠 Model and knowledge-base information cards

---

## 🛡️ RAG Grounding

The chatbot is designed to minimize hallucinations by restricting responses to the retrieved medical context.

The prompt instructs the model to:

* Use only retrieved medical information.
* Avoid unsupported medical claims.
* Avoid inventing information.
* State when the answer is unavailable.
* Avoid unsupported diagnosis.
* Provide concise and understandable responses.

If the retrieved documents don't contain the required information, the chatbot responds:

```text
I don't know based on the provided medical documents.
```

---

## ☁️ Deployment

The application can be deployed on platforms that support Streamlit and Python applications.

Configure the following environment variables:

```text
PINECONE_API_KEY
PINECONE_INDEX_NAME
```

The Hugging Face model is downloaded when the application initializes.

> ⚠️ **Memory requirement:** Qwen2.5-3B-Instruct is considerably larger than FLAN-T5-base, so the deployment environment should have sufficient RAM.

---

## ⚠️ Medical Disclaimer

This project is intended for **educational and research purposes only**.

It is **not a medical diagnostic system** and should not be used as a replacement for a qualified healthcare professional.

Always consult a licensed medical professional for diagnosis, treatment, medication, or emergency medical decisions.

---

## 🔮 Future Improvements

* 🧠 Conversation-aware RAG
* 📌 Source/document citations
* 📄 PDF upload and indexing
* 🌐 Multilingual medical Q&A
* 🎤 Voice-based interaction
* 🔎 Advanced hybrid search
* 📊 Retrieval confidence scores
* ⚡ Quantized LLM inference
* 🔐 Authentication and user management
* 🩻 Medical document visualization

---

## 👨‍💻 Author

**Milan**

Built with ❤️ using Python, LangChain, Pinecone, Hugging Face, and Streamlit.

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
