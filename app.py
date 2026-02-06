import os
import streamlit as st
import torch
from dotenv import load_dotenv

# LangChain core
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Vector store + embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# LLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical RAG Chatbot")
st.caption("Answers are generated strictly from the uploaded medical documents.")


# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    st.markdown(
        """
        **Medical RAG Assistant**  
        Source - "The Gale Encyclopedia of Medicine" 
        """
    )

    top_k = st.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=5,
        value=3
    )

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built for research & educational use only.")


# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


# ---------------------------
# Cache heavy resources
# ---------------------------
@st.cache_resource(show_spinner="Loading embeddings...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def load_vectorstore(_embeddings):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    return PineconeVectorStore(
        index=index,
        embedding=_embeddings,
        text_key="text"
    )


@st.cache_resource(show_spinner="Loading language model...")
def load_llm():
    model_id = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.3,
        device=0 if torch.cuda.is_available() else -1
    )

    return HuggingFacePipeline(pipeline=hf_pipeline)


# ---------------------------
# Load components
# ---------------------------
embeddings = load_embeddings()
vectorstore = load_vectorstore(embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": top_k}
)

llm = load_llm()


# ---------------------------
# Prompt
# ---------------------------
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a medical assistant.
Answer ONLY from the context below.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------
# QA Chain
# ---------------------------
qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ---------------------------
# Chat memory
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ---------------------------
# Chat input
# ---------------------------
user_query = st.chat_input("Ask a medical question...")

if user_query:
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving documents and generating answer..."):
            answer = qa_chain.invoke(user_query)
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )


# ---------------------------
# Footer
# ---------------------------
st.divider()
st.caption("⚠️ Not a substitute for professional medical advice.")
