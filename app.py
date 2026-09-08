import os
import time
import json
import datetime
import streamlit as st
from dotenv import load_dotenv

# ============================================================
# LANGCHAIN & GROQ & RAG IMPORTS
# ============================================================
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_groq import ChatGroq

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="MedBot • Medical AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "med-chatbot")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Default Model as requested
GROQ_MODEL = "openai/gpt-oss-120b"

# ============================================================
# CUSTOM CSS (PREMIUM CLINICAL GLASSMORPHISM)
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Background with subtle medical mesh gradient */
    .stApp {
        background:
            radial-gradient(circle at 5% 10%, rgba(99, 102, 241, 0.08), transparent 30%),
            radial-gradient(circle at 95% 15%, rgba(6, 182, 212, 0.08), transparent 30%),
            radial-gradient(circle at 50% 90%, rgba(16, 185, 129, 0.05), transparent 40%),
            linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #eef2ff 100%);
        background-attachment: fixed;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #0e7490 100%);
        padding: 26px 32px;
        border-radius: 20px;
        margin-bottom: 22px;
        color: #ffffff;
        box-shadow: 0 16px 36px -12px rgba(30, 27, 75, 0.30);
        border: 1px solid rgba(255, 255, 255, 0.12);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(6, 182, 212, 0.22) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }

    .hero-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
    }

    .hero-title {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .hero-subtitle {
        font-size: 14px;
        margin-top: 5px;
        opacity: 0.90;
        line-height: 1.5;
        max-width: 680px;
    }

    .hero-badges {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 12px;
    }

    .hero-badge {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        color: #f8fafc;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 50%, #f0fdf4 100%);
        border-right: 1px solid rgba(226, 232, 240, 0.8);
    }

    .sidebar-header-card {
        background: linear-gradient(135deg, #3730a3 0%, #0284c7 100%);
        padding: 16px 18px;
        border-radius: 14px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 10px 22px -5px rgba(55, 48, 163, 0.22);
    }

    .sidebar-header-title {
        font-size: 20px;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.5px;
    }

    .sidebar-header-sub {
        font-size: 11.5px;
        opacity: 0.88;
        margin-top: 2px;
    }

    .sidebar-section-title {
        font-size: 11px;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 16px;
        margin-bottom: 8px;
    }

    .meta-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        padding: 10px 12px;
        border-radius: 12px;
        margin-bottom: 6px;
        border: 1px solid rgba(203, 213, 225, 0.6);
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.02);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .meta-card-label {
        font-size: 11.5px;
        color: #64748b;
        font-weight: 600;
    }

    .meta-card-value {
        font-size: 11.5px;
        color: #0f172a;
        font-weight: 700;
    }

    /* Welcome Container */
    .welcome-container {
        text-align: center;
        padding: 26px 20px 18px 20px;
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(16px);
        border-radius: 18px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin-bottom: 20px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.02);
    }

    .welcome-icon-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #e0e7ff 0%, #cffafe 100%);
        border-radius: 16px;
        font-size: 30px;
        margin-bottom: 10px;
        box-shadow: 0 6px 14px rgba(99, 102, 241, 0.12);
    }

    .welcome-heading {
        font-size: 22px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 4px;
        letter-spacing: -0.4px;
    }

    .welcome-desc {
        font-size: 13.5px;
        color: #64748b;
        max-width: 520px;
        margin: 0 auto 12px auto;
        line-height: 1.5;
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 12px 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #eef2ff 0%, #f0fdfa 100%);
        border: 1px solid rgba(99, 102, 241, 0.18);
        box-shadow: 0 3px 12px rgba(99, 102, 241, 0.04);
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(226, 232, 240, 0.9);
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.03);
    }

    .msg-meta-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid rgba(226, 232, 240, 0.6);
        font-size: 11px;
        color: #94a3b8;
    }

    .source-chunk-card {
        background: #f8fafc;
        border-left: 3px solid #0891b2;
        padding: 9px 12px;
        border-radius: 5px 10px 10px 5px;
        margin-bottom: 6px;
        font-size: 12.5px;
        color: #334155;
        line-height: 1.45;
    }

    .source-chunk-header {
        font-weight: 700;
        color: #0e7490;
        font-size: 10.5px;
        text-transform: uppercase;
        margin-bottom: 3px;
        letter-spacing: 0.5px;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 10px 12px;
        border-radius: 10px;
        color: #92400e;
        font-size: 11.5px;
        line-height: 1.45;
        margin-top: 14px;
    }

    /* Interactive Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 38px;
        border: 1px solid rgba(203, 213, 225, 0.8);
        background: #ffffff;
        color: #1e293b;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #4f46e5;
        color: #4f46e5;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.10);
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 11.5px;
        padding: 24px 10px 12px 10px;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# INITIALIZE SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ============================================================
# CACHED RESOURCE LOADERS
# ============================================================
@st.cache_resource(show_spinner=False)
def get_embeddings():
    """Load HuggingFace all-MiniLM-L6-v2 embeddings."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource(show_spinner=False)
def get_vectorstore(_embeddings):
    """Connect to Pinecone vector store."""
    if not PINECONE_API_KEY:
        st.sidebar.error("⚠️ PINECONE_API_KEY is not set.")
        return None
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=_embeddings,
            text_key="text",
        )
        return vectorstore
    except Exception as e:
        st.sidebar.error(f"⚠️ Pinecone Connection Error: {str(e)}")
        return None

def get_groq_llm(temperature=0.2):
    """Load high-speed Groq LLM client."""
    if not GROQ_API_KEY:
        st.sidebar.error("⚠️ GROQ_API_KEY is not set in .env.")
        return None
    try:
        return ChatGroq(
            model_name=GROQ_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=temperature,
            max_tokens=1024,
            streaming=True,
        )
    except Exception as e:
        st.sidebar.error(f"⚠️ Groq Loading Error: {str(e)}")
        return None

# ============================================================
# SIDEBAR CONTROLS & SESSION METRICS (NO SYSTEM STATUS)
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header-card">
            <div class="sidebar-header-title">
                🩺 MedBot
            </div>
            <div class="sidebar-header-sub">
                Fast Medical AI Assistant (Powered by Groq)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-title">⚙️ Settings</div>', unsafe_allow_html=True)
    top_k = st.slider(
        "Top-K Document Chunks",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of medical context chunks retrieved from Pinecone.",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower values yield more factual and deterministic answers.",
    )

    st.markdown('<div class="sidebar-section-title">💬 Chat Session</div>', unsafe_allow_html=True)
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown(
            f"""
            <div class="meta-card" style="flex-direction: column; align-items: flex-start;">
                <span class="meta-card-label">Turns</span>
                <span class="meta-card-value" style="font-size: 16px;">{len(st.session_state.messages) // 2}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_stat2:
        st.markdown(
            f"""
            <div class="meta-card" style="flex-direction: column; align-items: flex-start;">
                <span class="meta-card-label">Queries</span>
                <span class="meta-card-value" style="font-size: 16px;">{st.session_state.total_queries}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Conversation Actions
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    if len(st.session_state.messages) > 0:
        # Export chat format
        chat_export_text = "# MedBot Conversation History\n\n"
        for m in st.session_state.messages:
            role_name = "User" if m["role"] == "user" else "MedBot"
            ts = m.get("timestamp", "")
            chat_export_text += f"### {role_name} ({ts})\n{m['content']}\n\n"
        
        st.download_button(
            label="📥 Export Chat (Markdown)",
            data=chat_export_text,
            file_name=f"medbot_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.markdown(
        """
        <div class="disclaimer-box">
            ⚠️ <b>Medical Disclaimer:</b><br>
            MedBot is designed for educational and research reference only. It is not a substitute for professional medical diagnosis or treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# LOAD COMPONENTS & GROQ LLM
# ============================================================
with st.spinner("Initializing MedBot and knowledge vector store..."):
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(embeddings)
    llm = get_groq_llm(temperature=temperature)

# ============================================================
# HERO HEADER
# ============================================================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title-row">
            <h1 class="hero-title">🩺 MedBot</h1>
        </div>
        <p class="hero-subtitle">
            Fast, evidence-grounded clinical AI assistant powered by Groq ultra-low latency inference, Pinecone semantic retrieval, and multi-turn conversational history.
        </p>
        <div class="hero-badges">
            <span class="hero-badge">⚡ Ultra-Fast Groq Engine</span>
            <span class="hero-badge">🧠 openai/gpt-oss-120b</span>
            <span class="hero-badge">📚 Gale Medical Encyclopedia</span>
            <span class="hero-badge">💬 Multi-Turn History & Context</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PROMPT TEMPLATE & RAG LOGIC
# ============================================================
rag_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are MedBot, an expert, evidence-grounded clinical medical knowledge assistant.

You have access to:
1. CONVERSATION HISTORY (to understand ongoing context, follow-ups, and user references).
2. MEDICAL KNOWLEDGE CONTEXT (retrieved directly from indexed medical reference documents).

==================================================
CONVERSATION HISTORY:
{chat_history}
==================================================
RETRIEVED MEDICAL CONTEXT:
{context}
==================================================

CURRENT USER QUESTION:
{question}

INSTRUCTIONS FOR MEDBOT:
- Answer accurately and thoroughly using the provided Medical Context, taking into account the Conversation History for context continuity.
- If the user refers back to previous questions (e.g. "what causes it?", "how is it treated?", "what else?"), utilize the Conversation History to understand what condition or topic they are discussing.
- If the retrieved context does not contain sufficient details to answer, clearly state: "Based on the provided medical reference documents, I don't have enough specific information to fully answer this question."
- Organize the response clearly using Markdown headings, bullet points, and bold text for readability.
- Maintain a professional, empathetic, and informative medical assistant tone.

Answer:"""
)

def format_docs(docs):
    """Combine retrieved document chunks."""
    if not docs:
        return "No relevant medical reference documents found."
    return "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)

def format_chat_history(messages, max_turns=4):
    """Format recent turns of chat history for context continuity."""
    recent_msgs = messages[-max_turns * 2:] if len(messages) > max_turns * 2 else messages
    if not recent_msgs:
        return "No prior conversation history."
    history_lines = []
    for msg in recent_msgs:
        role = "User" if msg["role"] == "user" else "MedBot"
        content = msg["content"][:400] + ("..." if len(msg["content"]) > 400 else "")
        history_lines.append(f"{role}: {content}")
    return "\n".join(history_lines)

# ============================================================
# WELCOME SCREEN (WHEN NO MESSAGES)
# ============================================================
if len(st.session_state.messages) == 0:
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-icon-box">🩺</div>
            <div class="welcome-heading">Welcome to MedBot</div>
            <div class="welcome-desc">
                Ask any clinical question or explore topics from our indexed medical knowledge base. MedBot retrieves exact reference excerpts and reasons with multi-turn conversation memory.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### 💡 Suggested Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🫀 What is hypertension and its causes?", use_container_width=True):
            st.session_state.pending_query = "What is hypertension and its causes?"
            st.rerun()
    with col2:
        if st.button("🩸 What causes anemia and how is it detected?", use_container_width=True):
            st.session_state.pending_query = "What causes anemia and how is it detected?"
            st.rerun()
    with col3:
        if st.button("🫁 What are the primary triggers for asthma?", use_container_width=True):
            st.session_state.pending_query = "What are the primary triggers for asthma?"
            st.rerun()

# ============================================================
# RENDER CONVERSATION HISTORY
# ============================================================
for msg in st.session_state.messages:
    is_user = msg["role"] == "user"
    avatar = "👤" if is_user else "🩺"
    
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        
        # Display MedBot metadata & citations accordion
        if not is_user:
            sources = msg.get("sources", [])
            latency = msg.get("latency")
            timestamp = msg.get("timestamp")
            
            meta_str = []
            if timestamp:
                meta_str.append(f"🕒 {timestamp}")
            if latency:
                meta_str.append(f"⚡ {latency}s (Groq)")
            if sources:
                meta_str.append(f"📚 {len(sources)} sources retrieved")
                
            if meta_str:
                st.markdown(
                    f'<div class="msg-meta-row">{" &nbsp;•&nbsp; ".join(meta_str)}</div>',
                    unsafe_allow_html=True,
                )
                
            if sources:
                with st.expander(f"🔍 View Medical Reference Sources ({len(sources)} chunks)"):
                    for idx, src in enumerate(sources, start=1):
                        content = src.get("content", "").strip()
                        st.markdown(
                            f"""
                            <div class="source-chunk-card">
                                <div class="source-chunk-header">Reference Chunk #{idx}</div>
                                {content}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# ============================================================
# CHAT INPUT & EXECUTION
# ============================================================
user_input = st.chat_input("💬 Ask MedBot a medical question or follow up on previous answers...")

if "pending_query" in st.session_state:
    user_input = st.session_state.pop("pending_query")

if user_input:
    curr_time = datetime.datetime.now().strftime("%I:%M %p")
    
    # 1. Record and display User message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": curr_time,
    })
    st.session_state.total_queries += 1
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    # 2. Process MedBot response
    with st.chat_message("assistant", avatar="🩺"):
        if not vectorstore or not llm:
            error_msg = "⚠️ System components (Pinecone vectorstore or Groq LLM) are not properly loaded."
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": curr_time,
            })
        else:
            with st.spinner("🔍 Retrieving medical context and generating fast response..."):
                start_time = time.time()
                try:
                    # Retrieve relevant docs
                    retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": top_k}
                    )
                    retrieved_docs = retriever.invoke(user_input)
                    context_str = format_docs(retrieved_docs)
                    
                    # Format history
                    history_str = format_chat_history(st.session_state.messages[:-1])
                    
                    # Construct prompt
                    formatted_prompt = rag_prompt.format(
                        context=context_str,
                        chat_history=history_str,
                        question=user_input
                    )
                    
                    # Stream response live for instant rendering
                    def stream_generator():
                        for chunk in llm.stream(formatted_prompt):
                            if hasattr(chunk, "content"):
                                yield chunk.content
                            else:
                                yield str(chunk)
                    
                    answer = st.write_stream(stream_generator())
                    
                    elapsed = round(time.time() - start_time, 2)
                    
                    # Format sources for storage
                    sources_list = [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in retrieved_docs
                    ]
                    
                    # Render meta row & sources
                    st.markdown(
                        f'<div class="msg-meta-row">🕒 {curr_time} &nbsp;•&nbsp; ⚡ {elapsed}s (Groq) &nbsp;•&nbsp; 📚 {len(sources_list)} sources retrieved</div>',
                        unsafe_allow_html=True,
                    )
                    
                    if sources_list:
                        with st.expander(f"🔍 View Medical Reference Sources ({len(sources_list)} chunks)"):
                            for idx, src in enumerate(sources_list, start=1):
                                content = src.get("content", "").strip()
                                st.markdown(
                                    f"""
                                    <div class="source-chunk-card">
                                        <div class="source-chunk-header">Reference Chunk #{idx}</div>
                                        {content}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                    
                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": curr_time,
                        "sources": sources_list,
                        "latency": elapsed,
                    })
                    
                except Exception as e:
                    err_msg = f"⚠️ An error occurred while generating the answer: {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "timestamp": curr_time,
                    })

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="app-footer">
        🩺 <b>MedBot</b> • Ultra-fast Medical AI Assistant powered by Groq (openai/gpt-oss-120b) & Pinecone
    </div>
    """,
    unsafe_allow_html=True,
)