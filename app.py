import os
import streamlit as st
import torch
from dotenv import load_dotenv

# ============================================================
# LANGCHAIN
# ============================================================

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ============================================================
# VECTOR STORE + EMBEDDINGS
# ============================================================

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone

# ============================================================
# HUGGING FACE LLM
# ============================================================

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from langchain_community.llms import HuggingFacePipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MediRAG AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99, 102, 241, 0.12),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(6, 182, 212, 0.12),
                transparent 28%
            ),
            linear-gradient(
                135deg,
                #f8fafc,
                #eef2ff
            );
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .hero {
        padding: 30px;
        border-radius: 24px;
        margin-bottom: 25px;

        background:
            linear-gradient(
                135deg,
                #312e81 0%,
                #4f46e5 45%,
                #0891b2 100%
            );

        color: white;

        box-shadow:
            0 15px 40px rgba(79, 70, 229, 0.25);
    }

    .hero-title {
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 16px;
        opacity: 0.92;
        margin: 0;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #eef2ff 0%,
                #f8fafc 50%,
                #ecfeff 100%
            );
    }

    .sidebar-brand {
        padding: 20px;
        border-radius: 20px;

        background:
            linear-gradient(
                135deg,
                #4f46e5,
                #0891b2
            );

        color: white;

        margin-bottom: 22px;

        box-shadow:
            0 10px 25px rgba(79, 70, 229, 0.2);
    }

    .sidebar-brand-title {
        font-size: 22px;
        font-weight: 800;
    }

    .sidebar-brand-subtitle {
        font-size: 13px;
        margin-top: 5px;
        opacity: 0.9;
    }


    /* ========================================================
       INFO CARDS
       ======================================================== */

    .info-card {
        background: rgba(255,255,255,0.85);

        border-radius: 18px;

        padding: 17px;

        margin-bottom: 12px;

        border:
            1px solid rgba(99,102,241,0.12);

        box-shadow:
            0 5px 20px rgba(15,23,42,0.06);
    }

    .info-label {
        font-size: 12px;
        color: #64748b;
        margin-bottom: 5px;
    }

    .info-value {
        font-size: 16px;
        font-weight: 700;
        color: #1e293b;
    }


    /* ========================================================
       CHAT
       ======================================================== */

    [data-testid="stChatMessage"] {
        border-radius: 18px;

        padding:
            10px
            14px;

        margin-bottom: 12px;
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-user"]
    ) {

        background:
            linear-gradient(
                135deg,
                rgba(79,70,229,0.08),
                rgba(6,182,212,0.08)
            );

        border:
            1px solid rgba(79,70,229,0.08);
    }

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-assistant"]
    ) {

        background:
            rgba(255,255,255,0.90);

        border:
            1px solid rgba(99,102,241,0.10);

        box-shadow:
            0 4px 15px rgba(15,23,42,0.04);
    }


    /* ========================================================
       CHAT INPUT
       ======================================================== */

    [data-testid="stChatInput"] {

        border-radius: 20px;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {

        border-radius: 13px;

        font-weight: 600;

        border:
            1px solid rgba(79,70,229,0.15);

        transition:
            all 0.2s ease;
    }

    .stButton > button:hover {

        transform:
            translateY(-2px);

        box-shadow:
            0 7px 18px rgba(79,70,229,0.15);
    }


    /* ========================================================
       WELCOME SECTION
       ======================================================== */

    .welcome {

        text-align: center;

        padding:
            35px
            20px;

        margin:
            10px
            0
            20px
            0;
    }

    .welcome-icon {

        font-size: 60px;

        margin-bottom: 8px;
    }

    .welcome-title {

        font-size: 28px;

        font-weight: 800;

        color: #1e293b;

        margin-bottom: 7px;
    }

    .welcome-text {

        color: #64748b;

        font-size: 15px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {

        text-align: center;

        color: #64748b;

        font-size: 12px;

        padding: 25px 10px;
    }


    /* ========================================================
       REMOVE STREAMLIT DEFAULT FOOTER
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🩺 MediRAG AI
        </div>

        <p class="hero-subtitle">
            Intelligent medical question answering powered by
            Retrieval-Augmented Generation
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <div class="sidebar-brand-title">
                🩺 MediRAG AI
            </div>

            <div class="sidebar-brand-subtitle">
                Medical Knowledge Assistant
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ⚙️ Controls")

    top_k = st.slider(
        "📚 Documents to retrieve",
        min_value=1,
        max_value=5,
        value=3,
        help=(
            "Number of relevant document chunks "
            "retrieved from Pinecone."
        )
    )

    st.markdown("### 🧠 AI Model")

    st.markdown(
        """
        <div class="info-card">

            <div class="info-label">
                Language Model
            </div>

            <div class="info-value">
                🤗 Qwen 2.5 3B Instruct
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

            <div class="info-label">
                Retrieval Engine
            </div>

            <div class="info-value">
                📌 Pinecone
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

            <div class="info-label">
                Knowledge Source
            </div>

            <div class="info-value">
                📖 Gale Encyclopedia of Medicine
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 💬 Conversation")

    if st.button(
        "🧹 Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.markdown(
        """
        <div class="info-card">

            <div class="info-label">
                ⚠️ Important
            </div>

            <div style="
                font-size:12px;
                color:#475569;
                line-height:1.5;
            ">

                This assistant is designed for
                educational and research purposes.

                <br><br>

                It does not replace professional
                medical advice.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource(
    show_spinner="🔄 Loading embeddings..."
)
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )


# ============================================================
# LOAD PINECONE
# ============================================================

@st.cache_resource(
    show_spinner="🔗 Connecting to Pinecone..."
)
def load_vectorstore(_embeddings):

    if not PINECONE_API_KEY:

        st.error(
            "PINECONE_API_KEY is missing."
        )

        st.stop()

    if not PINECONE_INDEX_NAME:

        st.error(
            "PINECONE_INDEX_NAME is missing."
        )

        st.stop()

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    index = pc.Index(
        PINECONE_INDEX_NAME
    )

    return PineconeVectorStore(
        index=index,
        embedding=_embeddings,
        text_key="text"
    )


# ============================================================
# LOAD HUGGING FACE LLM
# ============================================================

@st.cache_resource(
    show_spinner="🧠 Loading Qwen 2.5 3B..."
)
def load_llm():

    model_id = (
        "Qwen/Qwen2.5-3B-Instruct"
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if torch.cuda.is_available():

        device = "cuda"

        dtype = torch.float16

    else:

        device = "cpu"

        dtype = torch.float32

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype
    )

    model.to(device)

    model.eval()

    # --------------------------------------------------------
    # Generation pipeline
    # --------------------------------------------------------

    hf_pipeline = pipeline(
        task="text-generation",

        model=model,

        tokenizer=tokenizer,

        max_new_tokens=256,

        temperature=0.2,

        do_sample=True,

        repetition_penalty=1.1,

        return_full_text=False,

        device=0 if device == "cuda" else -1
    )

    # --------------------------------------------------------
    # LangChain wrapper
    # --------------------------------------------------------

    return HuggingFacePipeline(
        pipeline=hf_pipeline
    )


# ============================================================
# LOAD COMPONENTS
# ============================================================

embeddings = load_embeddings()

vectorstore = load_vectorstore(
    embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": top_k
    }
)

llm = load_llm()


# ============================================================
# RAG PROMPT
# ============================================================

prompt = PromptTemplate(
    input_variables=[
        "context",
        "question"
    ],

    template="""
You are a careful medical information assistant.

Answer the user's question ONLY using the
information provided in the context.

Important rules:

1. Do not use information outside the context.

2. Do not invent or assume medical facts.

3. If the answer is not present in the context,
   say exactly:

"I don't know based on the provided medical documents."

4. Give a clear and concise answer.

5. Use simple language whenever possible.

6. Use bullet points when they make the
   answer easier to understand.

7. Do not provide a diagnosis unless it is
   explicitly supported by the context.

8. Do not fabricate references or sources.

Context:
-------------------------
{context}
-------------------------

Question:
{question}

Answer:
"""
)


# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# QA CHAIN
# ============================================================

qa_chain = (

    {
        "context":
            retriever | format_docs,

        "question":
            RunnablePassthrough()
    }

    | prompt

    | llm

    | StrOutputParser()
)


# ============================================================
# CHAT MEMORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# WELCOME SCREEN
# ============================================================

if len(st.session_state.messages) == 0:

    st.markdown(
        """
        <div class="welcome">

            <div class="welcome-icon">
                🩺
            </div>

            <div class="welcome-title">
                How can I help you?
            </div>

            <div class="welcome-text">
                Ask a medical question and I'll search
                the medical knowledge base for relevant
                information.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 💡 Try asking")

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.button(
            "🫀 What is hypertension?",
            use_container_width=True
        ):

            st.session_state.pending_query = (
                "What is hypertension?"
            )

            st.rerun()

    with col2:

        if st.button(
            "🩸 What causes anemia?",
            use_container_width=True
        ):

            st.session_state.pending_query = (
                "What causes anemia?"
            )

            st.rerun()

    with col3:

        if st.button(
            "🫁 What is asthma?",
            use_container_width=True
        ):

            st.session_state.pending_query = (
                "What is asthma?"
            )

            st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        avatar = "👤"

    else:

        avatar = "🩺"

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "💬 Ask a medical question..."
)


# ============================================================
# QUICK QUESTION HANDLER
# ============================================================

if "pending_query" in st.session_state:

    user_query = st.session_state.pop(
        "pending_query"
    )


# ============================================================
# PROCESS USER QUERY
# ============================================================

if user_query:

    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(
            user_query
        )

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🩺"
    ):

        with st.spinner(
            "🔎 Searching medical knowledge base..."
        ):

            try:

                answer = qa_chain.invoke(
                    user_query
                )

            except Exception as e:

                answer = (
                    "⚠️ Sorry, I encountered an "
                    "error while generating the answer."
                )

                st.error(
                    str(e)
                )

        st.markdown(
            answer
        )

    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🩺 <b>MediRAG AI</b>
        &nbsp; • &nbsp;
        Hugging Face + Pinecone + Streamlit

        <br><br>

        ⚠️ This application is for educational
        and research purposes only.

        <br>

        It is not a substitute for professional
        medical advice.

    </div>
    """,
    unsafe_allow_html=True
)