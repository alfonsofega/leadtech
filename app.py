"""
Streamlit App - AI CV Screener
--------------------------------
This app implements a chat interface for querying CVs using a RAG pipeline.
- Left sidebar: shows the number of available CVs and a button to clear the chat.
- Main area: chat interface powered by LlamaIndex pipeline (CVScreenerRAG).
"""

import os
from datetime import datetime
import streamlit as st
from utils.rag import CVScreenerRAG

# ==================== CONFIG ====================
st.set_page_config(
    page_title="AI CV Screener",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================== HEADER ====================
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0;">🔍 AI-Powered CV Screener</h1>
    <p style="text-align:center; color:#5f6368; font-size:1.05rem; margin-top:6px;">
        Ask questions about candidate profiles using natural language
    </p>
    """,
    unsafe_allow_html=True,
)


# ==================== INITIALIZATION ====================
@st.cache_resource
def init_rag():
    """Initialize the RAG pipeline (cached to run only once)."""
    rag = CVScreenerRAG()
    rag.load_or_create_index()
    return rag


# Create pipeline instance once
if "rag" not in st.session_state:
    with st.spinner("🔨 Loading AI system..."):
        st.session_state.rag = init_rag()

# Initialize chat history
if "messages" not in st.session_state:
    # Each message is a dict: {"role": "user"|"assistant", "content": str, "sources": list|None, "ts": datetime}
    st.session_state.messages = []

CVS_DIR = "cvs"


def count_pdfs():
    """Return the number of PDF CVs in the cvs/ directory."""
    return len([f for f in os.listdir(CVS_DIR) if f.lower().endswith(".pdf")]) if os.path.exists(CVS_DIR) else 0


# ==================== SIDEBAR ====================
with st.sidebar:
    num_cvs = count_pdfs()

    # Display number of CVs
    st.markdown(
        f"""
        <div style="
            text-align:center; 
            margin: 6px 0 10px; 
            padding: 12px 10px; 
            border-radius: 10px; 
            background: #f6f7f9; 
            border: 1px solid #e6e8eb;">
            <div style="font-size: 1.6rem; font-weight: 700; line-height: 1;">{num_cvs}</div>
            <div style="color:#666; margin-top:2px;">CVs loaded</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    # Clear chat button
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat cleared", icon="🧼")
        st.rerun()

# ==================== CHAT HISTORY ====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display retrieved sources if available
        sources = msg.get("sources") or []
        if sources:
            with st.expander("📄 Sources"):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f"""
                        <div style="padding:8px 10px; border:1px solid #e6e8eb; border-radius:8px; margin-bottom:8px;">
                            <strong>{i}. {src.get('file', '')}</strong><br>
                            <span style="color:#666;">Relevance: {src.get('score', '')}</span><br>
                            <span style="color:#8a8f98; font-size:0.85rem;">
                                {(src.get('text', '') or '')[:160] + ('…' if len(src.get('text', '')) > 160 else '')}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ==================== CHAT INPUT ====================
user_input = st.chat_input("Type your question about the CVs…")
if user_input:
    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "sources": None, "ts": datetime.now()}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process query with RAG pipeline
    with st.chat_message("assistant"):
        with st.status("Searching through CVs…", expanded=False) as status:
            try:
                answer, sources = st.session_state.rag.query(user_input)
                status.update(label="Found relevant results", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Error", state="error", expanded=True)
                st.error(f"❌ {e}")
                answer, sources = "I couldn't complete that query due to an internal error.", []

        # Show assistant response
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources"):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f"""
                        <div style="padding:8px 10px; border:1px solid #e6e8eb; border-radius:8px; margin-bottom:8px;">
                            <strong>{i}. {src.get('file', '')}</strong><br>
                            <span style="color:#666;">Relevance: {src.get('score', '')}</span><br>
                            <span style="color:#8a8f98; font-size:0.85rem;">
                                {(src.get('text', '') or '')[:160] + ('…' if len(src.get('text', '')) > 160 else '')}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Save assistant message and refresh
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "ts": datetime.now()}
    )
    st.rerun()
