"""
07_frontend/app.py
────────────────────
Streamlit web application for the AI-Powered Multi-Document Chatbot.

Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │  Sidebar                  │  Main Area                      │
    │  ─────────                │  ──────────                     │
    │  Session ID               │  Chat history (bubbles)         │
    │  File uploader            │  Source expander (collapsible)  │
    │  Process button           │  Chat input box                 │
    │  Status / chunk count     │                                 │
    └─────────────────────────────────────────────────────────────┘

Run:
    streamlit run 07_frontend/app.py
"""

import sys
import uuid
import logging
from pathlib import Path
from typing import List, Dict

import streamlit as st

# Ensure project root is on the path and load .env FIRST
_project_root = Path(__file__).resolve().parents[1]
_env_file = _project_root / ".env"
if _env_file.exists():
    import dotenv
    dotenv.load_dotenv(_env_file)
sys.path.insert(0, str(_project_root))
import pathsetup  # noqa: F401 — registers pipeline, ingestion, embedding, etc.

from pipeline.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocBot — AI Multi-Document Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Chat bubbles */
.user-bubble {
    background: #1a73e8;
    color: white;
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    margin: 6px 0 6px 15%;
    text-align: right;
    font-size: 0.95rem;
}
.assistant-bubble {
    background: #f1f3f4;
    color: #202124;
    padding: 10px 16px;
    border-radius: 18px 18px 18px 4px;
    margin: 6px 15% 6px 0;
    font-size: 0.95rem;
}
.source-tag {
    font-size: 0.75rem;
    color: #5f6368;
    margin-top: 2px;
}
/* Sidebar header */
.sidebar-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a73e8;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ─────────────────────────────────────────────

def _init_session_state():
    """Populate Streamlit session_state with default values on first load."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []          # list of {"role", "content", "sources"}
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline()
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []


_init_session_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-title">📚 DocBot — Document Upload</div>', unsafe_allow_html=True)
    st.caption("SevenMentor | RAG Chatbot Project")
    st.divider()

    # Session info
    with st.expander("🔑 Session Info", expanded=False):
        st.code(f"Session ID: {st.session_state.session_id}", language=None)
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())[:8]
            st.session_state.chat_history = []
            st.session_state.index_ready = False
            st.session_state.total_chunks = 0
            st.session_state.processed_files = []
            st.rerun()

    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Max 20 MB per file. Upload one or more documents.",
    )

    process_btn = st.button(
        "⚙️ Process Documents",
        use_container_width=True,
        disabled=(not uploaded_files),
        type="primary",
    )

    if process_btn and uploaded_files:
        files_data = [(f.name, f.read()) for f in uploaded_files]
        with st.spinner("Extracting → Chunking → Embedding ..."):
            try:
                result = st.session_state.pipeline.ingest(
                    files=files_data,
                    session_id=st.session_state.session_id,
                )
                st.session_state.index_ready = result["total_chunks"] > 0
                st.session_state.total_chunks = result["total_chunks"]
                st.session_state.processed_files = [f[0] for f in files_data]

                if result["skipped_files"]:
                    for fname, reason in result["skipped_files"]:
                        st.warning(f"Skipped **{fname}**: {reason}")

                if st.session_state.index_ready:
                    st.success(
                        f"✅ {result['files_processed']} file(s) processed\n\n"
                        f"📊 {result['total_chunks']} chunks indexed"
                    )
                else:
                    st.error("No text could be extracted from the uploaded files.")

            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    # Status panel
    st.divider()
    st.subheader("2. Status")
    if st.session_state.index_ready:
        st.success(f"✅ Index ready — {st.session_state.total_chunks} chunks")
        if st.session_state.processed_files:
            st.caption("Indexed files:")
            for fname in st.session_state.processed_files:
                st.caption(f"  • {fname}")

        # Vector visualization (2D PCA)
        with st.expander("📊 View / visualize vectors", expanded=False):
            try:
                from embedding.vector_store import VectorStore
                from sklearn.decomposition import PCA
                import plotly.graph_objects as go

                vs = VectorStore(session_id=st.session_state.session_id)
                if vs.load():
                    vectors, metadata = vs.get_vectors_and_metadata()
                    if len(vectors) > 0:
                        xy = PCA(n_components=2, random_state=42).fit_transform(vectors)
                        sources = [m.get("source", "?") for m in metadata]
                        text_preview = [
                            (m.get("text", "")[:100] + "…" if len(m.get("text", "")) > 100 else m.get("text", ""))
                            for m in metadata
                        ]
                        hover = [
                            f"<b>{sources[i]}</b> (p.{metadata[i].get('page', '?')})<br>{text_preview[i]}"
                            for i in range(len(metadata))
                        ]
                        unique_sources = list(dict.fromkeys(sources))
                        fig = go.Figure()
                        for src in unique_sources:
                            inds = [i for i, s in enumerate(sources) if s == src]
                            fig.add_trace(
                                go.Scatter(
                                    x=xy[inds, 0], y=xy[inds, 1], mode="markers",
                                    name=src, text=[hover[i] for i in inds],
                                    hovertemplate="%{text}<extra></extra>",
                                    marker=dict(size=6, opacity=0.8),
                                )
                            )
                        fig.update_layout(
                            title="Chunk embeddings (PCA 2D)",
                            xaxis_title="PC1", yaxis_title="PC2", height=380,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption("No vectors to show.")
                else:
                    st.caption("Could not load index for this session.")
            except Exception as e:
                st.caption(f"Visualization unavailable: {e}")
    else:
        st.info("No index yet. Upload and process documents above.")

    st.divider()
    st.subheader("3. Settings")
    top_k = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=1, max_value=10, value=5,
        help="Higher = more context for the LLM, but slower.",
    )

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("📚 AI Multi-Document Chatbot")
st.caption(
    "Upload your documents in the sidebar, then ask questions below. "
    "The AI answers using only your uploaded documents."
)

# Render chat history
chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        st.markdown(
            "<div style='text-align:center; color:#9aa0a6; margin-top:60px;'>"
            "👋 Upload documents and ask your first question to get started."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        for turn in st.session_state.chat_history:
            if turn["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">🧑 {turn["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="assistant-bubble">🤖 {turn["content"]}</div>',
                    unsafe_allow_html=True,
                )
                # Collapsible sources
                sources = turn.get("sources", [])
                if sources:
                    with st.expander(
                        f"📎 View {len(sources)} source excerpt(s)", expanded=False
                    ):
                        for i, chunk in enumerate(sources, 1):
                            score_pct = int(chunk.get("score", 0) * 100)
                            st.markdown(
                                f"**[{i}] {chunk['source']}** — page {chunk['page']} "
                                f"<span class='source-tag'>(similarity: {score_pct}%)</span>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"> {chunk['text'][:400]}{'...' if len(chunk['text']) > 400 else ''}",
                            )
                            if i < len(sources):
                                st.divider()

# Chat input
st.divider()
with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([10, 1])
    with cols[0]:
        user_input = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g. What is the main topic of the uploaded report?",
            label_visibility="collapsed",
        )
    with cols[1]:
        submitted = st.form_submit_button("Send", use_container_width=True, type="primary")

if submitted and user_input.strip():
    if not st.session_state.index_ready:
        st.warning("Please upload and process documents first.")
    else:
        # Build conversation history for multi-turn context (last 6 turns)
        history = [
            {"role": t["role"], "content": t["content"]}
            for t in st.session_state.chat_history[-6:]
        ]

        # Append user turn to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input.strip(),
            "sources": [],
        })

        with st.spinner("Thinking ..."):
            try:
                result = st.session_state.pipeline.query(
                    question=user_input.strip(),
                    session_id=st.session_state.session_id,
                    conversation_history=history,
                    top_k=top_k,
                )
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as exc:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {exc}",
                    "sources": [],
                })

        st.rerun()
