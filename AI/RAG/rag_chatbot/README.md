# AI-Powered Multi-Document Chatbot (RAG System)
### SevenMentor — End-to-End Capstone Project

A **Retrieval-Augmented Generation (RAG)** system that lets users upload multiple documents (PDF, DOCX, TXT) and converse with their content in natural language. Uses **SentenceTransformers**, **FAISS**, **LangChain** chunking, **Streamlit**, and an **external LLM** (Groq, OpenAI, Gemini, Ollama, etc.).

---

## Architecture

```
[User] ──► Upload Files ──► Validate → Extract → Clean → Chunk
                                │
                         Embed (MiniLM)
                                │
                          [FAISS Index] ◄── saved under tmp/index/
                                │
[User] ──► Ask Question ──► Embed Query ──► Top-K Chunks
                                                    │
                              [External LLM] ◄── Context + Query
                                                    │
                                           Answer + Sources ──► [User]
```

---

## Project Structure

```
rag_chatbot/
├── config.py                   # Central settings (chunking, embedding, LLM)
├── EXTERNAL_LLM_SETUP.md       # How to set Groq / OpenAI / Ollama / …
├── requirements.txt
├── .env.example
│
├── 01_ingestion/
│   └── file_validator.py
├── 02_preprocessing/
│   ├── text_extractor.py
│   ├── cleaner.py
│   └── chunker.py
├── 03_embedding/
│   ├── embedder.py
│   └── vector_store.py
├── 04_retrieval/
│   └── retriever.py
├── 05_generation/
│   ├── prompt_builder.py
│   └── external_llm.py
├── 06_pipeline/
│   └── rag_pipeline.py
├── 07_frontend/
│   └── app.py
└── tests/
```

---

## Quick Start

### 1. Install dependencies
```bash
cd rag_chatbot
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Set LLM_PROVIDER and API key — see EXTERNAL_LLM_SETUP.md
```

### 3. Launch the Streamlit app
```bash
streamlit run 07_frontend/app.py
```

---

## Syllabus Mapping

| Module | Syllabus Section | Key Concepts |
|--------|-----------------|--------------|
| `01_ingestion/` | Section 4: Data Collection | File validation, I/O |
| `02_preprocessing/` | Section 7: NLP Preprocessing | Tokenization, chunking, text cleaning |
| `03_embedding/` + `04_retrieval/` | Section 7: NLP + Embeddings | SentenceTransformers, FAISS, similarity search |
| `05_generation/` | Section 6 & 7: Deep Learning + Transformers | LLMs, RAG, prompt engineering |
| `07_frontend/` | Section 8: Deployment | Streamlit UI |

---

*Built for SevenMentor AI/ML Capstone — February 2025*
