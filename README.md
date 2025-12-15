# 📄 RAG Document Assistant

**Chat with your documents. 100% local. Your data never leaves your server.**

> Most RAG tutorials send your private documents to OpenAI. This runs entirely on your machine using Ollama. Self-hosted. Air-gapped if you want.

---

## What It Does

Upload documents. Ask questions. Get answers with sources.

```
You: "What's our refund policy?"
Bot: "Based on employee_handbook.pdf (page 12): Refunds are processed 
     within 5-7 business days for all purchases made within 30 days..."
```

---

## Use Cases

| Scenario | Example |
|----------|---------|
| **Internal Wiki** | "How do I submit a PTO request?" |
| **Customer Support** | "What's covered under warranty?" |
| **Sales Enablement** | "Find case studies about healthcare clients" |
| **Legal/Compliance** | "Search all contracts for indemnification clauses" |
| **Onboarding** | "What's the dress code policy?" |

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed
- Docker (optional, for containerized deployment)

### 1. Clone and Install

```bash
git clone https://github.com/ChrisxDamien/rag-document-assistant.git
cd rag-document-assistant
pip install -r requirements.txt
```

### 2. Pull Required Models

```bash
# LLM for chat
ollama pull llama3.2

# Embeddings for vector search
ollama pull nomic-embed-text
```

### 3. Run the App

```bash
streamlit run app/main.py
```

Open http://localhost:8501 in your browser.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      STREAMLIT UI                           │
│  - Upload documents                                         │
│  - Chat interface                                           │
│  - Source citations                                         │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                             │
└─────────────────────────────┬───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   INGEST     │      │   RETRIEVE   │      │   GENERATE   │
│              │      │              │      │              │
│ • Load docs  │      │ • Embed query│      │ • Build prompt│
│ • Chunk text │      │ • Search DB  │      │ • Call LLM   │
│ • Embed      │      │ • Rerank     │      │ • Stream     │
│ • Store      │      │ • Return top │      │   response   │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  CHROMADB    │      │   OLLAMA     │      │   OLLAMA     │
│  (Vectors)   │      │ (Embeddings) │      │   (LLM)      │
└──────────────┘      └──────────────┘      └──────────────┘
```

---

## Project Structure

```
rag-document-assistant/
├── app/
│   ├── main.py              # Streamlit interface
│   └── rag/
│       ├── __init__.py
│       ├── ingest.py        # Document loading + chunking
│       ├── embeddings.py    # Vector embedding with Ollama
│       ├── retrieval.py     # Semantic search + reranking
│       └── chat.py          # Conversational RAG chain
├── data/
│   └── sample_docs/         # Example documents for demo
├── docs/
│   └── ARCHITECTURE.md      # Technical deep-dive
├── docker-compose.yml       # One-command deployment
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Supported File Types

| Type | Extensions |
|------|------------|
| Documents | `.pdf`, `.docx`, `.txt`, `.md` |
| Spreadsheets | `.csv` (coming soon) |
| Web | `.html` (coming soon) |

---

## Configuration

### Using Ollama (Default - Free)

No configuration needed. Ensure Ollama is running:

```bash
ollama serve
```

### Using OpenAI (Optional)

Create a `.env` file:

```bash
cp .env.example .env
# Add your OPENAI_API_KEY
```

---

## Docker Deployment

### Run with Docker Compose

```bash
docker-compose up -d
```

This starts:
- Streamlit app on port 8501
- ChromaDB for vector storage
- Ollama for LLM (if not running externally)

### For Production

```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## How It Works

### 1. Document Ingestion

```python
from app.rag.ingest import ingest_document

# Load and chunk a document
chunks = ingest_document("company_handbook.pdf")
# Returns: List of text chunks with metadata
```

### 2. Embedding + Storage

```python
from app.rag.embeddings import embed_and_store

# Embed chunks and store in vector DB
embed_and_store(chunks, collection="company_docs")
```

### 3. Retrieval

```python
from app.rag.retrieval import retrieve

# Find relevant chunks
results = retrieve("What's the vacation policy?", collection="company_docs", top_k=5)
```

### 4. Generation

```python
from app.rag.chat import chat

# Generate answer with sources
response = chat("What's the vacation policy?", collection="company_docs")
print(response.answer)
print(response.sources)
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **100% Local** | No data leaves your machine |
| **Source Citations** | Every answer shows where it came from |
| **Conversational** | Remembers context within a session |
| **Chunk Overlap** | Smart chunking preserves context |
| **Reranking** | Better results than naive similarity |

---

## Roadmap

- [x] PDF ingestion
- [x] Basic RAG pipeline
- [x] Streamlit UI
- [x] Source citations
- [ ] DOCX support
- [ ] CSV/Excel support
- [ ] Multi-collection search
- [ ] Hybrid search (keyword + semantic)
- [ ] Conversation memory persistence
- [ ] API endpoint (FastAPI)

---

## Why Local RAG?

| Concern | Cloud RAG | Local RAG |
|---------|-----------|-----------|
| **Privacy** | Docs sent to third party | Stays on your server |
| **Cost** | Per-token pricing adds up | Free after hardware |
| **Latency** | Network round-trip | Local = fast |
| **Compliance** | May violate data policies | Full control |
| **Availability** | Depends on provider uptime | Runs offline |

---

## Contributing

PRs welcome. Please:

1. Keep it simple - this is meant to be understandable
2. Test with Ollama (free tier must work)
3. Update documentation

---

## License

MIT - Use it however you want.

---

## About

Built by [Chris Damien](https://linkedin.com/in/chris-damien) as part of my work helping businesses leverage AI.

**More resources:**
- [LinkedIn](https://linkedin.com/in/chris-damien) - Weekly AI automation content
- [Other Projects](https://github.com/ChrisxDamien) - More tools

---

*If this saved you time, star the repo ⭐*
