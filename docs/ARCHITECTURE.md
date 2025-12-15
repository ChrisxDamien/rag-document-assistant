# Architecture Overview

## Design Philosophy

This RAG system is designed around three principles:

1. **Local First** - Everything runs on your machine by default
2. **Privacy by Design** - Your documents never leave your server
3. **Simplicity** - Easy to understand, easy to extend

---

## Component Deep Dive

### Document Ingestion (`app/rag/ingest.py`)

**Purpose:** Load documents and split them into chunks.

**Chunking Strategy:**
- Uses RecursiveCharacterTextSplitter from LangChain
- Default chunk size: 1000 characters
- Default overlap: 200 characters (20%)
- Separators prioritize: paragraphs > sentences > words

**Why overlap matters:**
Without overlap, a sentence like "The refund policy is described in the next section" would lose context. Overlap ensures semantic continuity.

### Embeddings (`app/rag/embeddings.py`)

**Purpose:** Convert text chunks into vectors for semantic search.

**Default Model:** nomic-embed-text via Ollama
- Open source
- 768-dimensional vectors
- Good balance of quality and speed

**Vector Storage:** ChromaDB
- Persisted to disk (./data/chroma/)
- HNSW index for fast similarity search
- Cosine similarity metric

### Retrieval (`app/rag/retrieval.py`)

**Purpose:** Find the most relevant chunks for a query.

**Process:**
1. Embed the query using the same model
2. Search ChromaDB for nearest neighbors
3. Return top-k results with scores

### Chat (`app/rag/chat.py`)

**Purpose:** Generate answers using retrieved context.

**Process:**
1. Retrieve relevant chunks
2. Format as context
3. Build prompt with system instructions
4. Generate response with LLM
5. Return answer with source citations

---

## Data Flow

```
User Question
     |
     v
[Query Embedding] --> nomic-embed-text (Ollama)
     |
     v
[Vector Search] --> ChromaDB
     |
     v
[Top-K Chunks with Metadata]
     |
     v
[Prompt Assembly: Context + Query]
     |
     v
[LLM Generation] --> llama3.2 (Ollama)
     |
     v
Answer + Sources
```

---

## Configuration

All configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| LLM_PROVIDER | ollama | LLM backend |
| OLLAMA_MODEL | llama3.2 | Chat model |
| OLLAMA_EMBED_MODEL | nomic-embed-text | Embedding model |
| CHROMA_PERSIST_DIR | ./data/chroma | Vector DB location |
| CHUNK_SIZE | 1000 | Characters per chunk |
| CHUNK_OVERLAP | 200 | Overlap between chunks |
| TOP_K | 5 | Documents to retrieve |

---

## Security Notes

1. **No data exfiltration** - All processing is local
2. **Persistent storage** - Vectors stored on disk, survives restarts
3. **No telemetry** - ChromaDB telemetry disabled
4. **Docker isolation** - Container has no network access except Ollama
