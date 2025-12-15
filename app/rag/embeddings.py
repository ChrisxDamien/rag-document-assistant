"""
Embeddings and Vector Storage
-----------------------------
Embed document chunks and store in ChromaDB.
"""

import os
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

from .ingest import DocumentChunk


# Default settings
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def get_embeddings():
    """Get the embedding model."""
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)


def get_chroma_client():
    """Get or create ChromaDB client."""
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )


def get_or_create_collection(name: str = "documents"):
    """Get or create a ChromaDB collection."""
    client = get_chroma_client()
    
    # Get embedding function
    embeddings = get_embeddings()
    
    # Create custom embedding function for Chroma
    class OllamaEmbeddingFunction:
        def __init__(self, embeddings):
            self.embeddings = embeddings
            
        def __call__(self, input: List[str]) -> List[List[float]]:
            return self.embeddings.embed_documents(input)
    
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=OllamaEmbeddingFunction(embeddings),
        metadata={"hnsw:space": "cosine"}
    )
    
    return collection


def add_documents(
    collection,
    chunks: List[DocumentChunk],
) -> None:
    """
    Add document chunks to a collection.
    
    Args:
        collection: ChromaDB collection
        chunks: List of DocumentChunk objects
    """
    if not chunks:
        return
    
    # Prepare data for Chroma
    ids = [f"{chunk.metadata['source']}_{chunk.metadata['chunk_index']}" for chunk in chunks]
    documents = [chunk.content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )


def delete_collection(name: str = "documents") -> None:
    """Delete a collection."""
    client = get_chroma_client()
    try:
        client.delete_collection(name)
    except ValueError:
        pass  # Collection doesn't exist
