"""
Retrieval
---------
Semantic search over embedded documents.
"""

import os
from typing import List, Optional
from dataclasses import dataclass

from .embeddings import get_or_create_collection, get_embeddings


TOP_K = int(os.getenv("TOP_K", "5"))


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    content: str
    source: str
    page: int
    score: float
    metadata: dict


def retrieve(
    query: str,
    collection_name: str = "documents",
    top_k: int = None,
) -> List[RetrievalResult]:
    """
    Retrieve relevant document chunks for a query.
    
    Args:
        query: The search query
        collection_name: Name of the collection to search
        top_k: Number of results to return
        
    Returns:
        List of RetrievalResult objects
    """
    if top_k is None:
        top_k = TOP_K
    
    collection = get_or_create_collection(collection_name)
    
    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Convert to our format
    retrieval_results = []
    
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0
            
            retrieval_results.append(RetrievalResult(
                content=doc,
                source=metadata.get("source", "Unknown"),
                page=metadata.get("page", 0),
                score=1 - distance,  # Convert distance to similarity
                metadata=metadata
            ))
    
    return retrieval_results


def format_context(results: List[RetrievalResult]) -> str:
    """Format retrieval results as context for the LLM."""
    if not results:
        return "No relevant documents found."
    
    context_parts = []
    for i, result in enumerate(results, 1):
        source_info = f"{result.source}"
        if result.page:
            source_info += f" (page {result.page})"
        
        context_parts.append(
            f"[Source {i}: {source_info}]\n{result.content}"
        )
    
    return "\n\n---\n\n".join(context_parts)
