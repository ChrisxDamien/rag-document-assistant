"""
Chat Interface
--------------
Conversational RAG with source citations.
"""

import os
from typing import List, Optional
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .retrieval import retrieve, format_context, RetrievalResult


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


@dataclass
class ChatResponse:
    """Response from the chat function."""
    answer: str
    sources: List[str]
    context_used: List[RetrievalResult]


# System prompt for RAG
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

IMPORTANT RULES:
1. Only answer based on the context provided below
2. If the context doesn't contain the answer, say "I don't have enough information to answer that based on the documents."
3. Always cite your sources by mentioning which document the information came from
4. Be concise but thorough
5. If asked about something not in the documents, acknowledge that

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}
"""


def get_llm():
    """Get the LLM."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.3,  # Lower temperature for more factual responses
    )


def get_chat_history(messages: List[dict]) -> str:
    """Format chat history for the prompt."""
    if not messages:
        return "No previous conversation."
    
    history_parts = []
    for msg in messages[-6:]:  # Last 3 exchanges
        role = "Human" if msg["role"] == "user" else "Assistant"
        history_parts.append(f"{role}: {msg['content']}")
    
    return "\n".join(history_parts)


def chat(
    query: str,
    collection_name: str = "documents",
    chat_history: str = "",
    top_k: int = 5,
) -> ChatResponse:
    """
    Chat with documents using RAG.
    
    Args:
        query: The user's question
        collection_name: Name of the document collection
        chat_history: Formatted chat history
        top_k: Number of documents to retrieve
        
    Returns:
        ChatResponse with answer and sources
    """
    # Retrieve relevant documents
    results = retrieve(query, collection_name, top_k)
    
    if not results:
        return ChatResponse(
            answer="I don't have any documents to search. Please upload some documents first.",
            sources=[],
            context_used=[]
        )
    
    # Format context
    context = format_context(results)
    
    # Build prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    # Create chain
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    
    # Generate response
    answer = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": query
    })
    
    # Extract unique sources
    sources = list(set(
        f"{r.source}" + (f" (page {r.page})" if r.page else "")
        for r in results
    ))
    
    return ChatResponse(
        answer=answer,
        sources=sources,
        context_used=results
    )
