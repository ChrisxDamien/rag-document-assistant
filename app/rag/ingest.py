"""
Document Ingestion
------------------
Load documents and split them into chunks for embedding.
"""

import os
from pathlib import Path
from typing import List
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)


@dataclass
class DocumentChunk:
    """A chunk of text with metadata."""
    content: str
    metadata: dict
    

def get_loader(file_path: str):
    """Get the appropriate loader based on file extension."""
    ext = Path(file_path).suffix.lower()
    
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }
    
    loader_class = loaders.get(ext)
    if not loader_class:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return loader_class(file_path)


def ingest_document(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[DocumentChunk]:
    """
    Load a document and split it into chunks.
    
    Args:
        file_path: Path to the document
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of DocumentChunk objects
    """
    # Load document
    loader = get_loader(file_path)
    documents = loader.load()
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    splits = splitter.split_documents(documents)
    
    # Convert to our format
    chunks = []
    file_name = Path(file_path).name
    
    for i, split in enumerate(splits):
        chunks.append(DocumentChunk(
            content=split.page_content,
            metadata={
                "source": file_name,
                "chunk_index": i,
                "page": split.metadata.get("page", 0),
                **split.metadata
            }
        ))
    
    return chunks


def ingest_directory(
    dir_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[DocumentChunk]:
    """
    Load all supported documents from a directory.
    
    Args:
        dir_path: Path to the directory
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of all DocumentChunks from all files
    """
    supported_extensions = {".pdf", ".txt", ".md"}
    all_chunks = []
    
    for file_path in Path(dir_path).rglob("*"):
        if file_path.suffix.lower() in supported_extensions:
            try:
                chunks = ingest_document(
                    str(file_path),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Warning: Failed to process {file_path}: {e}")
    
    return all_chunks
