"""
RAG Document Assistant - Streamlit Interface
---------------------------------------------
Upload documents and chat with them using local LLMs.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.ingest import ingest_document
from app.rag.embeddings import get_or_create_collection, add_documents
from app.rag.chat import chat, get_chat_history

# Page config
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = []


def main():
    st.title("📄 RAG Document Assistant")
    st.markdown("**Chat with your documents. 100% local. Your data never leaves your server.**")
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Documents")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Supported: PDF, TXT, Markdown"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    for uploaded_file in uploaded_files:
                        if uploaded_file.name not in st.session_state.documents_loaded:
                            # Save temporarily
                            temp_path = Path(f"/tmp/{uploaded_file.name}")
                            temp_path.write_bytes(uploaded_file.getvalue())
                            
                            # Ingest and embed
                            try:
                                chunks = ingest_document(str(temp_path))
                                collection = get_or_create_collection("documents")
                                add_documents(collection, chunks)
                                st.session_state.documents_loaded.append(uploaded_file.name)
                                st.success(f"✅ {uploaded_file.name}")
                            except Exception as e:
                                st.error(f"❌ {uploaded_file.name}: {str(e)}")
                            finally:
                                temp_path.unlink(missing_ok=True)
        
        # Show loaded documents
        if st.session_state.documents_loaded:
            st.markdown("---")
            st.markdown("**Loaded:**")
            for doc in st.session_state.documents_loaded:
                st.markdown(f"• {doc}")
        
        st.markdown("---")
        st.markdown("### Settings")
        st.markdown(f"**Model:** Ollama (llama3.2)")
        st.markdown(f"**Embeddings:** nomic-embed-text")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.markdown(f"- {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if documents are loaded
        if not st.session_state.documents_loaded:
            st.warning("⚠️ Please upload and process documents first.")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chat(
                        query=prompt,
                        collection_name="documents",
                        chat_history=get_chat_history(st.session_state.messages)
                    )
                    
                    st.markdown(response.answer)
                    
                    # Show sources
                    if response.sources:
                        with st.expander("📚 Sources"):
                            for source in response.sources:
                                st.markdown(f"- {source}")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure Ollama is running: `ollama serve`")


if __name__ == "__main__":
    main()
