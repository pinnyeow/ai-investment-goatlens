"""
RAG (Retrieval-Augmented Generation) module for GOATlens.

This demonstrates Step 8: RAG in AI Product Sense.

Features:
- Vector store for earnings transcripts
- Semantic search for relevant context
- Enriches agent analysis with CEO commentary and strategic insights

Uses FAISS for vector storage (can be upgraded to ChromaDB or Pinecone later).
"""

from .retriever import RAGRetriever

__all__ = ["RAGRetriever"]
