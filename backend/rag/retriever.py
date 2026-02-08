"""
RAG Retriever for GOATlens.

Retrieves relevant earnings transcripts and analyst reports to enrich agent analysis.
This demonstrates Step 8: RAG in AI Product Sense.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Try to import vector store libraries (optional dependencies)
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.schema import Document
    HAS_VECTOR_STORE = True
except ImportError:
    HAS_VECTOR_STORE = False
    print("[RAG] Vector store libraries not installed. RAG will be disabled.")
    print("[RAG] Install with: pip install faiss-cpu langchain-openai")


class RAGRetriever:
    """
    Retrieval-Augmented Generation retriever for earnings transcripts.
    
    Stores earnings call transcripts in a vector store and retrieves
    relevant context based on semantic similarity.
    """
    
    def __init__(self, enable_rag: bool = True):
        """
        Initialize RAG retriever.
        
        Args:
            enable_rag: Whether to enable RAG (requires vector store libraries)
        """
        self.enable_rag = enable_rag and HAS_VECTOR_STORE
        self.vector_store = None
        self.embeddings = None
        
        if self.enable_rag:
            try:
                # Initialize embeddings (using OpenAI by default)
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.embeddings = OpenAIEmbeddings(
                        model="text-embedding-3-small",
                        openai_api_key=api_key,
                    )
                else:
                    print("[RAG] OPENAI_API_KEY not set. RAG disabled.")
                    self.enable_rag = False
            except Exception as e:
                print(f"[RAG] Failed to initialize embeddings: {e}")
                self.enable_rag = False
    
    def _load_vector_store(self, ticker: str) -> Optional[Any]:
        """
        Load vector store for a specific ticker.
        
        In a full implementation, you'd have pre-indexed transcripts.
        For now, this is a placeholder that returns None.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Vector store instance or None
        """
        if not self.enable_rag:
            return None
        
        # TODO: Load pre-indexed vector store from disk
        # Example:
        # store_path = Path(f"backend/rag/stores/{ticker.lower()}.faiss")
        # if store_path.exists():
        #     return FAISS.load_local(str(store_path), self.embeddings)
        
        return None
    
    async def retrieve(
        self,
        ticker: str,
        query: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        This is the main RAG function - it searches the vector store
        for semantically similar content and returns the top results.
        
        Args:
            ticker: Stock ticker symbol
            query: Search query (e.g., "AI infrastructure investment")
            top_k: Number of results to return
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.enable_rag:
            return []
        
        vector_store = self._load_vector_store(ticker)
        if vector_store is None:
            # No vector store available - return empty (graceful degradation)
            return []
        
        try:
            # Perform semantic search
            docs = vector_store.similarity_search(query, k=top_k)
            
            # Format results
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get("source", "unknown"),
                })
            
            return results
        
        except Exception as e:
            print(f"[RAG] Retrieval failed: {e}")
            return []
    
    async def retrieve_earnings_context(
        self,
        ticker: str,
        quarters: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve context from recent earnings transcripts.
        
        This is a convenience method that retrieves the most relevant
        snippets from the last N quarters of earnings calls.
        
        Args:
            ticker: Stock ticker symbol
            quarters: Number of recent quarters to search
            
        Returns:
            List of relevant earnings call snippets
        """
        # Query for strategic topics that agents care about
        queries = [
            f"{ticker} CEO strategic priorities",
            f"{ticker} capital allocation guidance",
            f"{ticker} competitive advantages",
            f"{ticker} growth outlook",
        ]
        
        all_results = []
        for query in queries:
            results = await self.retrieve(ticker, query, top_k=1)
            all_results.extend(results)
        
        # Deduplicate and return top results
        seen = set()
        unique_results = []
        for result in all_results:
            content_hash = hash(result["content"][:100])  # Hash first 100 chars
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(result)
        
        return unique_results[:quarters]
    
    def add_documents(
        self,
        ticker: str,
        documents: List[Dict[str, str]],
    ):
        """
        Add documents to the vector store.
        
        This is for indexing earnings transcripts.
        In production, you'd call this when new transcripts are available.
        
        Args:
            ticker: Stock ticker symbol
            documents: List of dicts with "content" and "metadata" keys
        """
        if not self.enable_rag:
            return
        
        try:
            # Convert to LangChain Document format
            docs = [
                Document(
                    page_content=doc["content"],
                    metadata=doc.get("metadata", {}),
                )
                for doc in documents
            ]
            
            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(docs, self.embeddings)
            else:
                # Add to existing store
                self.vector_store.add_documents(docs)
            
            # Save to disk (for persistence)
            store_path = Path(f"backend/rag/stores/{ticker.lower()}.faiss")
            store_path.parent.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(store_path))
        
        except Exception as e:
            print(f"[RAG] Failed to add documents: {e}")
