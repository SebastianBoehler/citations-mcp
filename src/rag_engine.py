"""RAG engine for semantic search over research papers using LangChain and ChromaDB."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from src.config import get_settings
from src.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


class RAGEngine:
    """RAG engine for searching and retrieving information from research papers."""

    def __init__(self):
        """Initialize RAG engine with vector store and retriever."""
        self.settings = get_settings()
        self.pdf_processor = PDFProcessor()
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.settings.llm_model,
            openai_api_key=self.settings.openai_api_key,
            temperature=0,
        )
        
        # Vector store (initialized later)
        self.vector_store: Optional[Chroma] = None
        self.retriever = None
        
    def initialize_vector_store(
        self, force_rebuild: bool = False
    ) -> None:
        """
        Initialize or load the vector store.

        Args:
            force_rebuild: If True, rebuild the vector store from scratch
        """
        persist_directory = str(self.settings.vector_db_path)
        
        # Check if vector store exists and should be loaded
        if not force_rebuild and Path(persist_directory).exists():
            logger.info(f"Loading existing vector store from {persist_directory}")
            self.vector_store = Chroma(
                collection_name=self.settings.collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory,
            )
        else:
            # Build new vector store
            logger.info("Building new vector store from research papers...")
            documents = self.pdf_processor.process_directory_sync()
            
            if not documents:
                raise ValueError("No documents found to index")
            
            logger.info(f"Creating vector store with {len(documents)} chunks")
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.settings.collection_name,
                persist_directory=persist_directory,
            )
            logger.info(f"Vector store created and persisted to {persist_directory}")
        
        # Initialize retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        
    def search_papers(
        self, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant passages in research papers.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of dictionaries containing search results with metadata
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
        
        # Perform similarity search
        docs = self.vector_store.similarity_search(query, k=k)
        
        # Format results
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                },
            })
        
        return results
    
    def search_with_scores(
        self, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant passages with similarity scores.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of dictionaries containing search results with scores
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
        
        # Perform similarity search with scores
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Format results
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "content": doc.page_content,
                "score": float(score),
                "metadata": {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                },
            })
        
        return results
    
    def search_in_paper(
        self, query: str, filename: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant passages within a specific paper only.

        Args:
            query: Search query
            filename: Name of the PDF file to search in
            k: Number of results to return

        Returns:
            List of dictionaries containing search results with scores
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
        
        # First, get all chunks from this specific paper
        try:
            # Perform similarity search with metadata filter
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query,
                k=k * 3,  # Get more results to filter
                filter={"filename": filename}
            )
            
            # Take top k after filtering
            docs_with_scores = docs_with_scores[:k]
            
        except Exception:
            # Fallback: manual filtering if the vector store doesn't support metadata filtering
            all_docs = self.vector_store.similarity_search_with_score(query, k=k * 10)
            docs_with_scores = [
                (doc, score) for doc, score in all_docs 
                if doc.metadata.get("filename") == filename
            ][:k]
        
        if not docs_with_scores:
            return []
        
        # Format results
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "content": doc.page_content,
                "score": float(score),
                "metadata": {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                },
            })
        
        return results
    
    def find_citation(
        self, topic: str, k: int = 3
    ) -> Dict[str, Any]:
        """
        Find relevant citations for a given topic.

        Args:
            topic: Topic or concept to find citations for
            k: Number of citations to find

        Returns:
            Dictionary with citations and context
        """
        results = self.search_with_scores(topic, k=k)
        
        # Group by source document
        citations_by_paper = {}
        for result in results:
            filename = result["metadata"]["filename"]
            if filename not in citations_by_paper:
                citations_by_paper[filename] = []
            
            citations_by_paper[filename].append({
                "page": result["metadata"]["page"],
                "excerpt": result["content"][:300] + "..." if len(result["content"]) > 300 else result["content"],
                "relevance_score": result["score"],
            })
        
        return {
            "topic": topic,
            "citations": citations_by_paper,
            "total_sources": len(citations_by_paper),
        }
    
    def answer_question(
        self, question: str, return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Args:
            question: Question to answer
            return_sources: Whether to return source documents

        Returns:
            Dictionary with answer and optional sources
        """
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized. Call initialize_vector_store() first.")
        
        # Create QA chain with custom prompt
        prompt_template = """You are a research assistant helping with a thesis. 
Use the following pieces of context from research papers to answer the question. 
If you quote from the papers, indicate which paper and page the quote is from.
If you don't know the answer based on the provided context, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=return_sources,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        result = qa_chain.invoke({"query": question})
        
        # Format response
        response = {
            "question": question,
            "answer": result["result"],
        }
        
        if return_sources and "source_documents" in result:
            response["sources"] = [
                {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                }
                for doc in result["source_documents"]
            ]
        
        return response
    
    def get_paper_info(self, filename: str) -> Dict[str, Any]:
        """
        Get information about a specific paper.

        Args:
            filename: Name of the PDF file

        Returns:
            Dictionary with paper information
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
        
        # Search for documents from this file
        all_docs = self.vector_store.get(
            where={"filename": filename}
        )
        
        if not all_docs or not all_docs.get("documents"):
            return {
                "filename": filename,
                "found": False,
                "message": f"No documents found for {filename}",
            }
        
        # Extract metadata from first document
        metadata = all_docs["metadatas"][0] if all_docs.get("metadatas") else {}
        
        return {
            "filename": filename,
            "found": True,
            "total_chunks": len(all_docs["documents"]),
            "source": metadata.get("source", "Unknown"),
            "pages": metadata.get("page", "Unknown"),
        }
    
    def list_papers(self) -> List[str]:
        """
        List all papers in the vector store.

        Returns:
            List of paper filenames
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
        
        # Get all documents
        collection = self.vector_store._collection
        all_metadata = collection.get()
        
        # Extract unique filenames
        filenames = set()
        if all_metadata.get("metadatas"):
            for metadata in all_metadata["metadatas"]:
                if "filename" in metadata:
                    filenames.add(metadata["filename"])
        
        return sorted(list(filenames))
