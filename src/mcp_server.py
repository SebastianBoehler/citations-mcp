"""MCP server for research paper citations with RAG capabilities."""

import logging
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from src.config import get_settings
from src.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

# Initialize settings and RAG engine
settings = get_settings()
rag_engine = RAGEngine()

# Create FastMCP server
mcp = FastMCP("Research Citations MCP Server")


@mcp.tool()
async def search_papers(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search for relevant passages in research papers using semantic search.
    
    Args:
        query: The search query or topic to find in papers
        num_results: Number of results to return (default: 5)
        
    Returns:
        Dictionary containing search results with content and metadata
    """
    try:
        results = rag_engine.search_with_scores(query, k=num_results)
        
        return {
            "query": query,
            "num_results": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        return {
            "error": str(e),
            "query": query,
        }


@mcp.tool()
async def find_citation(topic: str, num_citations: int = 3) -> Dict[str, Any]:
    """
    Find relevant citations for a specific topic or concept.
    Groups results by source paper for easy citation.
    
    Args:
        topic: The topic or concept to find citations for
        num_citations: Number of citations to find (default: 3)
        
    Returns:
        Dictionary with citations grouped by paper
    """
    try:
        citations = rag_engine.find_citation(topic, k=num_citations)
        return citations
    except Exception as e:
        logger.error(f"Error finding citations: {e}")
        return {
            "error": str(e),
            "topic": topic,
        }


@mcp.tool()
async def answer_question(question: str) -> Dict[str, Any]:
    """
    Answer a research question using RAG over your papers.
    Provides answer with source references.
    
    Args:
        question: The research question to answer
        
    Returns:
        Dictionary with answer and source documents
    """
    try:
        response = rag_engine.answer_question(question, return_sources=True)
        return response
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return {
            "error": str(e),
            "question": question,
        }


@mcp.tool()
async def get_paper_info(filename: str) -> Dict[str, Any]:
    """
    Get information about a specific research paper.
    
    Args:
        filename: Name of the PDF file (e.g., "smith_2023.pdf")
        
    Returns:
        Dictionary with paper information and metadata
    """
    try:
        info = rag_engine.get_paper_info(filename)
        return info
    except Exception as e:
        logger.error(f"Error getting paper info: {e}")
        return {
            "error": str(e),
            "filename": filename,
        }


@mcp.tool()
async def search_in_paper(query: str, filename: str, num_results: int = 5) -> Dict[str, Any]:
    """
    Search for relevant passages within a specific research paper only.
    Useful when you want to find citations from just one particular paper.
    
    Args:
        query: The search query or topic to find
        filename: Name of the PDF file to search in (e.g., "smith_2023.pdf")
        num_results: Number of results to return (default: 5)
        
    Returns:
        Dictionary containing search results from the specified paper
    """
    try:
        results = rag_engine.search_in_paper(query, filename, k=num_results)
        
        if not results:
            return {
                "query": query,
                "filename": filename,
                "found": False,
                "message": f"No results found in '{filename}' for query '{query}'",
                "results": [],
            }
        
        return {
            "query": query,
            "filename": filename,
            "found": True,
            "num_results": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Error searching in paper: {e}")
        return {
            "error": str(e),
            "query": query,
            "filename": filename,
        }


@mcp.tool()
async def list_papers() -> Dict[str, Any]:
    """
    List all research papers currently indexed in the system.
    
    Returns:
        Dictionary with list of paper filenames
    """
    try:
        papers = rag_engine.list_papers()
        return {
            "total_papers": len(papers),
            "papers": papers,
        }
    except Exception as e:
        logger.error(f"Error listing papers: {e}")
        return {
            "error": str(e),
        }


@mcp.tool()
async def rebuild_index(force: bool = False) -> Dict[str, Any]:
    """
    Rebuild the vector store index from the papers directory.
    Use this when you add new papers or want to refresh the index.
    
    Args:
        force: Force rebuild even if index exists (default: False)
        
    Returns:
        Dictionary with rebuild status
    """
    try:
        logger.info(f"Rebuilding index (force={force})...")
        rag_engine.initialize_vector_store(force_rebuild=force)
        papers = rag_engine.list_papers()
        
        return {
            "success": True,
            "message": "Index rebuilt successfully",
            "total_papers": len(papers),
            "papers": papers,
        }
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# Initialize the RAG engine when the module is loaded
def initialize_rag():
    """Initialize the RAG engine and vector store."""
    logger.info("Initializing RAG engine...")
    try:
        rag_engine.initialize_vector_store(force_rebuild=False)
        papers = rag_engine.list_papers()
        logger.info(f"RAG engine initialized with {len(papers)} papers")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        logger.warning("Server starting without initialized vector store")


# Export the mcp instance and initialization function for use in main.py
__all__ = ["mcp", "initialize_rag"]
