"""MCP server for research paper citations with RAG capabilities."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import mcp.types as types
from mcp.server.fastmcp import FastMCP

from src.config import get_settings
from src.rag_engine import RAGEngine, PaperMetadataLevel

logger = logging.getLogger(__name__)

# Initialize settings and RAG engine
settings = get_settings()
rag_engine = RAGEngine()

# Create FastMCP server
mcp = FastMCP("Research Citations MCP Server")

WIDGET_URI = "ui://widget/find-citations.html"
PAPERS_WIDGET_URI = "ui://widget/find-papers.html"
LIST_PAPERS_WIDGET_URI = "ui://widget/list-papers.html"
_WIDGET_HTML_CACHE: str | None = None
_PAPERS_WIDGET_HTML_CACHE: str | None = None
_LIST_PAPERS_WIDGET_HTML_CACHE: str | None = None
_WIDGET_HTML_PATH = (
    Path(__file__).resolve().parent.parent / "public" / "find-citations.html"
)
_PAPERS_WIDGET_HTML_PATH = (
    Path(__file__).resolve().parent.parent / "public" / "find-papers.html"
)
_LIST_PAPERS_WIDGET_HTML_PATH = (
    Path(__file__).resolve().parent.parent / "public" / "list-papers.html"
)

WIDGET_META = {
    "openai/outputTemplate": WIDGET_URI,
    "openai/toolInvocation/invoking": "Gathering citations",
    "openai/toolInvocation/invoked": "Rendered citations",
    "openai/widgetAccessible": True,
    "openai/resultCanProduceWidget": True,
}

PAPERS_WIDGET_META = {
    "openai/outputTemplate": PAPERS_WIDGET_URI,
    "openai/toolInvocation/invoking": "Searching papers",
    "openai/toolInvocation/invoked": "Rendered paper search results",
    "openai/widgetAccessible": True,
    "openai/resultCanProduceWidget": True,
}

LIST_PAPERS_WIDGET_META = {
    "openai/outputTemplate": LIST_PAPERS_WIDGET_URI,
    "openai/toolInvocation/invoking": "Listing papers",
    "openai/toolInvocation/invoked": "Rendered paper list",
    "openai/widgetAccessible": True,
    "openai/resultCanProduceWidget": True,
}


def _load_widget_html() -> str:
    """Load the Apps SDK widget HTML from disk (cached)."""
    global _WIDGET_HTML_CACHE  # noqa: PLW0603 - cache maintained module-wide
    if _WIDGET_HTML_CACHE is None:
        try:
            _WIDGET_HTML_CACHE = _WIDGET_HTML_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error(
                "Citation widget HTML not found at %s. Falling back to placeholder.",
                _WIDGET_HTML_PATH,
            )
            _WIDGET_HTML_CACHE = (
                "<html><body><p>Citation widget missing from server deployment.</p></body></html>"
            )
    return _WIDGET_HTML_CACHE


def _load_papers_widget_html() -> str:
    """Load the Apps SDK papers widget HTML from disk (cached)."""
    global _PAPERS_WIDGET_HTML_CACHE  # noqa: PLW0603 - cache maintained module-wide
    if _PAPERS_WIDGET_HTML_CACHE is None:
        try:
            _PAPERS_WIDGET_HTML_CACHE = _PAPERS_WIDGET_HTML_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error(
                "Papers widget HTML not found at %s. Falling back to placeholder.",
                _PAPERS_WIDGET_HTML_PATH,
            )
            _PAPERS_WIDGET_HTML_CACHE = (
                "<html><body><p>Papers widget missing from server deployment.</p></body></html>"
            )
    return _PAPERS_WIDGET_HTML_CACHE


def _load_list_papers_widget_html() -> str:
    """Load the Apps SDK list papers widget HTML from disk (cached)."""
    global _LIST_PAPERS_WIDGET_HTML_CACHE  # noqa: PLW0603 - cache maintained module-wide
    if _LIST_PAPERS_WIDGET_HTML_CACHE is None:
        try:
            _LIST_PAPERS_WIDGET_HTML_CACHE = _LIST_PAPERS_WIDGET_HTML_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error(
                "List papers widget HTML not found at %s. Falling back to placeholder.",
                _LIST_PAPERS_WIDGET_HTML_PATH,
            )
            _LIST_PAPERS_WIDGET_HTML_CACHE = (
                "<html><body><p>List papers widget missing from server deployment.</p></body></html>"
            )
    return _LIST_PAPERS_WIDGET_HTML_CACHE


@mcp.resource(
    WIDGET_URI,
    name="find-citations-widget",
    title="Find Citations Widget",
    description="Renders grouped citations with bibliography and excerpts.",
    mime_type="text/html+skybridge",
)
async def find_citations_widget() -> str:
    """Expose the Apps SDK widget HTML to ChatGPT."""
    return _load_widget_html()


@mcp.resource(
    PAPERS_WIDGET_URI,
    name="find-papers-widget",
    title="Find Papers Widget",
    description="Renders paper search results with scores and content.",
    mime_type="text/html+skybridge",
)
async def find_papers_widget() -> str:
    """Expose the Apps SDK papers widget HTML to ChatGPT."""
    return _load_papers_widget_html()


@mcp.resource(
    LIST_PAPERS_WIDGET_URI,
    name="list-papers-widget",
    title="List Papers Widget",
    description="Renders a minimalistic list of all papers.",
    mime_type="text/html+skybridge",
)
async def list_papers_widget() -> str:
    """Expose the Apps SDK list papers widget HTML to ChatGPT."""
    return _load_list_papers_widget_html()


@mcp._mcp_server.list_tools()
async def _apps_list_tools() -> List[types.Tool]:
    """Expose tools with Apps metadata so ChatGPT can discover widgets."""
    tool_infos = mcp._tool_manager.list_tools()
    results = []
    for info in tool_infos:
        if info.name == "find_citation":
            meta = WIDGET_META
        elif info.name == "search_papers":
            meta = PAPERS_WIDGET_META
        elif info.name == "list_papers":
            meta = LIST_PAPERS_WIDGET_META
        else:
            meta = None
        annotations = info.annotations
        results.append(
            types.Tool(
                name=info.name,
                title=info.title,
                description=info.description,
                inputSchema=info.parameters,
                outputSchema=info.output_schema,
                annotations=annotations,
                icons=info.icons,
                _meta=meta,
            )
        )
    return results


@mcp._mcp_server.list_resources()
async def _apps_list_resources() -> List[types.Resource]:
    """Expose widget resources with Apps metadata."""
    resources = []
    for resource in mcp._resource_manager.list_resources():
        uri = str(resource.uri)
        if uri == WIDGET_URI:
            meta = WIDGET_META
        elif uri == PAPERS_WIDGET_URI:
            meta = PAPERS_WIDGET_META
        elif uri == LIST_PAPERS_WIDGET_URI:
            meta = LIST_PAPERS_WIDGET_META
        else:
            meta = None
        resources.append(
            types.Resource(
                name=resource.name or "",
                title=resource.title,
                uri=uri,
                description=resource.description,
                mimeType=resource.mime_type,
                icons=resource.icons,
                _meta=meta,
            )
        )
    return resources


@mcp._mcp_server.list_resource_templates()
async def _apps_list_resource_templates() -> List[types.ResourceTemplate]:
    """Include metadata for widget resource templates."""
    templates = []
    for template in mcp._resource_manager.list_templates():
        if template.uri_template == WIDGET_URI:
            meta = WIDGET_META
        elif template.uri_template == PAPERS_WIDGET_URI:
            meta = PAPERS_WIDGET_META
        elif template.uri_template == LIST_PAPERS_WIDGET_URI:
            meta = LIST_PAPERS_WIDGET_META
        else:
            meta = None
        templates.append(
            types.ResourceTemplate(
                name=template.name or "",
                title=template.title,
                uriTemplate=template.uri_template,
                description=template.description,
                mimeType=template.mime_type,
                _meta=meta,
            )
        )
    return templates


@mcp.tool(annotations=PAPERS_WIDGET_META)
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
            "content": [{"type": "text", "text": f"Found {len(results)} results for '{query}'"}],
            "structuredContent": {
                "query": query,
                "num_results": len(results),
                "results": results,
            },
            "_meta": PAPERS_WIDGET_META,
        }
    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        error_payload = {
            "error": str(e),
            "query": query,
        }
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Failed to search papers for '{query}': {e}",
                }
            ],
            "structuredContent": error_payload,
            "_meta": PAPERS_WIDGET_META,
        }


@mcp.tool(annotations={"openai/outputTemplate": WIDGET_URI})
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
        summary = (
            f"Found {citations.get('total_sources', 0)} source(s) for '{topic}'."
        )
        return {
            "content": [{"type": "text", "text": summary}],
            "structuredContent": citations,
            "_meta": WIDGET_META,
        }
    except Exception as e:
        logger.error(f"Error finding citations: {e}")
        error_payload = {
            "error": str(e),
            "topic": topic,
        }
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Failed to find citations for '{topic}': {e}",
                }
            ],
            "structuredContent": error_payload,
            "_meta": WIDGET_META,
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


@mcp.tool(annotations=LIST_PAPERS_WIDGET_META)
async def list_papers(
    metadata_level: str = "filename_only"
) -> Dict[str, Any]:
    """
    List all research papers currently indexed in the system with optional metadata.
    
    Args:
        metadata_level: Level of detail to return. Options:
            - "filename_only": Just the filename (default)
            - "with_authors": Filename, title, and authors
            - "with_bibliography": Filename, title, and APA citation
            - "full": All available metadata (title, authors, year, publication, DOI, APA citation)
    
    Returns:
        Dictionary with list of papers (format depends on metadata_level)
    """
    try:
        # Convert string to enum
        try:
            level = PaperMetadataLevel(metadata_level)
        except ValueError:
            error_payload = {
                "error": f"Invalid metadata_level: {metadata_level}",
                "valid_options": [level.value for level in PaperMetadataLevel],
            }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Invalid metadata_level: {metadata_level}",
                    }
                ],
                "structuredContent": error_payload,
                "_meta": LIST_PAPERS_WIDGET_META,
            }
        
        papers = rag_engine.list_papers(metadata_level=level)
        
        return {
            "content": [{"type": "text", "text": f"Found {len(papers)} papers in the collection"}],
            "structuredContent": {
                "total_papers": len(papers),
                "metadata_level": metadata_level,
                "papers": papers,
            },
            "_meta": LIST_PAPERS_WIDGET_META,
        }
    except Exception as e:
        logger.error(f"Error listing papers: {e}")
        error_payload = {
            "error": str(e),
        }
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Failed to list papers: {e}",
                }
            ],
            "structuredContent": error_payload,
            "_meta": LIST_PAPERS_WIDGET_META,
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
        await rag_engine.initialize_vector_store(force_rebuild=force)
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


@mcp.tool()
async def extract_methodology(filename: str) -> Dict[str, Any]:
    """
    Extract methodology details from a research paper including datasets, 
    models, evaluation metrics, and experimental setup.
    
    Args:
        filename: Name of the PDF file (e.g., "smith_2023.pdf")
        
    Returns:
        Dictionary with structured methodology information
    """
    try:
        result = rag_engine.extract_methodology(filename)
        return result
    except Exception as e:
        logger.error(f"Error extracting methodology: {e}")
        return {
            "error": str(e),
            "filename": filename,
        }


@mcp.tool()
async def summarize_paper(
    filename: str, 
    custom_prompt: str = None,
    focus: str = "general"
) -> Dict[str, Any]:
    """
    Generate a summary of a research paper with optional custom focus.
    
    Args:
        filename: Name of the PDF file (e.g., "smith_2023.pdf")
        custom_prompt: Optional custom prompt for specific summarization needs
                      (e.g., "Summarize the experimental results and their statistical significance")
        focus: Pre-defined focus if custom_prompt not used. Options:
               - "general": Comprehensive overview (default)
               - "key_findings": Main results and discoveries
               - "methodology": Research methods and approach
               - "limitations": Constraints and future work
               - "contributions": Main contributions to the field
        
    Returns:
        Dictionary with paper summary
    """
    try:
        result = rag_engine.summarize_paper(
            filename=filename,
            custom_prompt=custom_prompt,
            focus=focus
        )
        return result
    except Exception as e:
        logger.error(f"Error summarizing paper: {e}")
        return {
            "error": str(e),
            "filename": filename,
        }


@mcp.tool()
async def extract_bibliography(filename: str) -> Dict[str, Any]:
    """
    Extract the bibliography/references from a research paper and format in APA style.
    This shows what citations are used within the paper.
    
    Args:
        filename: Name of the PDF file (e.g., "smith_2023.pdf")
        
    Returns:
        Dictionary with bibliography in APA format
    """
    try:
        result = rag_engine.extract_bibliography(filename)
        return result
    except Exception as e:
        logger.error(f"Error extracting bibliography: {e}")
        return {
            "error": str(e),
            "filename": filename,
        }


# Initialize the RAG engine when the module is loaded
async def initialize_rag():
    """Initialize the RAG engine and vector store."""
    logger.info("Initializing RAG engine...")
    try:
        await rag_engine.initialize_vector_store(force_rebuild=False)
        papers = rag_engine.list_papers()
        logger.info(f"RAG engine initialized with {len(papers)} papers")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        logger.warning("Server starting without initialized vector store")


# Export the mcp instance and initialization function for use in main.py
__all__ = ["mcp", "initialize_rag"]
