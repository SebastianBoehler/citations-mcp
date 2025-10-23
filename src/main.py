"""Main application entry point for Research Citations MCP Server."""

import logging
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.mcp_server import mcp, initialize_rag
from src.sse_transport import create_sse_server

# Suppress PDF parsing warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")
warnings.filterwarnings("ignore", message=".*FloatObject.*")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Starting Research Citations MCP Server...")
    logger.info(f"Papers directory: {settings.papers_directory}")
    logger.info(f"Vector DB path: {settings.vector_db_path}")
    
    # Initialize RAG engine on startup
    await initialize_rag()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Research Citations MCP Server...")


# Create FastAPI application
app = FastAPI(
    title="Research Citations MCP Server",
    description="MCP server for searching and citing research papers using RAG",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {
        "name": "Research Citations MCP Server",
        "status": "running",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "mcp_sse": "/mcp/sse",
            "mcp_messages": "/mcp/messages",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "papers_directory": str(settings.papers_directory),
        "vector_db_path": str(settings.vector_db_path),
    }


# Mount MCP SSE server at /mcp
sse_app = create_sse_server(mcp)
app.mount("/mcp", sse_app)

logger.info("FastAPI application created with MCP SSE transport")
