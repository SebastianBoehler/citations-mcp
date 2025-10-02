"""Tests for PDF processor module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from langchain.schema import Document

from src.pdf_processor import PDFProcessor


@pytest.fixture
def pdf_processor():
    """Create a PDF processor instance for testing."""
    with patch('src.pdf_processor.get_settings') as mock_settings:
        mock_settings.return_value = Mock(
            chunk_size=1000,
            chunk_overlap=200,
            papers_directory=Path("/tmp/test_papers")
        )
        return PDFProcessor()


def test_pdf_processor_initialization(pdf_processor):
    """Test PDF processor initializes correctly."""
    assert pdf_processor is not None
    assert pdf_processor.text_splitter is not None


def test_chunk_documents(pdf_processor):
    """Test document chunking."""
    # Create test documents
    docs = [
        Document(
            page_content="This is a test document. " * 100,
            metadata={"source": "test.pdf", "page": 0}
        )
    ]
    
    # Chunk the documents
    chunks = pdf_processor.chunk_documents(docs)
    
    # Verify chunks were created
    assert len(chunks) > 0
    
    # Verify metadata was added
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "total_chunks" in chunk.metadata


@pytest.mark.asyncio
async def test_load_pdf():
    """Test PDF loading (mock)."""
    with patch('src.pdf_processor.get_settings') as mock_settings:
        mock_settings.return_value = Mock(
            chunk_size=1000,
            chunk_overlap=200,
        )
        
        processor = PDFProcessor()
        
        # Mock PyPDFLoader
        with patch('src.pdf_processor.PyPDFLoader') as mock_loader:
            mock_doc = Document(
                page_content="Test content",
                metadata={"source": "test.pdf", "page": 0}
            )
            
            mock_instance = Mock()
            mock_instance.alazy_load = AsyncMock(return_value=[mock_doc].__aiter__())
            mock_loader.return_value = mock_instance
            
            # Load PDF
            result = await processor.load_pdf(Path("test.pdf"))
            
            # Verify result
            assert len(result) > 0
            assert result[0].metadata["filename"] == "test.pdf"
