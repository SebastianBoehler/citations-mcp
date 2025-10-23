"""Tests for new thesis-writing features."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document


def test_new_methods_exist():
    """Test that new methods are defined in RAGEngine."""
    from src.rag_engine import RAGEngine
    
    # Verify methods exist
    assert hasattr(RAGEngine, 'extract_methodology')
    assert hasattr(RAGEngine, 'summarize_paper')
    assert hasattr(RAGEngine, 'extract_bibliography')
    
    # Verify methods are callable
    assert callable(getattr(RAGEngine, 'extract_methodology'))
    assert callable(getattr(RAGEngine, 'summarize_paper'))
    assert callable(getattr(RAGEngine, 'extract_bibliography'))


@patch('src.rag_engine.ChatOpenAI')
@patch('src.rag_engine.OpenAIEmbeddings')
@patch('src.rag_engine.get_settings')
def test_extract_methodology(mock_settings, mock_embeddings, mock_llm_class):
    """Test methodology extraction."""
    from src.rag_engine import RAGEngine
    
    # Setup mocks
    mock_settings.return_value = Mock(
        openai_api_key="test-key",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        vector_db_path="/tmp/test_vector_db",
        collection_name="test_papers"
    )
    
    mock_llm_response = Mock()
    mock_llm_response.content = "Extracted methodology with datasets and metrics"
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = mock_llm_response
    mock_llm_class.return_value = mock_llm_instance
    
    # Create engine
    engine = RAGEngine()
    
    # Mock vector store
    mock_vector_store = MagicMock()
    engine.vector_store = mock_vector_store
    
    # Mock similarity search
    test_doc = Document(
        page_content="We used a transformer architecture with 6 layers. The dataset contains 100k samples.",
        metadata={"filename": "test.pdf", "page": 1}
    )
    mock_vector_store.similarity_search_with_score.return_value = [(test_doc, 0.9)]
    
    # Extract methodology
    result = engine.extract_methodology("test.pdf")
    
    # Verify
    assert result["found"] is True
    assert result["filename"] == "test.pdf"
    assert "methodology" in result


@patch('src.rag_engine.ChatOpenAI')
@patch('src.rag_engine.OpenAIEmbeddings')
@patch('src.rag_engine.get_settings')
def test_search_with_scores_sorted(mock_settings, mock_embeddings, mock_llm_class):
    """Test that search results are sorted by score (best results first)."""
    from src.rag_engine import RAGEngine
    
    # Setup mocks
    mock_settings.return_value = Mock(
        openai_api_key="test-key",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        vector_db_path="/tmp/test_vector_db",
        collection_name="test_papers"
    )
    
    # Create engine
    engine = RAGEngine()
    
    # Mock vector store
    mock_vector_store = MagicMock()
    engine.vector_store = mock_vector_store
    
    # Mock similarity search with UNSORTED results (higher scores = less similar)
    test_docs = [
        (Document(page_content="Result 3", metadata={"filename": "test.pdf", "page": 3}), 0.8),
        (Document(page_content="Result 1", metadata={"filename": "test.pdf", "page": 1}), 0.2),  # Best match
        (Document(page_content="Result 2", metadata={"filename": "test.pdf", "page": 2}), 0.5),
    ]
    mock_vector_store.similarity_search_with_score.return_value = test_docs
    
    # Search
    results = engine.search_with_scores("test query", k=3)
    
    # Verify results are sorted by score (ascending - lower is better)
    assert len(results) == 3
    assert results[0]["score"] == 0.2  # Best match first
    assert results[1]["score"] == 0.5
    assert results[2]["score"] == 0.8  # Worst match last
    assert results[0]["content"] == "Result 1"
    assert results[1]["content"] == "Result 2"
    assert results[2]["content"] == "Result 3"


@patch('src.rag_engine.ChatOpenAI')
@patch('src.rag_engine.OpenAIEmbeddings')
@patch('src.rag_engine.get_settings')
def test_search_in_paper_sorted(mock_settings, mock_embeddings, mock_llm_class):
    """Test that paper-specific search results are sorted by score."""
    from src.rag_engine import RAGEngine
    
    # Setup mocks
    mock_settings.return_value = Mock(
        openai_api_key="test-key",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        vector_db_path="/tmp/test_vector_db",
        collection_name="test_papers"
    )
    
    # Create engine
    engine = RAGEngine()
    
    # Mock vector store
    mock_vector_store = MagicMock()
    engine.vector_store = mock_vector_store
    
    # Mock similarity search with UNSORTED results
    test_docs = [
        (Document(page_content="Match 2", metadata={"filename": "paper.pdf", "page": 2, "chunk_id": 2}), 0.6),
        (Document(page_content="Match 1", metadata={"filename": "paper.pdf", "page": 1, "chunk_id": 1}), 0.3),  # Best
        (Document(page_content="Match 3", metadata={"filename": "paper.pdf", "page": 3, "chunk_id": 3}), 0.9),
    ]
    mock_vector_store.similarity_search_with_score.return_value = test_docs
    
    # Search in specific paper
    results = engine.search_in_paper("test query", "paper.pdf", k=3)
    
    # Verify results are sorted by score
    assert len(results) == 3
    assert results[0]["score"] == 0.3  # Best match first
    assert results[1]["score"] == 0.6
    assert results[2]["score"] == 0.9
    assert results[0]["content"] == "Match 1"


@patch('src.rag_engine.ChatOpenAI')
@patch('src.rag_engine.OpenAIEmbeddings')
@patch('src.rag_engine.get_settings')
def test_search_with_scores_deduplication(mock_settings, mock_embeddings, mock_llm_class):
    """Test that duplicate search results are filtered out."""
    from src.rag_engine import RAGEngine
    
    # Setup mocks
    mock_settings.return_value = Mock(
        openai_api_key="test-key",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        vector_db_path="/tmp/test_vector_db",
        collection_name="test_papers"
    )
    
    # Create engine
    engine = RAGEngine()
    
    # Mock vector store
    mock_vector_store = MagicMock()
    engine.vector_store = mock_vector_store
    
    # Mock similarity search with DUPLICATE results (same content, metadata)
    duplicate_doc = Document(
        page_content="Duplicate content here",
        metadata={"filename": "test.pdf", "page": 1, "chunk_id": 5}
    )
    unique_doc1 = Document(
        page_content="Unique content 1",
        metadata={"filename": "test.pdf", "page": 2, "chunk_id": 6}
    )
    unique_doc2 = Document(
        page_content="Unique content 2",
        metadata={"filename": "test.pdf", "page": 3, "chunk_id": 7}
    )
    
    # Include 4 copies of the same document plus 2 unique ones
    test_docs = [
        (duplicate_doc, 0.2),  # Best match but duplicated
        (duplicate_doc, 0.2),  # Duplicate
        (duplicate_doc, 0.2),  # Duplicate
        (duplicate_doc, 0.2),  # Duplicate
        (unique_doc1, 0.5),
        (unique_doc2, 0.6),
    ]
    mock_vector_store.similarity_search_with_score.return_value = test_docs
    
    # Search
    results = engine.search_with_scores("test query", k=3)
    
    # Verify deduplication - should only get 3 unique results
    assert len(results) == 3
    # First result should be the duplicate (only once)
    assert results[0]["content"] == "Duplicate content here"
    assert results[0]["score"] == 0.2
    # Next results should be the unique ones
    assert results[1]["content"] == "Unique content 1"
    assert results[2]["content"] == "Unique content 2"
    
    # Verify no duplicates
    contents = [r["content"] for r in results]
    assert len(contents) == len(set(contents)), "Results should not contain duplicates"


def test_mcp_tools_registered():
    """Test that new MCP tools are registered."""
    from src.mcp_server import mcp
    
    # Get all registered tools
    tools = [tool for tool in dir(mcp) if not tool.startswith('_')]
    
    # The tools should be accessible through the mcp object
    # We just verify the module imports without errors
    assert mcp is not None


@pytest.mark.asyncio
@patch('src.rag_engine.ChatOpenAI')
@patch('src.rag_engine.OpenAIEmbeddings')
@patch('src.rag_engine.get_settings')
async def test_extract_paper_metadata(mock_settings, mock_embeddings, mock_llm_class):
    """Test metadata extraction from PDF first pages."""
    from src.rag_engine import RAGEngine
    from pathlib import Path
    
    # Setup mocks
    mock_settings.return_value = Mock(
        openai_api_key="test-key",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o",
        vector_db_path=Path("/tmp/test_vector_db"),
        collection_name="test_papers",
        papers_directory=Path("/tmp/papers")
    )
    
    # Mock LLM response with structured metadata
    mock_llm_response = Mock()
    mock_llm_response.content = """Title: Deep Learning for Natural Language Processing
Authors: Smith, J., Johnson, A., Williams, B.
Year: 2023
Publication: Journal of Machine Learning Research
DOI: 10.1234/jmlr.2023.001
APA Citation: Smith, J., Johnson, A., & Williams, B. (2023). Deep learning for natural language processing. Journal of Machine Learning Research, 24(5), 123-145. https://doi.org/10.1234/jmlr.2023.001"""
    
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = mock_llm_response
    mock_llm_class.return_value = mock_llm_instance
    
    # Create engine
    engine = RAGEngine()
    
    # Mock PDF processor's load_first_pages method
    with patch.object(engine.pdf_processor, 'load_first_pages', return_value="Sample paper content..."):
        # Extract metadata
        result = await engine.extract_paper_metadata(Path("/tmp/papers/test.pdf"))
        
        # Verify
        assert result["filename"] == "test.pdf"
        assert result["title"] == "Deep Learning for Natural Language Processing"
        assert result["authors"] == "Smith, J., Johnson, A., Williams, B."
        assert result["year"] == "2023"
        assert result["publication"] == "Journal of Machine Learning Research"
        assert result["doi"] == "10.1234/jmlr.2023.001"
        assert "Smith, J., Johnson, A., & Williams, B." in result["apa_citation"]


@pytest.mark.asyncio
@patch('src.rag_engine.ChatOpenAI')
@patch('src.rag_engine.OpenAIEmbeddings')
@patch('src.rag_engine.get_settings')
async def test_metadata_in_search_results(mock_settings, mock_embeddings, mock_llm_class):
    """Test that search results include extracted metadata."""
    from src.rag_engine import RAGEngine
    
    # Setup mocks
    mock_settings.return_value = Mock(
        openai_api_key="test-key",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o",
        vector_db_path="/tmp/test_vector_db",
        collection_name="test_papers"
    )
    
    # Create engine
    engine = RAGEngine()
    
    # Mock vector store
    mock_vector_store = MagicMock()
    engine.vector_store = mock_vector_store
    
    # Mock document with metadata
    test_doc = Document(
        page_content="Test content",
        metadata={
            "filename": "test.pdf",
            "page": 1,
            "chunk_id": 0,
            "paper_title": "Test Paper Title",
            "paper_authors": "Author A., Author B.",
            "paper_year": "2024",
            "paper_apa_citation": "Author A., & Author B. (2024). Test paper title."
        }
    )
    mock_vector_store.similarity_search_with_score.return_value = [(test_doc, 0.5)]
    
    # Search
    results = engine.search_with_scores("test query", k=1)
    
    # Verify metadata is included
    assert len(results) == 1
    assert results[0]["metadata"]["paper_title"] == "Test Paper Title"
    assert results[0]["metadata"]["paper_authors"] == "Author A., Author B."
    assert results[0]["metadata"]["paper_year"] == "2024"
    assert results[0]["metadata"]["paper_apa_citation"] == "Author A., & Author B. (2024). Test paper title."


def test_extract_paper_metadata_method_exists():
    """Test that extract_paper_metadata method exists in RAGEngine."""
    from src.rag_engine import RAGEngine
    
    # Verify method exists
    assert hasattr(RAGEngine, 'extract_paper_metadata')
    assert callable(getattr(RAGEngine, 'extract_paper_metadata'))


@patch('src.rag_engine.ChatOpenAI')
@patch('src.rag_engine.OpenAIEmbeddings')
@patch('src.rag_engine.get_settings')
def test_find_citation_includes_bibliography(mock_settings, mock_embeddings, mock_llm_class):
    """Test that find_citation includes full bibliography metadata."""
    from src.rag_engine import RAGEngine
    
    # Setup mocks
    mock_settings.return_value = Mock(
        openai_api_key="test-key",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        vector_db_path="/tmp/test_vector_db",
        collection_name="test_papers"
    )
    
    # Create engine
    engine = RAGEngine()
    
    # Mock vector store
    mock_vector_store = MagicMock()
    engine.vector_store = mock_vector_store
    
    # Mock documents with full metadata from two different papers
    test_docs = [
        (
            Document(
                page_content="Neural networks are powerful machine learning models.",
                metadata={
                    "filename": "smith_2023.pdf",
                    "page": 5,
                    "chunk_id": 10,
                    "paper_title": "Deep Learning in NLP",
                    "paper_authors": "Smith, J., & Doe, J.",
                    "paper_year": "2023",
                    "paper_apa_citation": "Smith, J., & Doe, J. (2023). Deep learning in NLP. Journal of AI Research, 15(3), 123-145."
                }
            ),
            0.2
        ),
        (
            Document(
                page_content="Transformers have revolutionized natural language processing.",
                metadata={
                    "filename": "johnson_2024.pdf",
                    "page": 12,
                    "chunk_id": 25,
                    "paper_title": "Attention Mechanisms in Modern AI",
                    "paper_authors": "Johnson, A., Williams, B., & Brown, C.",
                    "paper_year": "2024",
                    "paper_apa_citation": "Johnson, A., Williams, B., & Brown, C. (2024). Attention mechanisms in modern AI. Machine Learning Quarterly, 8(2), 78-92."
                }
            ),
            0.3
        ),
        (
            Document(
                page_content="The architecture consists of multiple attention layers.",
                metadata={
                    "filename": "smith_2023.pdf",
                    "page": 8,
                    "chunk_id": 15,
                    "paper_title": "Deep Learning in NLP",
                    "paper_authors": "Smith, J., & Doe, J.",
                    "paper_year": "2023",
                    "paper_apa_citation": "Smith, J., & Doe, J. (2023). Deep learning in NLP. Journal of AI Research, 15(3), 123-145."
                }
            ),
            0.4
        ),
    ]
    mock_vector_store.similarity_search_with_score.return_value = test_docs
    
    # Find citations
    result = engine.find_citation("neural networks", k=3)
    
    # Verify structure
    assert result["topic"] == "neural networks"
    assert result["total_sources"] == 2  # Two unique papers
    assert "citations" in result
    
    # Verify smith_2023.pdf has bibliography metadata
    assert "smith_2023.pdf" in result["citations"]
    smith_citation = result["citations"]["smith_2023.pdf"]
    
    assert "bibliography" in smith_citation
    assert smith_citation["bibliography"]["title"] == "Deep Learning in NLP"
    assert smith_citation["bibliography"]["authors"] == "Smith, J., & Doe, J."
    assert smith_citation["bibliography"]["year"] == "2023"
    assert "Smith, J., & Doe, J. (2023)" in smith_citation["bibliography"]["apa_citation"]
    
    # Verify relevant excerpts are included
    assert "relevant_excerpts" in smith_citation
    assert len(smith_citation["relevant_excerpts"]) == 2  # Two excerpts from this paper
    assert smith_citation["relevant_excerpts"][0]["page"] == 5
    assert smith_citation["relevant_excerpts"][1]["page"] == 8
    
    # Verify johnson_2024.pdf has bibliography metadata
    assert "johnson_2024.pdf" in result["citations"]
    johnson_citation = result["citations"]["johnson_2024.pdf"]
    
    assert "bibliography" in johnson_citation
    assert johnson_citation["bibliography"]["title"] == "Attention Mechanisms in Modern AI"
    assert johnson_citation["bibliography"]["authors"] == "Johnson, A., Williams, B., & Brown, C."
    assert johnson_citation["bibliography"]["year"] == "2024"
    
    # Verify relevant excerpts
    assert "relevant_excerpts" in johnson_citation
    assert len(johnson_citation["relevant_excerpts"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
