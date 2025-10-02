"""PDF processing module for extracting and chunking research papers using LangChain."""

import asyncio
import logging
import warnings
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from src.config import get_settings

# Suppress pypdf warnings about malformed PDF metadata
warnings.filterwarnings("ignore", category=UserWarning, module="pypdf")

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process PDF files using LangChain's PyPDFLoader and extract text with metadata."""

    def __init__(self):
        """Initialize PDF processor with settings."""
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def load_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Load a PDF file using LangChain's PyPDFLoader.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of Document objects, one per page
        """
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = []
            
            # Use async lazy loading
            async for page in loader.alazy_load():
                # Enhance metadata with filename
                page.metadata["filename"] = pdf_path.name
                page.metadata["file_path"] = str(pdf_path)
                pages.append(page)
            
            logger.info(f"Loaded {len(pages)} pages from {pdf_path.name}")
            return pages

        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            raise

    def load_pdf_sync(self, pdf_path: Path) -> List[Document]:
        """
        Synchronous wrapper for loading a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of Document objects, one per page
        """
        return asyncio.run(self.load_pdf(pdf_path))

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of Document objects to chunk

        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        return chunks

    async def process_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Load and chunk a single PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of chunked Document objects
        """
        pages = await self.load_pdf(pdf_path)
        chunks = self.chunk_documents(pages)
        
        logger.info(
            f"Processed {pdf_path.name}: {len(pages)} pages → {len(chunks)} chunks"
        )
        return chunks

    def process_pdf_sync(self, pdf_path: Path) -> List[Document]:
        """
        Synchronous wrapper for processing a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of chunked Document objects
        """
        return asyncio.run(self.process_pdf(pdf_path))

    async def process_directory(self, directory: Path = None) -> List[Document]:
        """
        Process all PDF files in a directory asynchronously.

        Args:
            directory: Directory containing PDFs (uses settings default if None)

        Returns:
            List of all chunked documents from all PDFs
        """
        if directory is None:
            directory = self.settings.papers_directory

        pdf_files = list(directory.glob("**/*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        all_chunks = []
        for pdf_path in pdf_files:
            try:
                chunks = await self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                continue

        logger.info(
            f"Processed {len(pdf_files)} PDFs into {len(all_chunks)} total chunks"
        )
        return all_chunks

    def process_directory_sync(self, directory: Path = None) -> List[Document]:
        """
        Synchronous wrapper for processing a directory of PDFs.

        Args:
            directory: Directory containing PDFs (uses settings default if None)

        Returns:
            List of all chunked documents from all PDFs
        """
        return asyncio.run(self.process_directory(directory))
