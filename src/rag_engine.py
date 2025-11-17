"""RAG engine for semantic search over research papers using LangChain and ChromaDB."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

from src.config import get_settings
from src.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


class PaperMetadataLevel(str, Enum):
    """Enum for controlling metadata detail level in list_papers."""
    FILENAME_ONLY = "filename_only"
    WITH_AUTHORS = "with_authors"
    WITH_BIBLIOGRAPHY = "with_bibliography"
    FULL = "full"


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
        
    async def initialize_vector_store(
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
            
            # Delete existing collection if force rebuild
            if force_rebuild and Path(persist_directory).exists():
                logger.info("Deleting existing collection for rebuild...")
                try:
                    temp_store = Chroma(
                        collection_name=self.settings.collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=persist_directory,
                    )
                    temp_store.delete_collection()
                    logger.info("Existing collection deleted")
                except Exception as e:
                    logger.warning(f"Could not delete existing collection: {e}")
            
            # Process PDFs individually to extract metadata per paper
            pdf_files = list(self.settings.papers_directory.glob("**/*.pdf"))
            
            if not pdf_files:
                raise ValueError("No PDF files found to index")
            
            logger.info(f"Found {len(pdf_files)} PDF files. Extracting metadata for each...")
            
            all_documents = []
            
            for pdf_path in pdf_files:
                try:
                    # Extract metadata from first 1-2 pages
                    logger.info(f"Extracting metadata from {pdf_path.name}...")
                    paper_metadata = await self.extract_paper_metadata(pdf_path)
                    
                    # Process the PDF to get chunks
                    logger.info(f"Processing {pdf_path.name}...")
                    chunks = await self.pdf_processor.process_pdf(pdf_path)
                    
                    # Attach extracted metadata to all chunks from this paper
                    for chunk in chunks:
                        chunk.metadata.update({
                            "paper_title": paper_metadata.get("title", "Not found"),
                            "paper_authors": paper_metadata.get("authors", "Not found"),
                            "paper_year": paper_metadata.get("year", "Not found"),
                            "paper_publication": paper_metadata.get("publication", "Not found"),
                            "paper_doi": paper_metadata.get("doi", "Not found"),
                            "paper_apa_citation": paper_metadata.get("apa_citation", "Not found"),
                        })
                    
                    all_documents.extend(chunks)
                    logger.info(f"Processed {pdf_path.name}: {len(chunks)} chunks with metadata")
                    
                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {e}")
                    continue
            
            if not all_documents:
                raise ValueError("No documents found to index")
            
            # Create unique IDs for each document to prevent duplicates
            doc_ids = [
                f"{doc.metadata.get('filename', 'unknown')}_{doc.metadata.get('page', 0)}_{doc.metadata.get('chunk_id', i)}"
                for i, doc in enumerate(all_documents)
            ]
            
            logger.info(f"Creating vector store with {len(all_documents)} chunks (with extracted metadata)")
            self.vector_store = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embeddings,
                collection_name=self.settings.collection_name,
                persist_directory=persist_directory,
                ids=doc_ids,
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
                    "paper_title": doc.metadata.get("paper_title", "Not found"),
                    "paper_authors": doc.metadata.get("paper_authors", "Not found"),
                    "paper_year": doc.metadata.get("paper_year", "Not found"),
                    "paper_apa_citation": doc.metadata.get("paper_apa_citation", "Not found"),
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
            List of dictionaries containing search results with scores (sorted by relevance, deduplicated)
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
        
        # Perform similarity search with scores - get more to account for deduplication
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k * 3)
        
        # Sort by score (lower is better for distance metrics)
        docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1])
        
        # Deduplicate based on content and metadata
        seen = set()
        unique_docs = []
        for doc, score in docs_with_scores:
            # Create unique key from content + filename + page + chunk_id
            doc_key = (
                doc.page_content,
                doc.metadata.get("filename", "Unknown"),
                doc.metadata.get("page", "Unknown"),
                doc.metadata.get("chunk_id", 0)
            )
            if doc_key not in seen:
                seen.add(doc_key)
                unique_docs.append((doc, score))
                if len(unique_docs) >= k:
                    break
        
        # Format results
        results = []
        for doc, score in unique_docs:
            results.append({
                "content": doc.page_content,
                "score": float(score),
                "metadata": {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                    "paper_title": doc.metadata.get("paper_title", "Not found"),
                    "paper_authors": doc.metadata.get("paper_authors", "Not found"),
                    "paper_year": doc.metadata.get("paper_year", "Not found"),
                    "paper_apa_citation": doc.metadata.get("paper_apa_citation", "Not found"),
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
            # Perform similarity search with metadata filter - get more for deduplication
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query,
                k=k * 5,  # Get more results to account for deduplication
                filter={"filename": filename}
            )
            
            # Sort by score (lower is better for distance metrics)
            docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1])
            
        except Exception:
            # Fallback: manual filtering if the vector store doesn't support metadata filtering
            all_docs = self.vector_store.similarity_search_with_score(query, k=k * 10)
            docs_with_scores = [
                (doc, score) for doc, score in all_docs 
                if doc.metadata.get("filename") == filename
            ]
            # Sort by score
            docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1])
        
        if not docs_with_scores:
            return []
        
        # Deduplicate based on content and metadata
        seen = set()
        unique_docs = []
        for doc, score in docs_with_scores:
            doc_key = (
                doc.page_content,
                doc.metadata.get("page", "Unknown"),
                doc.metadata.get("chunk_id", 0)
            )
            if doc_key not in seen:
                seen.add(doc_key)
                unique_docs.append((doc, score))
                if len(unique_docs) >= k:
                    break
        
        # Format results
        results = []
        for doc, score in unique_docs:
            results.append({
                "content": doc.page_content,
                "score": float(score),
                "metadata": {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "source": doc.metadata.get("source", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0),
                    "paper_title": doc.metadata.get("paper_title", "Not found"),
                    "paper_authors": doc.metadata.get("paper_authors", "Not found"),
                    "paper_year": doc.metadata.get("paper_year", "Not found"),
                    "paper_apa_citation": doc.metadata.get("paper_apa_citation", "Not found"),
                },
            })
        
        return results
    
    def find_citation(
        self, topic: str, k: int = 3
    ) -> Dict[str, Any]:
        """
        Find relevant citations for a given topic with full bibliography metadata.

        Args:
            topic: Topic or concept to find citations for
            k: Number of citations to find

        Returns:
            Dictionary with citations, context, and full bibliography metadata
        """
        results = self.search_with_scores(topic, k=k)
        
        # Group by source document and include metadata
        citations_by_paper = {}
        for result in results:
            filename = result["metadata"]["filename"]
            metadata = result["metadata"]
            
            if filename not in citations_by_paper:
                # Initialize with paper metadata
                citations_by_paper[filename] = {
                    "bibliography": {
                        "title": metadata.get("paper_title", "Not found"),
                        "authors": metadata.get("paper_authors", "Not found"),
                        "year": metadata.get("paper_year", "Not found"),
                        "apa_citation": metadata.get("paper_apa_citation", "Not found"),
                    },
                    "relevant_excerpts": []
                }
            
            # Add excerpt to this paper's excerpts
            citations_by_paper[filename]["relevant_excerpts"].append({
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id", 0),
                "excerpt": result["content"],
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
            "title": metadata.get("paper_title", "Not found"),
            "authors": metadata.get("paper_authors", "Not found"),
            "year": metadata.get("paper_year", "Not found"),
            "publication": metadata.get("paper_publication", "Not found"),
            "doi": metadata.get("paper_doi", "Not found"),
            "apa_citation": metadata.get("paper_apa_citation", "Not found"),
        }
    
    def extract_methodology(self, filename: str) -> Dict[str, Any]:
        """
        Extract methodology details from a research paper.
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            Dictionary with extracted methodology information
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
        
        # Search for methodology-related content in the paper
        methodology_query = "methodology methods approach experimental setup dataset model architecture evaluation metrics"
        
        # Get relevant sections from the paper
        docs_with_scores = self.vector_store.similarity_search_with_score(
            methodology_query,
            k=10,
            filter={"filename": filename}
        )
        
        if not docs_with_scores:
            return {
                "filename": filename,
                "found": False,
                "message": f"No methodology information found for {filename}",
            }
        
        # Sort by relevance (lower score is better)
        docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1])
        
        # Combine the most relevant chunks
        context = "\n\n".join([doc.page_content for doc, _ in docs_with_scores[:5]])
        
        # Use LLM to extract structured methodology information
        extraction_prompt = f"""Analyze the following research paper excerpts and extract methodology details.

Paper excerpts:
{context}

Extract and structure the following information if present:
1. Research approach/design
2. Dataset(s) used (name, size, characteristics)
3. Model/algorithm/technique used
4. Evaluation metrics
5. Experimental setup
6. Baseline comparisons
7. Implementation details

Provide a clear, structured summary of the methodology. If specific information is not found, indicate "Not mentioned".
"""
        
        response = self.llm.invoke(extraction_prompt)
        
        return {
            "filename": filename,
            "found": True,
            "methodology": response.content,
            "num_chunks_analyzed": len(docs_with_scores),
        }
    
    def summarize_paper(
        self, 
        filename: str, 
        custom_prompt: Optional[str] = None,
        focus: str = "general"
    ) -> Dict[str, Any]:
        """
        Generate a summary of a research paper with optional custom focus.
        
        Args:
            filename: Name of the PDF file
            custom_prompt: Optional custom prompt to guide the summary
            focus: Pre-defined focus area if custom_prompt not provided.
                   Options: "general", "key_findings", "methodology", "limitations", "contributions"
            
        Returns:
            Dictionary with paper summary
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
        
        # Get all chunks from the paper
        all_docs = self.vector_store.get(
            where={"filename": filename}
        )
        
        if not all_docs or not all_docs.get("documents"):
            return {
                "filename": filename,
                "found": False,
                "message": f"No content found for {filename}",
            }
        
        # Take representative chunks from beginning, middle, and end
        num_docs = len(all_docs["documents"])
        sample_indices = [
            0,  # Beginning
            num_docs // 4,
            num_docs // 2,  # Middle
            3 * num_docs // 4,
            num_docs - 1  # End
        ]
        
        sample_content = "\n\n".join([
            all_docs["documents"][i] 
            for i in sample_indices 
            if i < num_docs
        ])
        
        # Build prompt based on focus or custom prompt
        if custom_prompt:
            prompt = f"""Analyze the following research paper excerpts and provide a summary based on this request:

{custom_prompt}

Paper excerpts:
{sample_content}
"""
        else:
            focus_prompts = {
                "general": "Provide a comprehensive summary covering: main topic, research questions, methodology, key findings, and conclusions.",
                "key_findings": "Focus on the main findings, results, and discoveries. Include specific numbers, metrics, or outcomes where mentioned.",
                "methodology": "Focus on the research methodology, experimental design, datasets, models/algorithms used, and evaluation approach.",
                "limitations": "Identify and summarize the limitations, constraints, threats to validity, and future work mentioned by the authors.",
                "contributions": "Highlight the main contributions, novelty, and significance of this research. What advances does it provide to the field?"
            }
            
            focus_instruction = focus_prompts.get(focus, focus_prompts["general"])
            
            prompt = f"""Analyze the following research paper excerpts and provide a summary.

Focus: {focus_instruction}

Paper excerpts:
{sample_content}

Provide a clear, well-structured summary (200-400 words).
"""
        
        response = self.llm.invoke(prompt)
        
        return {
            "filename": filename,
            "found": True,
            "focus": custom_prompt if custom_prompt else focus,
            "summary": response.content,
            "chunks_analyzed": len(sample_indices),
        }
    
    def extract_bibliography(self, filename: str) -> Dict[str, Any]:
        """
        Extract citations/references from a research paper and format in APA style.
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            Dictionary with extracted bibliography in APA format
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
        
        # Search for reference/bibliography sections
        references_query = "references bibliography citations works cited"
        
        # Get all chunks from the paper
        all_docs = self.vector_store.get(
            where={"filename": filename}
        )
        
        if not all_docs or not all_docs.get("documents"):
            return {
                "filename": filename,
                "found": False,
                "message": f"No content found for {filename}",
            }
        
        # Look at the last 20% of the document (where references typically are)
        num_docs = len(all_docs["documents"])
        start_idx = max(0, int(num_docs * 0.8))
        
        reference_content = "\n\n".join(all_docs["documents"][start_idx:])
        
        # Use LLM to extract and format references
        extraction_prompt = f"""Extract the bibliography/references section from the following paper excerpts.
Format each citation in APA style. If the citations are already in APA format, preserve them.
If they're in another format, convert them to APA format.

Paper excerpts (typically from the references section):
{reference_content[:6000]}  

Provide the bibliography as a numbered list in proper APA format. Include as many references as you can identify.
If no clear references section is found, indicate this.
"""
        
        response = self.llm.invoke(extraction_prompt)
        
        return {
            "filename": filename,
            "found": True,
            "bibliography": response.content,
            "format": "APA",
            "note": "Extracted from the paper's reference section",
        }
    
    async def extract_paper_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract paper metadata (authors, year, title, APA citation) from first 1-2 pages using LLM.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted metadata including APA citation
        """
        try:
            # Load first 1-2 pages
            first_pages_text = await self.pdf_processor.load_first_pages(pdf_path, num_pages=2)
            
            # Use LLM to extract metadata
            extraction_prompt = f"""Analyze the first pages of this research paper and extract the following metadata:

Paper content:
{first_pages_text[:4000]}

Extract the following information:
1. **Title**: Full paper title
2. **Authors**: All author names (comma-separated)
3. **Year**: Publication year
4. **Journal/Conference**: Where it was published
5. **DOI**: Digital Object Identifier (if present)
6. **APA Citation**: Format the complete citation in APA 7th edition style

Provide the response in this **exact format**:
Title: [extracted title]
Authors: [author1, author2, author3, ...]
Year: [year]
Publication: [journal or conference name]
DOI: [doi if found, otherwise "Not found"]
APA Citation: [complete APA format citation]

If any field cannot be determined from the text, use "Not found" for that field.
Be precise and accurate with the information you extract."""

            response = self.llm.invoke(extraction_prompt)
            content = response.content
            
            # Parse the response
            metadata = {
                "filename": pdf_path.name,
                "title": "Not found",
                "authors": "Not found",
                "year": "Not found",
                "publication": "Not found",
                "doi": "Not found",
                "apa_citation": "Not found",
            }
            
            # Extract fields from response
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("Title:"):
                    metadata["title"] = line.replace("Title:", "").strip()
                elif line.startswith("Authors:"):
                    metadata["authors"] = line.replace("Authors:", "").strip()
                elif line.startswith("Year:"):
                    metadata["year"] = line.replace("Year:", "").strip()
                elif line.startswith("Publication:"):
                    metadata["publication"] = line.replace("Publication:", "").strip()
                elif line.startswith("DOI:"):
                    metadata["doi"] = line.replace("DOI:", "").strip()
                elif line.startswith("APA Citation:"):
                    # APA citation might span multiple lines
                    metadata["apa_citation"] = line.replace("APA Citation:", "").strip()
            
            logger.info(f"Extracted metadata for {pdf_path.name}: {metadata['title'][:50]}...")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {
                "filename": pdf_path.name,
                "title": "Error extracting metadata",
                "authors": "Unknown",
                "year": "Unknown",
                "publication": "Unknown",
                "doi": "Not found",
                "apa_citation": f"Unknown. ({pdf_path.name})",
                "error": str(e),
            }
    
    def list_papers(
        self, 
        metadata_level: PaperMetadataLevel = PaperMetadataLevel.FILENAME_ONLY
    ) -> List[Dict[str, Any]] | List[str]:
        """
        List all papers in the vector store with optional metadata.

        Args:
            metadata_level: Level of metadata to include in results
                - FILENAME_ONLY: Just the filename (default, backward compatible)
                - WITH_AUTHORS: Filename, title, and authors
                - WITH_BIBLIOGRAPHY: Filename, title, and APA citation
                - FULL: All available metadata

        Returns:
            List of paper filenames or dictionaries with metadata (depending on metadata_level)
        """
        if self.vector_store is None:
            raise RuntimeError("Vector store not initialized. Call initialize_vector_store() first.")
        
        # Get all documents
        collection = self.vector_store._collection
        all_metadata = collection.get()
        
        # Build paper information from unique filenames
        papers_dict = {}
        if all_metadata.get("metadatas"):
            for metadata in all_metadata["metadatas"]:
                filename = metadata.get("filename")
                if filename and filename not in papers_dict:
                    papers_dict[filename] = metadata
        
        # Return based on metadata level
        if metadata_level == PaperMetadataLevel.FILENAME_ONLY:
            return sorted(list(papers_dict.keys()))
        
        # Build detailed responses
        papers_list = []
        for filename in sorted(papers_dict.keys()):
            metadata = papers_dict[filename]
            
            if metadata_level == PaperMetadataLevel.WITH_AUTHORS:
                papers_list.append({
                    "filename": filename,
                    "title": metadata.get("paper_title", "Not found"),
                    "authors": metadata.get("paper_authors", "Not found"),
                })
            elif metadata_level == PaperMetadataLevel.WITH_BIBLIOGRAPHY:
                papers_list.append({
                    "filename": filename,
                    "title": metadata.get("paper_title", "Not found"),
                    "apa_citation": metadata.get("paper_apa_citation", "Not found"),
                })
            elif metadata_level == PaperMetadataLevel.FULL:
                papers_list.append({
                    "filename": filename,
                    "title": metadata.get("paper_title", "Not found"),
                    "authors": metadata.get("paper_authors", "Not found"),
                    "year": metadata.get("paper_year", "Not found"),
                    "publication": metadata.get("paper_publication", "Not found"),
                    "doi": metadata.get("paper_doi", "Not found"),
                    "apa_citation": metadata.get("paper_apa_citation", "Not found"),
                })
        
        return papers_list
