import os
import tempfile
from typing import List, Union, Optional
from enum import Enum
import wikipedia

from langchain_community.document_loaders import PyPDFLoader, WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.schema import Document

from src.config.config import GraphRAGConfig
from logger import logger  # Custom logger module

class SourceType(str, Enum):
    """Enum for different document source types"""
    PDF_FILE = "pdf_file"
    PDF_PATH = "pdf_path"
    WIKIPEDIA = "wikipedia"

class DocumentLoader:
    """Handles loading and processing of documents from various sources"""

    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self._validate_config()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        logger.info(f"DocumentLoader initialized with chunk_size={config.chunk_size}, "
                    f"chunk_overlap={config.chunk_overlap}")

    def _validate_config(self):
        """Validate configuration values"""
        if self.config.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.config.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def load_from_pdf_file(self, pdf_file) -> List[Document]:
        """
        Load documents from an uploaded PDF file (Streamlit file uploader)

        Returns:
            List of Document objects
        """
        tmp_file_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name

            logger.info(f"Temporary PDF file created at {tmp_file_path}")

            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} documents from PDF file")
            return documents

        except Exception as e:
            logger.error(f"Failed to load PDF file: {e}", exc_info=True)
            return []
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                logger.debug(f"Temporary file {tmp_file_path} deleted")

    def load_from_pdf_path(self, pdf_path: str) -> List[Document]:
        """
        Load documents from a PDF file path

        Returns:
            List of Document objects
        """
        try:
            logger.info(f"Loading PDF from path: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} documents from {pdf_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF from {pdf_path}: {e}", exc_info=True)
            return []

    def load_from_wikipedia(self, query: str, max_docs: int = 4) -> List[Document]:
        """
        Load documents from Wikipedia

        Returns:
            List of Document objects
        """
        try:
            logger.info(f"Fetching Wikipedia documents for query: '{query}'")
            loader = WikipediaLoader(query=query)
            documents = loader.load()
            sliced_docs = documents[:max_docs]
            logger.info(f"Fetched {len(sliced_docs)} Wikipedia documents for query '{query}'")
            return sliced_docs
        except Exception as e:
            logger.error(f"Error fetching Wikipedia documents for query '{query}': {e}", exc_info=True)
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Returns:
            List of document chunks
        """
        try:
            logger.info(f"Splitting {len(documents)} documents into chunks")
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {e}", exc_info=True)
            return []

    def load_and_split(self, source: Union[str, object], source_type: Union[str, SourceType] = SourceType.PDF_PATH) -> List[Document]:
        """
        Load and split documents in one step

        Returns:
            List of document chunks
        """
        try:
            logger.info(f"Loading and splitting documents from source_type={source_type}")

            if source_type == SourceType.PDF_FILE:
                documents = self.load_from_pdf_file(source)
            elif source_type == SourceType.PDF_PATH:
                documents = self.load_from_pdf_path(source)
            elif source_type == SourceType.WIKIPEDIA:
                documents = self.load_from_wikipedia(source)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")

            return self.split_documents(documents)

        except ValueError as ve:
            logger.error(f"Invalid source type provided: {ve}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Error in load_and_split: {e}", exc_info=True)
            return []
