import os
import tempfile
from typing import List, Union
from langchain.document_loaders import PyPDFLoader, WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import GraphRAGConfig

class DocumentLoader:
    """Handles loading and processing of documents from various sources"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    def load_from_pdf_file(self, pdf_file) -> List[Document]:
        """
        Load documents from an uploaded PDF file (Streamlit file uploader)
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            List of Document objects
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        
        try:
            # Load the PDF using Langchain's PyPDFLoader
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            return documents
        finally:
            # Clean up the temporary file
            os.unlink(tmp_file_path)
    
    def load_from_pdf_path(self, pdf_path: str) -> List[Document]:
        """
        Load documents from a PDF file path
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    
    def load_from_wikipedia(self, query: str, max_docs: int = 4) -> List[Document]:
        """
        Load documents from Wikipedia
        
        Args:
            query: Wikipedia search query
            max_docs: Maximum number of documents to load
            
        Returns:
            List of Document objects
        """
        loader = WikipediaLoader(query=query)
        documents = loader.load()
        return documents[:max_docs]
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        return self.text_splitter.split_documents(documents)
    
    def load_and_split(self, source: Union[str, object], source_type: str = "pdf") -> List[Document]:
        """
        Load and split documents in one step
        
        Args:
            source: Document source (file path, uploaded file, or query)
            source_type: Type of source ("pdf_file", "pdf_path", or "wikipedia")
            
        Returns:
            List of document chunks
        """
        if source_type == "pdf_file":
            documents = self.load_from_pdf_file(source)
        elif source_type == "pdf_path":
            documents = self.load_from_pdf_path(source)
        elif source_type == "wikipedia":
            documents = self.load_from_wikipedia(source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        return self.split_documents(documents)