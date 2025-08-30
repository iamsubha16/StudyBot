#!/usr/bin/env python3
"""
GraphRAG Main Application
A modular Knowledge Graph-based Retrieval Augmented Generation system
"""

import os
import sys
from typing import List, Optional
from config import GraphRAGConfig
from document_loader import DocumentLoader
from graph_manager import GraphManager
from retrieval_system import RetrievalSystem

class GraphRAGPipeline:
    """Main pipeline for GraphRAG system"""
    
    def __init__(self, config: GraphRAGConfig = None):
        """
        Initialize the GraphRAG pipeline
        
        Args:
            config: Configuration object (optional, will create default if not provided)
        """
        self.config = config or GraphRAGConfig()
        self.document_loader = None
        self.graph_manager = None
        self.retrieval_system = None
        self._initialize()
    
    def _initialize(self):
        """Initialize all components"""
        # Validate configuration
        if not self.config.validate():
            missing_fields = self.config.get_missing_fields()
            raise ValueError(f"Missing required configuration: {missing_fields}")
        
        # Initialize components
        self.document_loader = DocumentLoader(self.config)
        self.graph_manager = GraphManager(self.config)
        
        # Test database connection
        if not self.graph_manager.test_connection():
            raise ConnectionError("Failed to connect to Neo4j database")
        
        print("✅ GraphRAG Pipeline initialized successfully!")
    
    def process_documents(self, source, source_type: str = "pdf_file", clear_existing: bool = True):
        """
        Process documents and create knowledge graph
        
        Args:
            source: Document source (file, path, or query)
            source_type: Type of source ("pdf_file", "pdf_path", or "wikipedia")
            clear_existing: Whether to clear existing graph data
        """
        print(f"📄 Loading documents from {source_type}...")
        
        # Clear existing graph if requested
        if clear_existing:
            print("🗑️ Clearing existing graph data...")
            self.graph_manager.clear_graph()
        
        # Load and split documents
        document_chunks = self.document_loader.load_and_split(source, source_type)
        print(f"📑 Loaded {len(document_chunks)} document chunks")
        
        # Create knowledge graph
        print("🔗 Creating knowledge graph...")
        self.graph_manager.create_knowledge_graph(document_chunks)
        
        # Setup vector index
        print("🔍 Setting up vector index...")
        self.graph_manager.setup_vector_index()
        
        # Initialize retrieval system
        self.retrieval_system = RetrievalSystem(self.config, self.graph_manager)
        
        # Get graph statistics
        stats = self.graph_manager.get_graph_stats()
        print(f"📊 Graph created with {stats['nodes']} nodes and {stats['relationships']} relationships")
        
        return stats
    
    def ask_question(self, question: str, chat_history: List = None) -> str:
        """
        Ask a question using the GraphRAG system
        
        Args:
            question: Question to ask
            chat_history: Optional chat history
            
        Returns:
            Answer to the question
        """
        if not self.retrieval_system:
            raise RuntimeError("No documents have been processed yet. Please process documents first.")
        
        print(f"❓ Question: {question}")
        answer = self.retrieval_system.answer_question(question, chat_history)
        return answer
    
    def get_graph_stats(self) -> dict:
        """Get knowledge graph statistics"""
        if not self.graph_manager:
            return {"nodes": 0, "relationships": 0}
        return self.graph_manager.get_graph_stats()

def main():
    """Main function for command line usage"""
    print("🚀 Starting GraphRAG System")
    
    try:
        # Initialize pipeline
        pipeline = GraphRAGPipeline()
        
        # Example usage
        print("\n" + "="*50)
        print("GraphRAG System Ready!")
        print("="*50)
        
        # Interactive mode
        print("\nInteractive mode - Type 'quit' to exit")
        print("First, process some documents using pipeline.process_documents()")
        print("Then ask questions using pipeline.ask_question()")
        
        # You can add your document processing here
        # Example:
        # pipeline.process_documents("your_wikipedia_query", "wikipedia")
        # answer = pipeline.ask_question("Your question here")
        # print(f"Answer: {answer}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()