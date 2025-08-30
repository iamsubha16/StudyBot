from typing import List, Optional
from langchain.schema import Document
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase
from config import GraphRAGConfig

class GraphManager:
    """Manages knowledge graph creation and operations"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.llm = None
        self.graph = None
        self.vector_index = None
        self.embeddings = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LLM, graph, and embeddings"""
        # Initialize LLM
        self.llm = ChatGroq(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=None,
            groq_api_key=self.config.groq_api_key,
            timeout=60
        )
        
        # Initialize Neo4j graph
        self.graph = Neo4jGraph(
            url=self.config.neo4j_uri,
            username=self.config.neo4j_username,
            password=self.config.neo4j_password
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
    
    def create_knowledge_graph(self, document_chunks: List[Document]) -> None:
        """
        Create knowledge graph from document chunks
        
        Args:
            document_chunks: List of document chunks to process
        """
        # Initialize the Graph Transformer with LLM
        llm_transformer = LLMGraphTransformer(llm=self.llm)
        
        # Convert documents to graph documents
        graph_documents = llm_transformer.convert_to_graph_documents(document_chunks)
        
        # Add graph documents to Neo4j
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,  # Ensures nodes have labels
            include_source=True    # Keeps original document for traceability
        )
        
        # Create full-text index for entities
        self.graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
    
    def setup_vector_index(self) -> None:
        """Setup vector index for hybrid search"""
        self.vector_index = Neo4jVector.from_existing_graph(
            self.embeddings,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
    
    def clear_graph(self) -> None:
        """Clear all data from the graph"""
        self.graph.query("MATCH (n) DETACH DELETE n")
    
    def get_graph_stats(self) -> dict:
        """Get basic statistics about the graph"""
        node_count = self.graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        relationship_count = self.graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
        
        return {
            "nodes": node_count,
            "relationships": relationship_count
        }
    
    def visualize_graph(self, cypher_query: str = None) -> str:
        """
        Get graph data for visualization
        
        Args:
            cypher_query: Custom Cypher query for visualization
            
        Returns:
            Cypher query results as string
        """
        if cypher_query is None:
            cypher_query = "MATCH (s)-[r:!MENTIONS]->(t) RETURN s,r,t LIMIT 50"
        
        try:
            result = self.graph.query(cypher_query)
            return str(result)
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def test_connection(self) -> bool:
        """Test connection to Neo4j database"""
        try:
            self.graph.query("RETURN 1")
            return True
        except Exception:
            return False