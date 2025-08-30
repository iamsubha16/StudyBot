import os
from dataclasses import dataclass
from typing import Optional
from logger import logger  # Import the logger
from dotenv import load_dotenv 

# Load environment variables from .env file
load_dotenv()

@dataclass
class GraphRAGConfig:
    """Configuration class for GraphRAG system"""

    # Model Configuration
    # model_name: str = "gpt-4o"
    temperature: float = 0.3
    tokens_per_minute: int = 900

    # Text Processing Configuration
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Azure OpenAI Configuration
    azure_openai_endpoint: Optional[str] = None
    azure_openai_key: Optional[str] = None
    azure_openai_deployment: Optional[str] = None
    azure_api_version: Optional[str] = None

    # Neo4j Configuration
    neo4j_uri: Optional[str] = None
    neo4j_username: Optional[str] = None
    neo4j_password: Optional[str] = None

    # Hugging Face Token
    hf_token: Optional[str] = None

    # Groq API Configuration
    groq_api_key: Optional[str] = None

    def __post_init__(self):
        """Load environment variables if not provided"""
        # Azure
        self.azure_openai_endpoint = self.azure_openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_key = self.azure_openai_key or os.getenv("AZURE_OPENAI_KEY")
        self.azure_openai_deployment = self.azure_openai_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.azure_api_version = self.azure_api_version or os.getenv("AZURE_API_VERSION")

        # Neo4j
        self.neo4j_uri = self.neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_username = self.neo4j_username or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = self.neo4j_password or os.getenv("NEO4J_PASSWORD")

        # Hugging Face
        self.hf_token = self.hf_token or os.getenv("HF_TOKEN")

        # Groq API
        self.groq_api_key = self.groq_api_key or os.getenv("GROQ_API_KEY")

        logger.info("Configuration initialized.")

    def validate(self) -> bool:
        """Validate that all required configuration is present"""
        required_fields = [
            self.azure_openai_endpoint,
            self.azure_openai_key,
            self.azure_openai_deployment,
            self.azure_api_version,
            self.neo4j_uri,
            self.neo4j_username,
            self.neo4j_password,
            self.hf_token,
            self.groq_api_key
        ]
        is_valid = all(field is not None for field in required_fields)
        if is_valid:
            logger.info("All required configurations are present.")
        else:
            logger.warning(f"Missing fields: {self.get_missing_fields()}")
        return is_valid

    def get_missing_fields(self) -> list:
        """Get list of missing required configuration fields"""
        missing = []
        if not self.azure_openai_endpoint:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if not self.azure_openai_key:
            missing.append("AZURE_OPENAI_KEY")
        if not self.azure_openai_deployment:
            missing.append("AZURE_OPENAI_DEPLOYMENT")
        if not self.azure_api_version:
            missing.append("AZURE_API_VERSION")
        if not self.neo4j_uri:
            missing.append("NEO4J_URI")
        if not self.neo4j_username:
            missing.append("NEO4J_USERNAME")
        if not self.neo4j_password:
            missing.append("NEO4J_PASSWORD")
        if not self.hf_token:
            missing.append("HF_TOKEN")
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        
        return missing