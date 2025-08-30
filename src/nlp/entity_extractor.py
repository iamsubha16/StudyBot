from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from config import GraphRAGConfig

class Entities(BaseModel):
    """Pydantic model for entity extraction"""
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

class EntityExtractor:
    """Handles entity extraction from queries"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.llm = ChatGroq(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=None,
            groq_api_key=config.groq_api_key,
            timeout=60
        )
        self._setup_extraction_chain()
    
    def _setup_extraction_chain(self):
        """Setup the entity extraction chain"""
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are extracting organization and person entities from the text. "
                "Focus on named entities like people, companies, organizations, locations, "
                "and other proper nouns that might be relevant for knowledge graph queries."
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ])
        
        self.entity_chain = prompt | self.llm.with_structured_output(Entities)
    
    def extract_entities(self, question: str) -> List[str]:
        """
        Extract entities from a question
        
        Args:
            question: Input question to extract entities from
            
        Returns:
            List of extracted entity names
        """
        try:
            entities = self.entity_chain.invoke({"question": question})
            return entities.names
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []
    
    def extract_and_clean_entities(self, question: str) -> List[str]:
        """
        Extract entities and clean them
        
        Args:
            question: Input question to extract entities from
            
        Returns:
            List of cleaned entity names
        """
        entities = self.extract_entities(question)
        # Clean entities - remove empty strings and duplicates
        cleaned_entities = list(set([entity.strip() for entity in entities if entity.strip()]))
        return cleaned_entities