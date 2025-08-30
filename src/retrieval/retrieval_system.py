from typing import List, Tuple
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from graph_manager import GraphManager
from entity_extractor import EntityExtractor
from config import GraphRAGConfig

class RetrievalSystem:
    """Handles retrieval and question-answering using GraphRAG"""
    
    def __init__(self, config: GraphRAGConfig, graph_manager: GraphManager):
        self.config = config
        self.graph_manager = graph_manager
        self.entity_extractor = EntityExtractor(config)
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup the retrieval and QA chains"""
        # Template for condensing chat history
        condense_template = """Given the following conversation and a follow up question, 
        rephrase the follow up question to be a standalone question, in its original language.
        
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        
        self.CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)
        
        # Setup search query chain
        self._search_query = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                RunnablePassthrough.assign(
                    chat_history=lambda x: self._format_chat_history(x["chat_history"])
                )
                | self.CONDENSE_QUESTION_PROMPT
                | self.graph_manager.llm
                | StrOutputParser(),
            ),
            RunnableLambda(lambda x: x["question"]),
        )
        
        # QA template
        qa_template = """Answer the question based only on the following context:
        {context}

        Question: {question}

        Provide a comprehensive answer based on the context. If the context doesn't contain 
        enough information to answer the question completely, say so and provide what 
        information is available.

        Answer:"""
        
        self.qa_prompt = ChatPromptTemplate.from_template(qa_template)
        
        # Setup the main chain
        self.chain = (
            RunnableParallel(
                {
                    "context": self._search_query | self.retriever,
                    "question": RunnablePassthrough(),
                }
            )
            | self.qa_prompt
            | self.graph_manager.llm
            | StrOutputParser()
        )
    
    def _format_chat_history(self, chat_history: List[Tuple[str, str]]) -> List:
        """Format chat history for processing"""
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer
    
    def generate_full_text_query(self, input_text: str) -> str:
        """
        Generate full-text search query with fuzzy matching
        
        Args:
            input_text: Input text to create query from
            
        Returns:
            Formatted full-text search query
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input_text).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        if words:
            full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()
    
    def structured_retriever(self, question: str) -> str:
        """
        Retrieve structured data from the knowledge graph
        
        Args:
            question: Input question
            
        Returns:
            Structured data as string
        """
        result = ""
        entities = self.entity_extractor.extract_entities(question)
        
        for entity in entities:
            if not entity.strip():
                continue
                
            try:
                response = self.graph_manager.graph.query(
                    """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                    YIELD node,score
                    CALL {
                      WITH node
                      MATCH (node)-[r:!MENTIONS]->(neighbor)
                      RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                      UNION ALL
                      WITH node
                      MATCH (node)<-[r:!MENTIONS]-(neighbor)
                      RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                    }
                    RETURN output LIMIT 50
                    """,
                    {"query": self.generate_full_text_query(entity)},
                )
                result += "\n".join([el['output'] for el in response])
            except Exception as e:
                print(f"Error in structured retrieval for entity '{entity}': {e}")
                
        return result
    
    def retriever(self, question: str) -> str:
        """
        Combined retriever for structured and unstructured data
        
        Args:
            question: Input question
            
        Returns:
            Combined structured and unstructured data
        """
        print(f"Search query: {question}")
        
        # Get structured data from graph
        structured_data = self.structured_retriever(question)
        
        # Get unstructured data from vector search
        unstructured_data = []
        if self.graph_manager.vector_index:
            try:
                docs = self.graph_manager.vector_index.similarity_search(question)
                unstructured_data = [doc.page_content for doc in docs]
            except Exception as e:
                print(f"Error in vector search: {e}")
        
        # Combine the data
        final_data = f"""Structured data:
{structured_data}

Unstructured data:
{"#Document ".join(unstructured_data)}"""
        
        return final_data
    
    def answer_question(self, question: str, chat_history: List[Tuple[str, str]] = None) -> str:
        """
        Answer a question using the GraphRAG system
        
        Args:
            question: Input question
            chat_history: Optional chat history for context
            
        Returns:
            Answer to the question
        """
        try:
            input_data = {"question": question}
            if chat_history:
                input_data["chat_history"] = chat_history
                
            answer = self.chain.invoke(input_data)
            return answer
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def get_context(self, question: str) -> str:
        """
        Get the context that would be used for answering a question
        
        Args:
            question: Input question
            
        Returns:
            Context string
        """
        return self.retriever(question)