import streamlit as st
import os
from io import BytesIO
import tempfile
from config import GraphRAGConfig
from main import GraphRAGPipeline

# Configure page
st.set_page_config(
    page_title="GraphRAG System",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'graph_stats' not in st.session_state:
        st.session_state.graph_stats = {"nodes": 0, "relationships": 0}

def setup_sidebar():
    """Setup sidebar with configuration"""
    st.sidebar.title("🔗 GraphRAG Configuration")
    
    with st.sidebar.expander("📋 System Requirements", expanded=False):
        st.write("""
        **Required Environment Variables:**
        - `NEO4J_URI`: Neo4j database URI
        - `NEO4J_USERNAME`: Neo4j username
        - `NEO4J_PASSWORD`: Neo4j password
        - `GROQ_API_KEY`: Groq API key for LLM
        - `HF_TOKEN`: HuggingFace token (optional)
        """)
    
    # Configuration inputs
    st.sidebar.subheader("🔧 Configuration")
    
    neo4j_uri = st.sidebar.text_input("Neo4j URI", value=os.getenv("NEO4J_URI", ""))
    neo4j_username = st.sidebar.text_input("Neo4j Username", value=os.getenv("NEO4J_USERNAME", ""))
    neo4j_password = st.sidebar.text_input("Neo4j Password", type="password", value=os.getenv("NEO4J_PASSWORD", ""))
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
        model_name = st.selectbox(
            "LLM Model",
            ["deepseek-r1-distill-llama-70b", "llama3-70b-8192", "mixtral-8x7b-32768"],
            index=0
        )
        chunk_size = st.slider("Chunk Size", 256, 1024, 512)
        chunk_overlap = st.slider("Chunk Overlap", 0, 100, 24)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
    
    # Create configuration
    config = GraphRAGConfig(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        groq_api_key=groq_api_key,
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        temperature=temperature
    )
    
    # Validate configuration
    if st.sidebar.button("🔄 Initialize System"):
        with st.sidebar:
            with st.spinner("Initializing system..."):
                try:
                    if not config.validate():
                        missing_fields = config.get_missing_fields()
                        st.error(f"Missing required fields: {', '.join(missing_fields)}")
                    else:
                        st.session_state.pipeline = GraphRAGPipeline(config)
                        st.success("✅ System initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize: {str(e)}")
    
    # Display system status
    if st.session_state.pipeline:
        st.sidebar.success("🟢 System Ready")
        if st.session_state.documents_processed:
            st.sidebar.info(f"📊 Graph: {st.session_state.graph_stats['nodes']} nodes, {st.session_state.graph_stats['relationships']} relationships")
    else:
        st.sidebar.warning("🟡 System not initialized")

def main_interface():
    """Main application interface"""
    st.title("🔗 GraphRAG: Knowledge Graph-based RAG System")
    st.markdown("Upload documents or provide Wikipedia queries to build a knowledge graph and ask questions!")
    
    if not st.session_state.pipeline:
        st.warning("⚠️ Please initialize the system using the sidebar configuration first.")
        return
    
    # Document processing section
    st.header("📄 Document Processing")
    
    # Tabs for different input methods
    tab1, tab2 = st.tabs(["📎 PDF Upload", "📖 Wikipedia"])
    
    with tab1:
        st.subheader("Upload PDF Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            st.success(f"📄 File uploaded: {uploaded_file.name}")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                clear_existing = st.checkbox("Clear existing graph data", value=True)
            with col2:
                if st.button("🔄 Process PDF", key="process_pdf"):
                    process_document(uploaded_file, "pdf_file", clear_existing)
    
    with tab2:
        st.subheader("Wikipedia Query")
        wikipedia_query = st.text_input("Enter Wikipedia search query", placeholder="e.g., Artificial Intelligence")
        
        if wikipedia_query:
            col1, col2 = st.columns([1, 1])
            with col1:
                clear_existing = st.checkbox("Clear existing graph data", value=True, key="wiki_clear")
            with col2:
                if st.button("🔄 Process Wikipedia", key="process_wiki"):
                    process_document(wikipedia_query, "wikipedia", clear_existing)
    
    # Question answering section
    if st.session_state.documents_processed:
        st.header("💬 Ask Questions")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.expander(f"Q{i+1}: {question[:50]}..."):
                    st.write(f"**Question:** {question}")
                    st.write(f"**Answer:** {answer}")
        
        # New question input
        question = st.text_input("Ask a question about your documents:", placeholder="Enter your question here...")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🤔 Ask Question", disabled=not question):
                ask_question(question)
        with col2:
            if st.button("🔍 Show Context", disabled=not question):
                show_context(question)
        with col3:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

def process_document(source, source_type, clear_existing):
    """Process documents and update session state"""
    with st.spinner("Processing documents..."):
        try:
            stats = st.session_state.pipeline.process_documents(
                source, source_type, clear_existing
            )
            st.session_state.documents_processed = True
            st.session_state.graph_stats = stats
            st.success(f"✅ Documents processed! Created graph with {stats['nodes']} nodes and {stats['relationships']} relationships.")
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")

def ask_question(question):
    """Ask a question and update chat history"""
    with st.spinner("Generating answer..."):
        try:
            # Convert chat history to the format expected by the pipeline
            chat_history = [(q, a) for q, a in st.session_state.chat_history[-5:]]  # Keep last 5 exchanges
            
            answer = st.session_state.pipeline.ask_question(question, chat_history)
            
            # Add to chat history
            st.session_state.chat_history.append((question, answer))
            
            # Display the answer
            st.subheader("Answer:")
            st.write(answer)
            
        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")

def show_context(question):
    """Show the context that would be used for answering"""
    with st.spinner("Retrieving context..."):
        try:
            context = st.session_state.pipeline.retrieval_system.get_context(question)
            
            st.subheader("Retrieved Context:")
            with st.expander("Show full context", expanded=True):
                st.text(context)
                
        except Exception as e:
            st.error(f"❌ Error retrieving context: {str(e)}")

def main():
    """Main Streamlit application"""
    initialize_session_state()
    setup_sidebar()
    main_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, LangChain, Neo4j, and Groq | "
        "[GitHub](https://github.com/yourusername/graphrag) | "
        "[Documentation](https://docs.yoursite.com)"
    )

if __name__ == "__main__":
    main()