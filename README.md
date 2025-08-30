# GraphRAG: Knowledge Graph-based Retrieval Augmented Generation

A modular and scalable system that combines knowledge graphs with retrieval-augmented generation (RAG) to provide intelligent question-answering over documents.

## 🌟 Features

- **Modular Architecture**: Clean, maintainable code structure with separated concerns
- **Multiple Input Sources**: Support for PDF uploads and Wikipedia queries
- **Knowledge Graph Construction**: Automatic entity and relationship extraction
- **Hybrid Search**: Combines structured (graph) and unstructured (vector) search
- **Interactive Web Interface**: User-friendly Streamlit application
- **Chat History Support**: Contextual conversations with memory
- **Configurable Models**: Support for various LLMs via Groq API

## 🏗️ Architecture

```
├── config.py              # Configuration management
├── document_loader.py     # Document loading and chunking
├── entity_extractor.py    # Entity extraction from queries
├── graph_manager.py       # Knowledge graph operations
├── retrieval_system.py    # RAG retrieval and QA
├── main.py               # Main pipeline orchestration
├── streamlit_app.py      # Web interface
└── requirements.txt      # Dependencies
```

## 🚀 Quick Start

### Prerequisites

1. **Neo4j Database**: Set up a Neo4j instance (local or cloud)
2. **Groq API Key**: Get your API key from [Groq](https://groq.com)
3. **Python 3.8+**: Ensure you have Python installed

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd graphrag-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file or set environment variables
export NEO4J_URI="neo4j+s://your-neo4j-uri"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"
export GROQ_API_KEY="your-groq-api-key"
export HF_TOKEN="your-huggingface-token"  # Optional
```

### Running the Application

#### Option 1: Streamlit Web Interface
```bash
streamlit run streamlit_app.py
```

#### Option 2: Python Script
```python
from main import GraphRAGPipeline

# Initialize pipeline
pipeline = GraphRAGPipeline()

# Process documents (Wikipedia example)
pipeline.process_documents("Artificial Intelligence", "wikipedia")

# Ask questions
answer = pipeline.ask_question("What is machine learning?")
print(answer)
```

## 📖 Usage

### Processing Documents

**PDF Upload:**
1. Click on the "PDF Upload" tab
2. Upload your PDF file
3. Choose whether to clear existing graph data
4. Click "Process PDF"

**Wikipedia Query:**
1. Click on the "Wikipedia" tab
2. Enter your search query (e.g., "Climate Change")
3. Choose whether to clear existing graph data
4. Click "Process Wikipedia"

### Asking Questions

Once documents are processed:
1. Enter your question in the text input
2. Click "Ask Question" to get an answer
3. Click "Show Context" to see retrieved information
4. View chat history for previous conversations

## ⚙️ Configuration

The system can be configured through the `GraphRAGConfig` class:

```python
config = GraphRAGConfig(
    model_name="deepseek-r1-distill-llama-70b",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    chunk_size=512,
    chunk_overlap=24,
    temperature=0.3
)
```

### Available Models

- `deepseek-r1-distill-llama-70b` (recommended)
- `llama3-70b-8192`
- `mixtral-8x7b-32768`

## 🔧 System Components

### Document Loader (`document_loader.py`)
- Handles PDF and Wikipedia document loading
- Implements text chunking with configurable parameters
- Supports multiple input sources

### Graph Manager (`graph_manager.py`)
- Manages Neo4j knowledge graph operations
- Creates nodes and relationships from documents
- Handles vector indexing for hybrid search

### Entity Extractor (`entity_extractor.py`)
- Extracts entities from user queries
- Uses structured output parsing
- Supports person, organization, and location entities

### Retrieval System (`retrieval_system.py`)
- Implements hybrid retrieval (graph + vector search)
- Handles chat history and context
- Provides comprehensive question-answering

## 📊 Graph Visualization

The system creates rich knowledge graphs with:
- **Entities**: People, organizations, locations, concepts
- **Relationships**: Connections between entities
- **Documents**: Source traceability

## 🛠️ Development

### Project Structure
```
graphrag-system/
│
├── config.py              # System configuration
├── document_loader.py     # Document processing
├── entity_extractor.py    # Entity extraction
├── graph_manager.py       # Graph operations
├── retrieval_system.py    # RAG system
├── main.py               # Main pipeline
├── streamlit_app.py      # Web interface
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

### Adding New Features

1. **New Document Types**: Extend `DocumentLoader` class
2. **Custom Models**: Update configuration and model initialization
3. **Enhanced Retrieval**: Modify `RetrievalSystem` class
4. **UI Improvements**: Update `streamlit_app.py`

## 🔍 Troubleshooting

### Common Issues

1. **Connection Errors**:
   - Verify Neo4j credentials and connectivity
   - Check if Neo4j instance is running

2. **Model Errors**:
   - Ensure Groq API key is valid
   - Check model name spelling

3. **Memory Issues**:
   - Reduce chunk_size for large documents
   - Process documents in smaller batches

### Debug Mode

Enable debug information by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [Neo4j](https://neo4j.com/) for graph database technology
- [Groq](https://groq.com/) for fast LLM inference
- [Streamlit](https://streamlit.io/) for the web interface
- [HuggingFace](https://huggingface.co/) for embeddings models

## 📧 Contact

For questions or support, please open an issue or contact [your-email@example.com](mailto:your-email@example.com).

---

**Built with ❤️ for the AI community**
