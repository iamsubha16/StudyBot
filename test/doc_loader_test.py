import pytest
from pathlib import Path
from langchain.schema import Document
from src.loaders.document_loaders import DocumentLoader, SourceType
from src.config.config import GraphRAGConfig
from logger import logger

# -------------------------
# Fixtures
# -------------------------
@pytest.fixture
def config():
    """GraphRAGConfig for testing with real data"""
    return GraphRAGConfig(
        chunk_size=500,
        chunk_overlap=50
    )


@pytest.fixture
def loader(config):
    return DocumentLoader(config)


# -------------------------
# Test loading an actual PDF file
# -------------------------
def test_load_from_pdf_path_real(loader):
    # pdf_path = Path("data/The-Merchant-of-Venice-PDF.pdf")
    pdf_path = Path(__file__).parent / "data" / "The-Merchant-of-Venice-PDF.pdf"

    if not pdf_path.exists():
        logger.warning(f"PDF file not found: {pdf_path}, skipping test")
        pytest.skip("Real PDF file not found at data/The-Merchant-of-Venice-PDF.pdf")

    try:
        docs = loader.load_from_pdf_path(str(pdf_path))
        logger.info(f"Number of pages/documents loaded: {len(docs)}")
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)
        logger.info(f"Loaded {len(docs)} documents from actual PDF")
    except Exception as e:
        logger.error(f"Error loading PDF from path {pdf_path}: {e}", exc_info=True)
        pytest.fail(f"Failed to load PDF: {e}")


# -------------------------
# Test splitting actual PDF documents
# -------------------------
def test_split_documents_real(loader):
    pdf_path = Path(__file__).parent / "data" / "The-Merchant-of-Venice-PDF.pdf"
    
    if not pdf_path.exists():
        logger.warning(f"PDF file not found: {pdf_path}, skipping test")
        pytest.skip("Real PDF file not found at data/The-Merchant-of-Venice-PDF.pdf")

    docs = loader.load_from_pdf_path(str(pdf_path))
    chunks = loader.split_documents(docs)
    assert isinstance(chunks, list)
    assert all(isinstance(c, Document) for c in chunks)
    for chunk in chunks:
        assert len(chunk.page_content) <= loader.config.chunk_size
    logger.info(f"Split PDF into {len(chunks)} chunks")


# -------------------------
# Test loading Wikipedia pages
# -------------------------
@pytest.mark.parametrize("query,max_docs", [
    ("Python (programming language)", 2),
    ("Artificial intelligence", 2)
])
def test_load_from_wikipedia_real(loader, query, max_docs):
    docs = loader.load_from_wikipedia(query, max_docs=max_docs)
    assert isinstance(docs, list)
    assert len(docs) <= max_docs
    assert all(isinstance(d, Document) for d in docs)
    logger.info(f"Wikipedia query '{query}' returned {len(docs)} documents")


# -------------------------
# Test full load_and_split workflow
# -------------------------
def test_load_and_split_wikipedia_real(loader):
    query = "Machine learning"
    chunks = loader.load_and_split(query, source_type=SourceType.WIKIPEDIA)
    assert isinstance(chunks, list)
    assert all(isinstance(c, Document) for c in chunks)
    logger.info(f"load_and_split WIKIPEDIA returned {len(chunks)} chunks")
