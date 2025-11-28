
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import KNOWLEDGE_BASE_DIR, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME
from data.loader import PDFLoader
from data.processor import DocumentProcessor
from vectorstore.embeddings import EmbeddingManager
from vectorstore.store import VectorStore
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_knowledge_base(force_recreate: bool = False):
    """Setup knowledge base from PDFs"""
    
    logger.info("="*80)
    logger.info("KNOWLEDGE BASE SETUP")
    logger.info("="*80)
    
    # Step 1: Load PDFs
    logger.info("\n[Step 1/4] Loading PDFs...")
    loader = PDFLoader(str(KNOWLEDGE_BASE_DIR))
    documents = loader.load_all_pdfs()
    
    if not documents:
        logger.error("No documents loaded. Please add PDF files to knowledge-base/books/")
        return False
    
    logger.info(f"✓ Loaded {len(documents)} document pages")
    
    # Step 2: Process and chunk documents
    logger.info(f"\n[Step 2/4] Processing documents (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = processor.split_documents(documents)
    
    stats = processor.get_chunk_statistics(chunks)
    logger.info(f"✓ Created {stats['total_chunks']} chunks")
    logger.info(f"  - Avg chunk length: {stats['avg_chunk_length']:.0f} characters")
    logger.info(f"  - Unique sources: {stats['unique_sources']}")
    
    # Step 3: Initialize embeddings
    logger.info("\n[Step 3/4] Initializing embeddings...")
    embedding_manager = EmbeddingManager()
    
    if not embedding_manager.test_embedding():
        logger.error("Embedding test failed. Check your OpenAI API key.")
        return False
    
    embeddings = embedding_manager.get_embeddings()
    
    # Step 4: Create vectorstore
    logger.info("\n[Step 4/4] Creating ChromaDB vectorstore...")
    vectorstore_manager = VectorStore(
        persist_directory=str(CHROMA_DB_DIR),
        embeddings=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    vectorstore_manager.create_vectorstore(chunks, force_recreate=force_recreate)
    
    vs_stats = vectorstore_manager.get_statistics()
    logger.info(f"✓ Vectorstore created successfully")
    logger.info(f"  - Total vectors: {vs_stats['total_vectors']:,}")
    
    # FIXED: Handle embedding_dimensions which might be string or int
    embedding_dim = vs_stats.get('embedding_dimensions', 'N/A')
    if isinstance(embedding_dim, (int, float)):
        logger.info(f"  - Embedding dimensions: {int(embedding_dim):,}")
    else:
        logger.info(f"  - Embedding dimensions: {embedding_dim}")
    
    logger.info(f"  - Location: {vs_stats['persist_directory']}")
    
    # Test retrieval
    logger.info("\n[Test] Testing retrieval...")
    test_query = "machine learning"
    results = vectorstore_manager.similarity_search(test_query, k=2)
    logger.info(f"✓ Retrieved {len(results)} results for query: '{test_query}'")
    
    if results:
        doc, score = results[0]
        logger.info(f"  Top result (score: {score:.4f}):")
        logger.info(f"  Source: {doc.metadata.get('source_file', 'unknown')}")
        logger.info(f"  Preview: {doc.page_content[:150]}...")
    
    logger.info("\n" + "="*80)
    logger.info("KNOWLEDGE BASE SETUP COMPLETE!")
    logger.info("="*80)
    logger.info("\nYou can now run the Streamlit app with: streamlit run ui/app.py")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup knowledge base from PDFs")
    parser.add_argument("--force", action="store_true", help="Force recreate vectorstore")
    args = parser.parse_args()
    
    success = setup_knowledge_base(force_recreate=args.force)
    
    if not success:
        sys.exit(1)