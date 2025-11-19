"""
Quick test script to verify the fixes
Place this in the root directory and run: python test_setup.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import CHROMA_DB_DIR, COLLECTION_NAME
from vectorstore.embeddings import EmbeddingManager
from vectorstore.store import VectorStore

def test_vectorstore():
    """Test if vectorstore can be loaded and stats retrieved"""
    print("="*80)
    print("TESTING VECTORSTORE")
    print("="*80)
    
    try:
        # Initialize embeddings
        print("\n[1/3] Initializing embeddings...")
        embedding_manager = EmbeddingManager()
        embeddings = embedding_manager.get_embeddings()
        print("✓ Embeddings initialized")
        
        # Load vectorstore
        print("\n[2/3] Loading vectorstore...")
        vectorstore_manager = VectorStore(
            persist_directory=str(CHROMA_DB_DIR),
            embeddings=embeddings,
            collection_name=COLLECTION_NAME
        )
        
        if not CHROMA_DB_DIR.exists():
            print("❌ Vectorstore not found. Please run: python scripts/setup_knowledge_base.py")
            return False
        
        vectorstore = vectorstore_manager.load_vectorstore()
        print("✓ Vectorstore loaded")
        
        # Get statistics
        print("\n[3/3] Getting statistics...")
        stats = vectorstore_manager.get_statistics()
        print("✓ Statistics retrieved successfully!")
        print(f"\nVectorstore Stats:")
        print(f"  - Total vectors: {stats.get('total_vectors', 0):,}")
        print(f"  - Collection name: {stats.get('collection_name', 'N/A')}")
        print(f"  - Embedding dimensions: {stats.get('embedding_dimensions', 'N/A')}")
        print(f"  - Location: {stats.get('persist_directory', 'N/A')}")
        
        # Test search
        print("\n[Test] Testing search...")
        results = vectorstore_manager.similarity_search("machine learning", k=2)
        print(f"✓ Search successful! Found {len(results)} results")
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nYou can now run: streamlit run ui/app.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vectorstore()
    sys.exit(0 if success else 1)