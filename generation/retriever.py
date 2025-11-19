"""Context retrieval from vector store with enhanced logging"""

from typing import List, Tuple
from models.schemas import RetrievedContext
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextRetriever:
    """Retrieves relevant context from vectorstore"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
    
    def retrieve_for_topic(self, topic: str, k: int = 3) -> List[RetrievedContext]:
        """
        Retrieve top-k relevant contexts for a topic
        
        Args:
            topic: The syllabus topic to search for
            k: Number of contexts to retrieve
            
        Returns:
            List of RetrievedContext objects
        """
        if not self.vectorstore:
            logger.error("Vectorstore not initialized")
            return []
        
        logger.info(f"Retrieving top-{k} contexts for topic: {topic}")
        
        try:
            # Perform similarity search with scores
            results = self.vectorstore.similarity_search_with_score(topic, k=k)
            
            contexts = []
            sources_found = set()
            
            for idx, (doc, score) in enumerate(results):
                # Extract source file
                source_file = doc.metadata.get('source_file', 'Unknown')
                sources_found.add(source_file)
                
                # Extract 2-3 lines for preview
                lines = doc.page_content.split('\n')
                preview_lines = [line.strip() for line in lines if line.strip()][:3]
                preview = ' '.join(preview_lines)[:300]
                
                context = RetrievedContext(
                    content=doc.page_content,
                    metadata={
                        'source_file': source_file,
                        'page': doc.metadata.get('page', 'N/A'),
                        'chunk_id': doc.metadata.get('chunk_id', idx),
                        'preview': preview
                    },
                    relevance_score=float(1 - score)  # Convert distance to similarity
                )
                contexts.append(context)
                
                # Log each retrieved context
                logger.info(f"  [{idx+1}] Source: {source_file}, Page: {doc.metadata.get('page', 'N/A')}, Score: {1-score:.4f}")
            
            logger.info(f"✓ Retrieved {len(contexts)} contexts from {len(sources_found)} different sources")
            logger.info(f"  Sources: {', '.join(sorted(sources_found))}")
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error retrieving contexts: {e}")
            return []
    
    def get_combined_context(self, contexts: List[RetrievedContext]) -> str:
        """Combine multiple contexts into a single string"""
        return "\n\n".join([ctx.content for ctx in contexts])
    
    def test_retrieval_distribution(self, test_queries: List[str], k: int = 5):
        """
        Test method to check if retrieval is working across all sources
        Useful for debugging
        """
        logger.info("="*60)
        logger.info("TESTING RETRIEVAL DISTRIBUTION")
        logger.info("="*60)
        
        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            sources = {}
            for doc, score in results:
                source = doc.metadata.get('source_file', 'Unknown')
                if source not in sources:
                    sources[source] = []
                sources[source].append((doc.metadata.get('page', 'N/A'), 1-score))
            
            logger.info(f"Found results from {len(sources)} sources:")
            for source, pages in sources.items():
                logger.info(f"  - {source}: {len(pages)} chunks (pages: {[p[0] for p in pages[:3]]})")
        
        logger.info("="*60)