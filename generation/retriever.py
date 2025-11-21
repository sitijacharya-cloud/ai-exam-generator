"""
Context retrieval from vector store with enhanced logging and reranking
UPDATED: Added Cross-Encoder reranking support
"""

from typing import List, Tuple
from models.schemas import RetrievedContext
from generation.reranker import ContextReranker  # NEW IMPORT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextRetriever:
    """Retrieves relevant context from vectorstore with optional reranking"""
    
    def __init__(
        self, 
        vectorstore, 
        relevance_threshold: float = 0.5,
        use_reranking: bool = True  # NEW PARAMETER
    ):
        """
        Args:
            vectorstore: ChromaDB vectorstore
            relevance_threshold: Minimum similarity score (0-1)
            use_reranking: Whether to use Cross-Encoder reranking (recommended)
        """
        self.vectorstore = vectorstore
        self.relevance_threshold = relevance_threshold
        self.use_reranking = use_reranking  # NEW
        
        # Initialize reranker if enabled
        if self.use_reranking:
            try:
                self.reranker = ContextReranker()
                logger.info("✓ Cross-Encoder reranker initialized")
            except Exception as e:
                logger.error(f"Failed to initialize reranker: {e}")
                logger.warning("Falling back to standard retrieval")
                self.use_reranking = False
                self.reranker = None
        else:
            self.reranker = None
            logger.info("Reranking disabled - using standard retrieval")
    
    def retrieve_for_topic(self, topic: str, k: int = 3) -> List[RetrievedContext]:
        """
        Retrieve top-k relevant contexts for a topic
        Now with optional reranking for better quality
        
        Args:
            topic: The syllabus topic to search for
            k: Number of contexts to retrieve
            
        Returns:
            List of RetrievedContext objects (empty if none are relevant enough)
        """
        if not self.vectorstore:
            logger.error("Vectorstore not initialized")
            return []
        
        logger.info(f"Retrieving top-{k} contexts for topic: {topic}")
        logger.info(f"Relevance threshold: {self.relevance_threshold}")
        logger.info(f"Reranking: {'enabled' if self.use_reranking else 'disabled'}")
        
        try:
            # Step 1: Initial retrieval
            # If reranking is enabled, retrieve more candidates for better reranking
            initial_k = k * 3 if self.use_reranking else k
            
            results = self.vectorstore.similarity_search_with_score(topic, k=initial_k)
            
            if not results:
                logger.warning(f"No results found for topic: {topic}")
                return []
            
            logger.info(f"Initial retrieval: {len(results)} documents")
            
            # Step 2: Reranking (if enabled)
            if self.use_reranking and self.reranker and len(results) > 1:
                # Extract documents from results
                documents = [doc for doc, score in results]
                
                # Rerank using Cross-Encoder
                logger.info(f"Reranking {len(documents)} documents...")
                reranked_results = self.reranker.rerank(topic, documents, top_k=k*2)
                
                # Convert reranker scores to our format
                # Reranker returns similarity scores (0-10 typically)
                # Normalize to 0-1 range
                if reranked_results:
                    max_score = max(score for _, score in reranked_results)
                    min_score = min(score for _, score in reranked_results)
                    score_range = max_score - min_score if max_score > min_score else 1.0
                    
                    # Normalize scores
                    normalized_results = []
                    for doc, score in reranked_results:
                        normalized_score = (score - min_score) / score_range if score_range > 0 else 0.5
                        normalized_results.append((doc, normalized_score))
                    
                    results = normalized_results
                    logger.info(f"✓ Reranking complete: top {len(results)} documents")
            else:
                # No reranking - convert distance to similarity
                results = [(doc, max(0, 1 - (distance / 2))) for doc, distance in results]
            
            # Step 3: Filter by threshold and convert to RetrievedContext
            contexts = []
            relevant_count = 0
            
            for idx, (doc, similarity_score) in enumerate(results):
                # Check if this context meets the relevance threshold
                is_relevant = similarity_score >= self.relevance_threshold
                
                logger.info(f"  [{idx+1}] Similarity: {similarity_score:.4f}, Relevant: {is_relevant}, Source: {doc.metadata.get('source_file', 'Unknown')}")
                
                if not is_relevant:
                    logger.warning(f"  ⚠️ Context {idx+1} below threshold ({similarity_score:.4f} < {self.relevance_threshold}), skipping")
                    continue
                
                relevant_count += 1
                
                # Extract source file
                source_file = doc.metadata.get('source_file', 'Unknown')
                
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
                    relevance_score=float(similarity_score)
                )
                contexts.append(context)
                
                # Stop if we have enough contexts
                if len(contexts) >= k:
                    break
            
            if relevant_count == 0:
                logger.warning(f"❌ No relevant contexts found for '{topic}' (all below threshold {self.relevance_threshold})")
                logger.info(f"💡 Will use LLM's general knowledge instead")
                return []
            
            logger.info(f"✓ Retrieved {len(contexts)} relevant contexts")
            
            # Log source diversity
            sources = set(ctx.metadata['source_file'] for ctx in contexts)
            logger.info(f"  Sources: {', '.join(sorted(sources))}")
            
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
                similarity = max(0, 1 - (score / 2))
                source = doc.metadata.get('source_file', 'Unknown')
                if source not in sources:
                    sources[source] = []
                sources[source].append((doc.metadata.get('page', 'N/A'), similarity))
            
            logger.info(f"Found results from {len(sources)} sources:")
            for source, pages in sources.items():
                avg_similarity = sum(p[1] for p in pages) / len(pages)
                logger.info(f"  - {source}: {len(pages)} chunks (avg similarity: {avg_similarity:.4f})")
        
        logger.info("="*60)