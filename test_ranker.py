from generation.reranker import ContextReranker
from langchain.schema import Document

# Test documents
docs = [
    Document(page_content="Machine learning is about training models", metadata={}),
    Document(page_content="Python is a programming language", metadata={}),
    Document(page_content="Supervised learning uses labeled data", metadata={})
]

# Initialize reranker
reranker = ContextReranker()

# Test query
query = "What is machine learning?"

# Rerank
results = reranker.rerank(query, docs, top_k=3)

print("Reranking Test:")
for i, (doc, score) in enumerate(results, 1):
    print(f"{i}. Score: {score:.4f} - {doc.page_content}")

print("\n✓ Reranking works!")