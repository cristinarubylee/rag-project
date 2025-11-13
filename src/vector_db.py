import json
import faiss
import numpy as np
from pathlib import Path

class VectorDB:
    """
    A vector database using FAISS for semantic document search.
    """

    def __init__(self, path=None):
        """
        Initialize the vector database.
        
        Args:
            path: Path to preprocessed_documents.json
                If None, uses default relative path
        """
        self.index = None
        self.documents = []
        self.dimension = 768 # BGE model output dimension

        if path is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            path = project_root / "data" / "processed" / "preprocessed_documents.json"
        
        self.path = path

    def load_documents(self):
        """Load preprocessed documents with embeddings from JSON file."""
        print(f"Loading documents from {self.path}...")
        
        with open(self.path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        
        print(f"Loaded {len(self.documents)} documents")
        return self.documents
    
    def build_index(self):
        """Build FAISS index from loaded documents."""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        print("Building FAISS index...")
        
        # Extract embeddings into a numpy array
        embeddings = np.array([doc["embedding"] for doc in self.documents], dtype=np.float32)
        
        # Create FAISS index (using L2 distance)
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add all embeddings to the index
        self.index.add(embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
        return self.index
    
    def search(self, query_embeddings, k=5):
        """
        Search for top-k most similar documents.
        
        Args:
            query_embeddings: Query vector(s) - shape (batch_size, 768)
            k: Number of results to return
            
        Returns:
            distances: Array of shape (batch_size, k) with L2 distances
            indices: Array of shape (batch_size, k) with document indices
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Ensure query is 2D: (batch_size, 768) and float32
        query_embeddings = np.atleast_2d(query_embeddings).astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embeddings, k)
        
        return distances, indices
    
    def get_documents_by_ids(self, doc_ids):
        """
        Retrieve document texts by their IDs.
        
        Args:
            doc_ids: List or array of document indices
            
        Returns:
            List of document dictionaries with 'id' and 'text'
        """
        results = []
        for doc_id in doc_ids:
            if 0 <= doc_id < len(self.documents):
                results.append({
                    "id": self.documents[doc_id]["id"],
                    "text": self.documents[doc_id]["text"]
                })
        return results
    
def main():
    print("=" * 60)
    print("Part 1: Vector Database Testing")
    print("=" * 60)
    
    # Initialize database
    db = VectorDB()
    
    # Load and build
    db.load_documents()
    db.build_index()
    
    print("\n" + "=" * 60)
    print("Testing: Self-Search Validation")
    print("=" * 60)
    
    # Test: Use document 42's embedding as query
    test_doc_id = 42
    test_embedding = np.array(db.documents[test_doc_id]["embedding"], dtype=np.float32)
    
    print(f"\nQuerying with document {test_doc_id}'s embedding...")
    print(f"Document text preview: {db.documents[test_doc_id]['text'][:100]}...")
    
    # Search
    distances, indices = db.search(test_embedding, k=5)
    
    # Display results
    print(f"\nTop 5 Results:")
    print(f"Distances: {distances[0]}")
    print(f"Indices:   {indices[0]}")
    
    # Verify top result is the document itself
    top_result_id = indices[0][0]
    top_distance = distances[0][0]
    
    print(f"\n✓ Validation Check:")
    print(f"  Expected top result: Document {test_doc_id}")
    print(f"  Actual top result:   Document {top_result_id}")
    print(f"  Distance:            {top_distance:.6f}")
    
    if top_result_id == test_doc_id and top_distance < 0.001:
        print("  Status: ✓ PASSED - Self-search works correctly!")
    else:
        print("  Status: ✗ FAILED - Something is wrong with the index!")
    
    # Show retrieved document texts
    print("\n" + "=" * 60)
    print("Retrieved Documents:")
    print("=" * 60)
    retrieved_docs = db.get_documents_by_ids(indices[0])
    for i, doc in enumerate(retrieved_docs):
        print(f"\n[{i+1}] Document ID: {doc['id']}")
        print(f"    Text: {doc['text'][:150]}...")

if __name__ == "__main__":
    main()