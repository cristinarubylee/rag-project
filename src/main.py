"""
Main RAG Pipeline - Interactive Question Answering System
Combines query encoding, vector search, document retrieval, and LLM generation.
"""

import sys
import threading
import time

from encode import QueryEncoder
from vector_db import VectorDB
from llm_generation import LLMGenerator

# Global flag for spinner animation
spinner_running = False


def spinner():
    """Display a loading spinner animation while LLM generates response."""
    frames = ["|", "/", "-", "\\"]
    i = 0
    while spinner_running:
        print(f"\rGenerating response... {frames[i % 4]}", end="", flush=True)
        time.sleep(0.15)
        i += 1


def process_query(user_input, encoder, vector_db, llm, top_k, similarity_threshold, debug=False):
    """
    Process a user query through the RAG pipeline.
    
    Args:
        user_input: User's question
        encoder: QueryEncoder instance
        vector_db: VectorDB instance
        llm: LLMGenerator instance
        top_k: Number of documents to retrieve
        debug: Whether to print detailed pipeline steps
    
    Returns:
        Generated answer string
    """
    if debug:
        print("\n" + "=" * 60)
        print(f"Query: {user_input}")
        print("=" * 60)
    
    # Step 1: Encode the query into a vector
    if debug:
        print("\n[Step 1] Encoding query...")
    query_embedding = encoder.encode(user_input)
    
    # Step 2: Search for top-k most similar documents
    if debug:
        print(f"\n[Step 2] Searching for top {top_k} relevant documents...")
    distances, indices = vector_db.search(query_embedding, k=top_k)
    
    if debug:
        print(f"  Found documents: {indices[0].tolist()}")
        print(f"  Distances: {[f'{d:.2f}' for d in distances[0]]}")
    
    # Step 3: Filter documents by similarity threshold
    filtered_indices = []
    filtered_distances = []

    if debug:
        print("\n[Step 3] Filtering documents...")
    for idx, dist in zip(indices[0], distances[0]):
        if dist <= similarity_threshold:
            filtered_indices.append(idx)
            filtered_distances.append(dist)

    if debug:
        print(f"After filtering (threshold={similarity_threshold}): {len(filtered_indices)} documents kept")
        if len(filtered_indices) < len(indices[0]):
            dropped = len(indices[0]) - len(filtered_indices)
            print(f"Dropped {dropped} document(s) as too dissimilar")
            
    # Step 4: Retrieve the actual document texts
    if debug:
        print("\n[Step 4] Retrieving document texts...")
    retrieved_docs = vector_db.get_documents_by_ids(filtered_indices)
    
    if debug:
        for i, doc in enumerate(retrieved_docs, 1):
            preview = doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text']
            print(f"  [{i}] ID {doc['id']}: {preview}")
    
    # Step 5: Create augmented prompt with retrieved context
    if debug:
        print("\n[Step 5] Creating augmented prompt...")
    prompt = llm.create_chat_prompt(user_input, retrieved_docs)
    
    # Step 6: Generate response using LLM with spinner animation
    if debug:
        print("\n[Step 6] Generating response with LLM...")
    
    global spinner_running
    spinner_running = True
    spin_thread = threading.Thread(target=spinner, daemon=True)
    spin_thread.start()
    
    answer = llm.generate(prompt, max_tokens=200, temperature=0.3)
    
    spinner_running = False
    spin_thread.join()
    print("\r", end="", flush=True)  # Clear spinner line
    
    return answer


def main():
    """Main entry point - Initialize components and run interactive loop."""
    
    # Check if debug mode is enabled via command line
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == "debug"
    
    try:
        # Display ASCII banner
        ascii_art = r"""
        +========================================================================+
        | ___       _                      _   _             ____      _    ____ |
        ||_ _|_ __ | |_ ___ _ __ __ _  ___| |_(_)_   _____  |  _ \    / \  / ___||
        | | || '_ \| __/ _ \ '__/ _` |/ __| __| \ \ / / _ \ | |_) |  / _ \| |  _ |
        | | || | | | ||  __/ | | (_| | (__| |_| |\ V /  __/ |  _ <  / ___ \ |_| ||
        ||___|_| |_|\__\___|_|  \__,_|\___|\__|_| \_/ \___| |_| \_\/_/   \_\____||
        +========================================================================+
        """
        print(ascii_art)
        
        # Initialize RAG pipeline components
        print("\n[1/3] Loading Query Encoder...")
        encoder = QueryEncoder()
        
        print("\n[2/3] Loading Vector Database...")
        vector_db = VectorDB()
        vector_db.load_documents()
        vector_db.build_index()
        
        print("\n[3/3] Loading LLM...")
        llm = LLMGenerator(n_ctx=2048, n_threads=4)
        
        print("\n" + "=" * 60)
        print("★ RAG Pipeline Ready! ★")
        print("=" * 60)
        
        # Display usage instructions
        print("Type your questions below.")
        print("Commands:")
        print("  'quit' or 'exit' - Exit the program")
        print("  'help' - Show this help message")
        print("=" * 60)
        
        top_k = 5  # Number of documents to retrieve per query
        similarity_threshold = 0.80 # Lower = stricter similarity check
        
        # Interactive query loop
        while True:
            try:
                user_input = input("\n> ").strip()
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Handle exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n\n👋 Goodbye!")
                    break
                
                # Handle help command
                if user_input.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  Type any question to get an AI-powered answer")
                    print("  'quit' or 'exit' - Exit the program")
                    continue
                
                # Process the query through RAG pipeline
                answer = process_query(user_input, encoder, vector_db, llm, top_k, similarity_threshold, debug)
                
                # Display the generated answer
                print("\n" + "=" * 60)
                print("★ Answer ★")
                print("=" * 60)
                print(answer)
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}")
                print("Please try another question.")
    
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during initialization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()