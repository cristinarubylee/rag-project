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


def process_query(user_input, encoder, vector_db, llm, top_k, debug=False):
    """
    Process a user query through the RAG pipeline.
    
    Args:
        user_input: User's question
        encoder: QueryEncoder instance
        vector_db: VectorDB instance
        llm: LLMGenerator instance
        top_k: Number of documents to retrieve
        debug: Whether to print detailed pipeline steps and performance metrics
    
    Returns:
        Generated answer string
    """
    # Performance tracking
    perf_metrics = {}
    
    # Step 1: Encode the query into a vector
    if debug:
        print("\n[Step 1] Encoding query...")
    
    t_start = time.perf_counter()
    query_embedding = encoder.encode(user_input)
    perf_metrics['encode_time'] = time.perf_counter() - t_start
    
    if debug:
        print(f"  ⏱️  Encoding time: {perf_metrics['encode_time']*1000:.2f} ms")
    
    # Step 2: Search for top-k most similar documents
    if debug:
        print(f"\n[Step 2] Searching for top {top_k} relevant documents...")
    
    t_start = time.perf_counter()
    distances, indices = vector_db.search(query_embedding, k=top_k)
    perf_metrics['search_time'] = time.perf_counter() - t_start
    
    if debug:
        print(f"  Found documents: {indices[0].tolist()}")
        print(f"  Distances: {[f'{d:.2f}' for d in distances[0]]}")
        print(f"  ⏱️  Search time: {perf_metrics['search_time']*1000:.2f} ms")
    
    # Step 3: Retrieve the actual document texts
    if debug:
        print("\n[Step 3] Retrieving document texts...")
    
    t_start = time.perf_counter()
    retrieved_docs = vector_db.get_documents_by_ids(indices[0])
    perf_metrics['retrieval_time'] = time.perf_counter() - t_start
    
    if debug:
        for i, doc in enumerate(retrieved_docs, 1):
            preview = doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text']
            print(f"  [{i}] ID {doc['id']}: {preview}")
        print(f"  ⏱️  Retrieval time: {perf_metrics['retrieval_time']*1000:.2f} ms")
    
    # Step 4: Create augmented prompt with retrieved context
    if debug:
        print("\n[Step 4] Creating augmented prompt...")
    
    t_start = time.perf_counter()
    prompt = llm.create_chat_prompt(user_input, retrieved_docs)
    perf_metrics['prompt_time'] = time.perf_counter() - t_start
    
    if debug:
        print(f"  ⏱️  Prompt creation time: {perf_metrics['prompt_time']*1000:.2f} ms")
    
    # Step 5: Generate response using LLM with spinner animation
    if debug:
        print("\n[Step 5] Generating response with LLM...")
    
    global spinner_running
    spinner_running = True
    spin_thread = threading.Thread(target=spinner, daemon=True)
    spin_thread.start()
    
    t_start = time.perf_counter()
    answer = llm.generate(prompt, max_tokens=200, temperature=0.3)
    perf_metrics['generation_time'] = time.perf_counter() - t_start
    
    spinner_running = False
    spin_thread.join()
    print("\r", end="", flush=True)  # Clear spinner line
    
    if debug:
        print(f"  ⏱️  Generation time: {perf_metrics['generation_time']*1000:.2f} ms")
    
    # Calculate total time
    perf_metrics['total_time'] = sum(perf_metrics.values())
    
    # Print performance summary
    if debug:
        print("\n" + "=" * 60)
        print("⏱️  PERFORMANCE BREAKDOWN")
        print("=" * 60)
        print(f"  Query Encoding:        {perf_metrics['encode_time']*1000:8.2f} ms ({perf_metrics['encode_time']/perf_metrics['total_time']*100:5.1f}%)")
        print(f"  Vector Search:   {perf_metrics['search_time']*1000:8.2f} ms ({perf_metrics['search_time']/perf_metrics['total_time']*100:5.1f}%)")
        print(f"  Doc Retrieval:   {perf_metrics['retrieval_time']*1000:8.2f} ms ({perf_metrics['retrieval_time']/perf_metrics['total_time']*100:5.1f}%)")
        print(f"  Prompt Augmentation: {perf_metrics['prompt_time']*1000:8.2f} ms ({perf_metrics['prompt_time']/perf_metrics['total_time']*100:5.1f}%)")
        print(f"  LLM Generation:  {perf_metrics['generation_time']*1000:8.2f} ms ({perf_metrics['generation_time']/perf_metrics['total_time']*100:5.1f}%)")
        print("-" * 60)
        print(f"  TOTAL:           {perf_metrics['total_time']*1000:8.2f} ms")
        print("=" * 60)
    
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
        
        if debug:
            print("\n🔍 DEBUG MODE ENABLED - Performance metrics will be displayed")
        
        # Initialize RAG pipeline components with performance tracking
        print("\n[1/3] Loading Query Encoder...")
        t_start = time.perf_counter()
        encoder = QueryEncoder()
        encoder_load_time = time.perf_counter() - t_start
        if debug:
            print(f"  ⏱️  Loaded in {encoder_load_time:.3f}s")
        
        print("\n[2/3] Loading Vector Database...")
        t_start = time.perf_counter()
        vector_db = VectorDB()
        vector_db.load_documents()
        vector_db.build_index()
        db_load_time = time.perf_counter() - t_start
        if debug:
            print(f"  ⏱️  Loaded in {db_load_time:.3f}s")
        
        print("\n[3/3] Loading LLM...")
        t_start = time.perf_counter()
        llm = LLMGenerator()
        llm_load_time = time.perf_counter() - t_start
        if debug:
            print(f"  ⏱️  Loaded in {llm_load_time:.3f}s")
            print(f"\n  Total initialization time: {encoder_load_time + db_load_time + llm_load_time:.3f}s")
        
        print("\n" + "=" * 60)
        print("★ RAG Pipeline Ready! ★")
        print("=" * 60)
        
        # Display usage instructions
        print("Type your questions below.")
        print("Commands:")
        print("  'quit' or 'exit' - Exit the program")
        print("  'help' - Show this help message")
        if not debug:
            print("  Run with 'python main.py debug' for performance metrics")
        print("=" * 60)
        
        top_k = 5  # Number of documents to retrieve per query
        
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
                answer = process_query(user_input, encoder, vector_db, llm, top_k, debug)
                
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