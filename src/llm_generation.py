from pathlib import Path
from llama_cpp import Llama

class LLMGenerator:
    """Wrapper for LLM generation using llama.cpp."""
    
    def __init__(self, n_ctx=None, n_threads=4, verbose=False):
        """
        Initialize LLM generator.
        
        Args:
            n_ctx: Context window size (uses model's recommended size if None)
            n_threads: Number of CPU threads to use
            verbose: Whether to print detailed logs
        """
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        
        # TinyLlama 1.1B - 0.7-0.8GB memory, recommended n_ctx=2048
        # path = project_root / "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"
        # recommended_ctx = 2048
        
        # Llama 3.2 3B (default) - balanced speed and quality, recommended n_ctx=8192
        path = project_root / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        recommended_ctx = 8192
        
        # Qwen2 1.5B Instruct - excellent for question answering, recommended n_ctx=4096
        # path = project_root / "qwen2-1_5b-instruct-q4_0.gguf"
        # recommended_ctx = 4096
        
        # Qwen2 7B Instruct - most capable (requires more memory), recommended n_ctx=4096
        # path = project_root / "qwen2-7b-instruct-q4_0.gguf"
        # recommended_ctx = 4096

        self.path = Path(path)
        
        if not self.path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Use recommended context size if not specified
        context_size = n_ctx if n_ctx is not None else recommended_ctx
        
        print(f"Loading LLM model from {self.path.name}...")
        print(f"Using context window: {context_size} tokens")
        
        self.llm = Llama(
            model_path=str(self.path),
            n_ctx=context_size,
            n_threads=n_threads,
            verbose=verbose
        )
        
        print("LLM model loaded successfully!")
    
    def generate(self, prompt, max_tokens, temperature, stop=None):
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt (query + context)
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
            stop: List of stop sequences
            
        Returns:
            Generated text string
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Default stop sequences to prevent rambling
        if stop is None:
            stop = ["\n\n\n", "Question:", "Context:"]
        
        # Generate response
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            echo=False  # Don't echo the prompt in output
        )
        
        # Extract generated text
        generated_text = output["choices"][0]["text"]
        
        return generated_text.strip()
    
    def create_chat_prompt(self, query, context_docs):
        """
        Create a structured prompt combining query and retrieved documents.
        
        Args:
            query: User's question
            context_docs: List of document dicts with 'id' and 'text'
            
        Returns:
            Formatted prompt string
        """
        # Build context section with clear separation
        if not context_docs:
            context_text = "[No relevant information available]\n\n"
        else:
            context_text = ""
            for i, doc in enumerate(context_docs, 1):
                context_text += f"[{i}] {doc['text']}\n\n"
        
        prompt = f"""You are a helpful assistant. Use the information below to answer the question directly and naturally.

Information:
{context_text}

Question: {query}

Instructions:
- Answer the question directly as if you naturally know this information
- Never mention "the context", "the information", "according to", or "based on"
- Write as if you're explaining something you already know
- If you cannot answer with the information provided, say "I don't have enough information to answer that question" and reference "my provided texts"
- Be concise and conversational

Answer:"""
        
        return prompt