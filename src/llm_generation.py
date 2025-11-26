from pathlib import Path
from llama_cpp import Llama

class LLMGenerator:
    """Wrapper for LLM generation using llama.cpp."""
    
    def __init__(self, path=None, n_ctx=2048, n_threads=4, verbose=False):
        """
        Initialize LLM generator.
        
        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size (default 2048)
            n_threads: Number of CPU threads to use
            verbose: Whether to print detailed logs
        """
        if path is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            path = project_root / "tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf"

        self.path = Path(path)
        
        if not self.path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        print(f"Loading LLM model from {self.path.name}...")
        
        self.llm = Llama(
            model_path=str(self.path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose
        )
        
        print("LLM model loaded successfully!")
    
    def generate(self, prompt, max_tokens=512, temperature=0.7, stop=None):
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
        context_text = ""
        for i, doc in enumerate(context_docs, 1):
            context_text += f"[{i}] {doc['text']}\n\n"
        
        # Ultra-simple prompt for basic models
        # Key: Use "based on" to anchor the response and limit hallucination
        prompt = f"""Question: {query}

Reference information:
{context_text}

Based only on the reference information above, provide a brief answer to the question. If the information is not relevant, say so.

Answer:"""
        
        return prompt
