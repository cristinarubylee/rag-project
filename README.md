# RAG System Analysis and Optimization

This project implements a Retrieval-Augmented Generation (RAG) pipeline and analyzes its performance from a systems perspective.
It was developed as the final project for Cornell CS 4414 (Fall 2025), built entirely from scratch without any provided starter code.

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/cristinarubylee/rag-project.git
cd rag-project
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download datasets:**
```bash
wget -c https://cornell.box.com/shared/static/ffwnimisulvjyzp1t9u3201uj3p0w3bk.json -O documents.json
wget -c https://cornell.box.com/shared/static/eg115icq6oz2shnq63vzjw3776v30mwy.json -O queries.json
```

Move both files into `data/raw/`

5. **Download LLM models:**

Choose one or more models to download. Llama 3.2 is required as the default:
```bash
# TinyLlama 1.1B (Optional - fastest, 0.7-0.8GB)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf

# Llama 3.2 3B (Required - default model, balanced speed/quality)
wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Qwen2 1.5B Instruct (Optional - optimized for Q&A)
wget https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_0.gguf -O qwen2-1_5b-instruct-q4_0.gguf

# Qwen2 7B Instruct (Optional - most capable, requires more memory)
wget "https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/resolve/main/qwen2-7b-instruct-q4_0.gguf" -O qwen2-7b-instruct-q4_0.gguf
```

Place all downloaded model files in the project root directory.

6. **Process the documents for use in similarity search:**
```bash
python src/data_preprocess.py
```

## Usage

### Interactive RAG Pipeline

Run the interactive question-answering system:
```bash
# Standard mode
python src/main.py

# Debug mode (shows performance metrics)
python src/main.py debug
```

In interactive mode:
- Type any question to get an AI-powered answer
- Type `help` to see available commands
- Type `quit` or `exit` to exit the program

### Switching LLM Models

To test different models for Part 3 experiments, edit `src/llm_generation.py`:
```python
# In the __init__ method, uncomment the model you want to use along with its recommended_ctx:

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
```

The model will automatically use its recommended context window size. You can override this by passing `n_ctx` when initializing: `LLMGenerator(n_ctx=2048)`.

### Debug Mode Performance Metrics

When running in debug mode (`python src/main.py debug`), you'll see detailed performance breakdowns for each component of the RAG pipeline, including query encoding, vector search, document retrieval, prompt augmentation, and LLM generation times.

## Project Structure
```
rag-project/
├── src/
│   ├── main.py              # Main RAG pipeline and interactive loop
│   ├── encode.py            # Query encoding with BGE model
│   ├── vector_db.py         # FAISS-based vector database
│   ├── llm_generation.py    # LLM generation with llama.cpp
│   └── data_preprocess.py   # Document preprocessing
├── data/
│   ├── raw/                 # Raw datasets (documents.json, queries.json)
│   └── processed/           # Preprocessed documents and embeddings
├── requirements.txt         # Python dependencies
└── README.md               # This file
```