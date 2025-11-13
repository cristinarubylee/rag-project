import json
import os
import torch
from sentence_transformers import SentenceTransformer

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from src/
    
    input_path = os.path.join(project_root, "data", "raw", "documents.json")
    output_dir = os.path.join(project_root, "data", "processed")
    output_path = os.path.join(output_dir, "preprocessed_documents.json")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load documents
    with open(input_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Load BGE encoder model
    model = SentenceTransformer("BAAI/bge-base-en")

    # Create embeddings
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Save as preprocessed_documents.json
    processed = [
        {"id": doc["id"], "text": doc["text"], "embedding": embeddings[i].tolist()}
        for i, doc in enumerate(documents)
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f)

    print(f"Done! Saved {len(processed)} embeddings to {output_path}")

if __name__ == "__main__":
    main()
