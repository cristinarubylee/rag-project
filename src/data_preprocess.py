import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

def main():
    # Get the project structure
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent  # Go up one level from src/
    
    input_path = project_root / "data" / "raw" / "documents.json"
    output_dir = project_root / "data" / "processed"
    output_path = output_dir / "preprocessed_documents.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading documents...")
    print(f"Source: {input_path}")
    with input_path.open("r", encoding="utf-8") as f:
        documents = json.load(f)
    print(f"Loaded {len(documents)} documents.\n")

    print("Loading encoder model (BAAI/bge-base-en)...")
    model = SentenceTransformer("BAAI/bge-base-en")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device.upper()}.\n")

    print("Encoding embeddings...")
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print("Finished encoding.\n")

    print("Saving processed documents...")
    processed = [
        {"id": doc["id"],
        "text": doc["text"],
        "embedding": embeddings[i].tolist()}
        for i, doc in enumerate(documents)
    ]
    with open(output_path, "w", encoding="utf-8") as f: json.dump(processed, f)

    print(f"Done! Saved {len(processed)} embeddings to {output_path}")

if __name__ == "__main__":
    main()
