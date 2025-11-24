import torch
from sentence_transformers import SentenceTransformer
import numpy as np

class QueryEncoder:
    def __init__(self, device=None):
        """
        Encoder to encode text queries into embedding vectors
        """
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def encode(self, text: str):
        """
        Encodes a string into a 768-dimensional vector.
        """
        emb = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return emb[0]
