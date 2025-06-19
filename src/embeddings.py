import numpy as np
from fastembed import TextEmbedding
import os

class EmbeddingModel:
    def __init__(self):
        self.embeddings_model = TextEmbedding(
            model_name=os.getenv('EMBEDDINGS_MODEL'),
            cache_dir=os.getenv('MODELS_DIR')
        )

    def embed(self, text:str) -> list[float]:
        vec = list(self.embeddings_model.embed(text))[0]
        return self.normalize(vec)
    
    @staticmethod
    def normalize(vec: list[float]) -> list[float]:
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()
