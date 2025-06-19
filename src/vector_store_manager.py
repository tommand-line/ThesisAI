import os
import chromadb

class VectorStoreManager:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(
            path=os.getenv('CHROMADB_DIR')
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name='documents',
            metadata={
                "author": "tommand-line"
            },
            configuration={
                'hnsw': {
                    'space': 'cosine'
                }
            }
        )

    def add_chunk(self, chunk_id: str, chunk_embeddings: list[float], chunk_text: str, metadata: dict):
        self.collection.add(
                    ids=[chunk_id],
                    embeddings=[chunk_embeddings],
                    documents=[chunk_text],
                    metadatas=[metadata]
                )
        
    def query_collection(self, query_embeddings: list[float], top_k: int = 3):
        return self.collection.query(
            query_embeddings=[query_embeddings],
            n_results=top_k,
            include=['documents', 'metadatas']
        )

    def count_collection_items(self) -> int:
        return self.collection.count()