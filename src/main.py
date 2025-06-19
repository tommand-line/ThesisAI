import os
from dotenv import load_dotenv
from ingestion_manager import IngestionManager
from embeddings import EmbeddingModel
from vector_store_manager import VectorStoreManager

load_dotenv(override=True)

def main():
    ingestor_manager = IngestionManager()
    embedding_model = EmbeddingModel()
    vector_index = VectorStoreManager()

    #ingestor_manager.ingest_documents()

    print(f"Chunks: {ingestor_manager.vector_store.count_collection_items()}")

    query = "aveda products ingredients vegetarian and vegan"

    query_embeddings = embedding_model.embed(query)
    results = vector_index.query_collection(query_embeddings)

    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print()
        print(f"pagina reference: {meta['page']}")
        print(f"titolo reference: {meta['title']}")
        print(doc)


if __name__ == '__main__':
    main()