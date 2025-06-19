from fastembed import TextEmbedding
import numpy as np
import os
import pymupdf
from dotenv import load_dotenv
import chromadb

def generate_chunks(text: str, chunk_size: int, overlap: int):
    start = 0
    #chunks = []
    while start < len(text):
        end = start + chunk_size
        #chunks.append(text[start:end])
        yield text[start:end]
        start += chunk_size - overlap
    #return chunks

def normalize_embeddings(vec: list[float]) -> list[float]:
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()

load_dotenv(override=True)

chroma_client = chromadb.PersistentClient(
    path=os.getenv('CHROMADB_DIR')
)

embedding_model = TextEmbedding(
    model_name=os.getenv('EMBEDDINGS_MODEL'),
    cache_dir=os.getenv('MODELS_DIR')
)

collection = chroma_client.get_or_create_collection(
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
'''
index = 0
docs_path = os.getenv('DOCUMENTS_DIR')
for doc in os.listdir(docs_path):
    title = os.path.splitext(doc)[0]
    with pymupdf.open(os.path.join(docs_path, doc)) as document:
        for page_num in range(len(document)):
            page_text = document.load_page(page_num).get_text()
            # empty page
            if not page_text.strip():
                continue
            #chunks = generate_chunks(text=page_text,chunk_size=300,overlap=100)
            for chunk in generate_chunks(text=page_text,chunk_size=300,overlap=100):
                chunk_embeddings = normalize_embeddings(list(embedding_model.embed(chunk))[0])
                collection.add(
                    ids=[str(index)],
                    embeddings=[chunk_embeddings],
                    documents=[chunk],
                    metadatas=[{
                        "title": title,
                        "page": page_num + 1
                    }]
                )
                index += 1'''
print(collection.count())

query = "aveda products ingredients vegetarian and vegan"
query_emb = normalize_embeddings(list(embedding_model.embed(query))[0])
results = collection.query(
    query_embeddings=[query_emb],
    n_results=1,
    include=['documents']
)

print(results)
for doc, meta, score in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
    print(f"trovato a pagina:{meta['page']}")
    print(f"titolo:{meta['title']}")
    print(doc)