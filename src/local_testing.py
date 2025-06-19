from fastembed import TextEmbedding
import numpy as np
import os
import pymupdf
from dotenv import load_dotenv
import uuid
import chromadb

def generate_chunks(text: str, chunk_size: int, overlap: int):
    start = 0
    chunks = []
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def normalize_embeddings(vec: list[float]) -> list[float]:
    vec = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.tolist()
    return (vec / norm).tolist()

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
docs_path = os.getenv('DOCUMENTS_DIR')
for doc in os.listdir(docs_path):
    title = os.path.splitext(doc)[0]
    with pymupdf.open(os.path.join(docs_path, doc)) as document:
        for page_num in range(len(document)):
            page_text = document.load_page(page_num).get_text()
            # empty page
            if not page_text.strip():
                continue
            chunks = generate_chunks(text=page_text,chunk_size=300,overlap=100)
            for idx, chunk in enumerate(chunks):
                chunk_embeddings = normalize_embeddings(list(embedding_model.embed(chunk))[0])
                collection.add(
                    ids=[str(page_num)+str(idx)],
                    embeddings=[chunk_embeddings],
                    documents=[chunk],
                    metadatas=[{
                        "title": title,
                        "page": page_num + 1
                    }]
                )'''
print(collection.count())

query = "what is encoder decoder architecture"
query_emb = normalize_embeddings(list(embedding_model.embed(query))[0])
results = collection.query(
    query_embeddings=[query_emb],
    n_results=1,
)

print(results)
for doc, meta, score in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
    print(f"trovato a pagina:{meta['page']}")

exit()

documents_folder = './documents'

text = ""
with pymupdf.open(documents_folder + '/doc.pdf') as doc:
    text = chr(12).join([page.get_text() for page in doc])
# write as a binary file to support non-ASCII characters
print(text)

def fixed_size_chunking(text, max_tokens):
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        current_tokens += len(word.split())
        if current_tokens <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_tokens = len(word.split())
    
    # Append the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

max_tokens = 100
embeddings_dim = 384

index = faiss.IndexFlatL2(embeddings_dim)
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir='./models')

chunks = fixed_size_chunking(text, max_tokens)
for i, chunk in enumerate(chunks):
    #print(f"Chunk {i+1}:\n{chunk}\n")
    embeddings = model.embed(chunk).astype('float32')
    index.app(embeddings.reshape(1, -1))

faiss.write_index(index, '/mnt/usb/ThesisAI/vector_database')
exit()

#embeddings_model = TextEmbedding('onnx-models/all-MiniLM-L6-v2-onnx')
sentences = [
    "This is an example sentence",
    "Each sentence is converted"
]

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir='./models')
embeddings = list(model.embed(sentences))
print(np.array(embeddings).shape)