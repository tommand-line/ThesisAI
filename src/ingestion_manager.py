import os
import pymupdf
from embeddings import EmbeddingModel
from vector_store_manager import VectorStoreManager
import re

class IngestionManager:
    def __init__(self):
        self.embeddings_model = EmbeddingModel()
        self.vector_store = VectorStoreManager()
        self.docs_path = os.getenv('DOCUMENTS_DIR')

    def generate_chunks(text: str, chunk_size: int, overlap: int):
        start = 0
        while start < len(text):
            yield text[start:start + chunk_size]
            start += chunk_size - overlap

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+',' ',text)
        return text.strip()

    def ingest_documents(self):
        index = 0
        for doc in os.listdir(self.docs_path):
            title = os.path.splitext(doc)[0]
            with pymupdf.open(os.path.join(self.docs_path, doc)) as document:
                for page_num in range(len(document)):
                    page_text = document.load_page(page_num).get_text()
                    cleaned_text = self.clean_text(page_text)
                    # empty page
                    if not page_text.strip():
                        continue
                    #chunks = generate_chunks(text=page_text,chunk_size=300,overlap=100)
                    for chunk in self.generate_chunks(text=cleaned_text,chunk_size=300,overlap=100):
                        chunk_embeddings = self.embeddings_model.embed(chunk)
                        self.vector_store.add_chunk(
                            chunk_id=index,
                            chunk_embeddings=chunk_embeddings,
                            chunk_text=chunk,
                            metadata={
                                'title': title,
                                'page_num': page_num + 1
                            }
                        )
                        index += 1