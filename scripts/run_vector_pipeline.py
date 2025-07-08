# run_vector_pipeline.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_ingestion.pdf_loader import extract_text_from_pdfs, chunk_text
from data_ingestion.vector_store import load_embedding_model, get_embeddings
from data_ingestion.vector_store import save_vector_store
import numpy as np
import os


# Define paths
pdf_folder = "docs"
index_path = "faiss_store/faiss_index.index"
metadata_path = "faiss_store/chunks.pkl"

# Ensure output directory exists
os.makedirs(os.path.dirname(index_path), exist_ok=True)

# Ingest and process PDFs
text = extract_text_from_pdfs(pdf_folder)
chunks = chunk_text(text)

# Generate embeddings
model = load_embedding_model()
embeddings = get_embeddings(model, chunks)
embeddings = np.array(embeddings)

# Save vector store
save_vector_store(
    embeddings, chunks,
    index_path=index_path,
    metadata_path=metadata_path
)

print("✅ Vector store created and saved at:")
print(f"   → Index: {index_path}")
print(f"   → Metadata: {metadata_path}")
