# agents/retriever_agent.py

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os

# Load FAISS index + chunk metadata
def load_vector_store(
    index_path="faiss_store/faiss_index.index",
    metadata_path="faiss_store/chunks.pkl"
):
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("FAISS index or chunk file not found")

    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

# Embed the query
def embed_query(query: str, model: SentenceTransformer):
    return np.array([model.encode(query)])

# Perform retrieval
def retrieve_context(query: str, index, chunks, embed_model, k: int = 3) -> str:
    query_vec = embed_query(query, embed_model)
    _, I = index.search(query_vec, k)
    context = "\n\n".join([chunks[i] for i in I[0]])
    return context

if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_vector_store()

    query = "What are the policies available?"
    context = retrieve_context(query, index, chunks, model)

    print("Retrieved Context:\n")
    print(context[:1000])  # preview
