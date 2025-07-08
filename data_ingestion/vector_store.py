from sentence_transformers import SentenceTransformer
import faiss
import pickle

def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def get_embeddings(model, texts):
    return model.encode(texts, show_progress_bar=True)



def save_vector_store(embeddings, chunks, index_path, metadata_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

def load_vector_store(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks