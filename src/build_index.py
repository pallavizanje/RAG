from src.ingest import load_documents
from src.chunking import create_chunks
from src.vector_store import create_vector_store


def build_index():

    print("========== RAG INDEXING PIPELINE ==========")

    print("\n[1] Loading documents...")

    documents = load_documents()

    print("\n[2] Creating chunks...")

    chunks = create_chunks(documents)

    print("\n[3] Creating vector database...")

    create_vector_store(chunks)

    print("\n========== INDEXING COMPLETED ==========")


if __name__ == "__main__":
    build_index()