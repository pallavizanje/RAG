from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings import get_embedding_model


VECTOR_DB_DIR = Path("vector_db")

COLLECTION_NAME = "rag_documents"


def create_vector_store(chunks):

    print("\nLoading embedding model...")

    embeddings = get_embedding_model()

    print("Creating Chroma vector database...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
        collection_name=COLLECTION_NAME,
    )

    print("Chroma vector database created successfully.")

    return vector_store