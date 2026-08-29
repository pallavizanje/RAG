from langchain_chroma import Chroma

from src.embeddings import get_embedding_model


VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "rag_documents"


def get_vector_store():

    print("Loading embedding model...")

    embeddings = get_embedding_model()

    print("Loading Chroma database...")

    vector_store = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def search(query: str, k: int = 4):

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        query,
        k=k,
    )

    return results


if __name__ == "__main__":

    print("\n========================================")
    print("        RAG RETRIEVAL TEST")
    print("========================================")

    while True:

        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = search(query, k=4)

        print("\n========================================")
        print(f"QUERY: {query}")
        print("========================================")

        for index, (document, score) in enumerate(results, start=1):

            print(f"\n========== RESULT {index} ==========")

            print(f"Distance score: {score}")

            print(
                f"Source: "
                f"{document.metadata.get('source')}"
            )

            print(
                f"Page: "
                f"{document.metadata.get('page')}"
            )

            print("\nContent:")

            print(document.page_content)