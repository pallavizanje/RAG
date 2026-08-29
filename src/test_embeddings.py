from src.embeddings import get_embedding_model


def main():

    embeddings = get_embedding_model()

    text = "What is Retrieval Augmented Generation?"

    vector = embeddings.embed_query(text)

    print("\n========== EMBEDDING TEST ==========")

    print(f"Input text: {text}")

    print(f"Vector dimensions: {len(vector)}")

    print("\nFirst 10 values:")

    print(vector[:10])


if __name__ == "__main__":
    main()