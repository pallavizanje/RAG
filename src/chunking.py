from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingest import load_documents


def create_chunks(documents):

    langchain_documents = [
        Document(
            page_content=document["text"],
            metadata=document["metadata"],
        )
        for document in documents
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(langchain_documents)

    print(f"Original pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    return chunks

def analyze_chunks(chunks):

    lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    print("\n========== CHUNK ANALYSIS ==========")

    print(f"Total chunks: {len(chunks)}")

    print(f"Smallest chunk: {min(lengths)} characters")

    print(f"Largest chunk: {max(lengths)} characters")

    print(
        f"Average chunk: "
        f"{sum(lengths) / len(lengths):.2f} characters"
    )

if __name__ == "__main__":

    documents = load_documents()

    chunks = create_chunks(documents)

    analyze_chunks(chunks)

    print("\n========== FIRST CHUNK ==========\n")

    print(chunks[0].page_content)

    print("\nMetadata:")

    print(chunks[0].metadata)