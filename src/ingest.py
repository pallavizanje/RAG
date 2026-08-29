from pathlib import Path

import pymupdf


DATA_DIR = Path("data/raw")


def load_documents():

    documents = []

    for pdf_file in DATA_DIR.glob("*.pdf"):

        print(f"Loading: {pdf_file}")

        pdf = pymupdf.open(pdf_file)

        for page_number, page in enumerate(pdf):

            text = page.get_text()

            if text.strip():

                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": str(pdf_file),
                            "page": page_number + 1,
                        },
                    }
                )

        pdf.close()

    print(f"\nTotal pages loaded: {len(documents)}")

    return documents


if __name__ == "__main__":

    docs = load_documents()

    if docs:

        print("\n========== FIRST DOCUMENT ==========\n")

        print(docs[0]["text"][:1000])

        print("\n========== METADATA ==========\n")

        print(docs[0]["metadata"])