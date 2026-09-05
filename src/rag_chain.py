from src.retriever import search
from src.llm import get_llm
from src.prompt import RAG_PROMPT


def build_context(results):
    context_parts = []

    for index, (document, score) in enumerate(results, start=1):

        source = document.metadata.get("source")
        page = document.metadata.get("page")

        context_parts.append(
            f"""
SOURCE {index}
File: {source}
Page: {page}

Content:
{document.page_content}
"""
        )

    return "\n".join(context_parts)


def ask_question(question: str, k: int = 4):

    print("\n[1] Searching Chroma...")
    results = search(question, k=k)

    print("\n========== RETRIEVED CONTEXT ==========")

    print(f"[2] Retrieved {len(results)} documents")

    for index, (document, score) in enumerate(results, start=1):
        print(f"\n--- RESULT {index} ---")
        print(f"Distance: {score:.4f}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Page: {document.metadata.get('page')}")
        print("Content:")
        print(document.page_content) 

    context = build_context(results)

    print("[3] Building RAG prompt...")

    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    print("[4] Calling Llama 3.2...")

    llm = get_llm()

    response = llm.invoke(prompt)

    return response.content, results


if __name__ == "__main__":

    print("\n========================================")
    print("             RAG QUESTION ANSWER")
    print("========================================")

    while True:

        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        answer, results = ask_question(question)

        print("\n========================================")
        print("                 ANSWER")
        print("========================================")

        print(answer)

        print("\n========================================")
        print("                SOURCES")
        print("========================================")

        for index, (document, score) in enumerate(results, start=1):

            print(
                f"{index}. "
                f"{document.metadata.get('source')} "
                f"- Page {document.metadata.get('page')} "
                f"(distance={score:.4f})"
            )