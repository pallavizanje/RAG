from src.llm import get_llm


def main():

    llm = get_llm()

    question = "Explain embeddings in simple words."

    response = llm.invoke(question)

    print("\n========================================")
    print("              LLM TEST")
    print("========================================")

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(response.content)


if __name__ == "__main__":
    main()