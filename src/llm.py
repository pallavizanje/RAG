from langchain_ollama import ChatOllama


MODEL_NAME = "llama3.2"


def get_llm():
    print(f"Loading LLM: {MODEL_NAME}")

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0,
    )

    return llm