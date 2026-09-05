from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful assistant answering questions about the
provided documents.

Use ONLY the information provided in the context below.

If the answer cannot be found in the context, say:

"I don't know based on the provided documents."

Do not make up information.
Do not use outside knowledge.

Context:
--------------------
{context}
--------------------

Question:
{question}

Answer:
"""
)