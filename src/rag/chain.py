from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from src.agents.base import create_llm


SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based on the provided context.
If the context doesn't contain relevant information, say so honestly.

Context:
{context}"""


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(vectorstore):
    """Create a RAG chain from a vectorstore using LCEL.

    Expects input dict: {"input": str, "chat_history": list}
    Returns: str (the answer)
    """
    llm = create_llm(temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # retriever receives the "input" key, context is formatted and passed along
    chain = (
        RunnableParallel(
            context=lambda x: _format_docs(retriever.invoke(x["input"])),
            input=lambda x: x["input"],
            chat_history=lambda x: x["chat_history"],
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
