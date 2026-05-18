from src.rag.loader import load_file
from src.rag.splitter import split_documents
from src.rag.vectorstore import create_vectorstore, add_documents
from src.rag.chain import create_rag_chain


MAX_HISTORY_TURNS = 10


class RAGAgent:
    """RAG (Retrieval-Augmented Generation) agent.

    Workflow:
    1. Load documents from files
    2. Split into chunks
    3. Embed and store in Chroma vector DB
    4. On query: retrieve relevant chunks → generate answer

    Usage:
        agent = RAGAgent()
        agent.load_document("path/to/file.md")
        answer = agent.ask("What is this document about?")
    """

    def __init__(self):
        self.vectorstore = None
        self.chain = None
        self.chat_history: list = []

    def load_document(self, file_path: str, collection_name: str = "default"):
        """Load a document into the vector store.

        If a vectorstore already exists, appends to it (won't overwrite).
        """
        docs = load_file(file_path)
        chunks = split_documents(docs)

        if self.vectorstore is not None:
            # 已有向量库，追加文档
            add_documents(self.vectorstore, chunks)
        else:
            # 首次加载，创建新向量库
            self.vectorstore = create_vectorstore(chunks, collection_name)
            self.chain = create_rag_chain(self.vectorstore)

        return len(chunks)

    def ask(self, question: str) -> str:
        """Ask a question using RAG."""
        if not self.chain:
            return "No documents loaded. Please load a document first using load_document()."

        answer = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
        })

        self._append_history(question, answer)

        return answer

    def stream_ask(self, question: str):
        """Ask a question and yield answer tokens one by one."""
        if not self.chain:
            yield "No documents loaded. Please load a document first using load_document()."
            return

        full_answer = ""
        for token in self.chain.stream({
            "input": question,
            "chat_history": self.chat_history,
        }):
            full_answer += token
            yield token

        self._append_history(question, full_answer)

    def _append_history(self, question: str, answer: str):
        self.chat_history.append(("human", question))
        self.chat_history.append(("ai", answer))
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.chat_history) > max_messages:
            self.chat_history = self.chat_history[-max_messages:]

    def clear_history(self):
        self.chat_history.clear()
