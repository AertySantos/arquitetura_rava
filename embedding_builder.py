from typing_extensions import TypedDict, List
from langgraph.graph import StateGraph, START
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain

from qwen_llm import QwenLLM


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class Embedding_builder:
    def __init__(self,
                 db_path: str = "datavector/db_faiss",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cuda:1"):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.device = device
        self.chat_history = []

        self._load_embeddings()
        self._load_vector_db()
        self._load_model()
        self._build_chain()

    def _load_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            encode_kwargs={"normalize_embeddings": True}
        )

    def _load_vector_db(self):
        self.vector_db = FAISS.load_local(
            self.db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def _load_model(self):
        self.llm = QwenLLM(device=self.device)

    def _build_chain(self):

        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Use three sentences maximum and keep the answer as concise as possible.

        {context}

        Question: {question}

        Helpful Answer:"""
        custom_rag_prompt = PromptTemplate.from_template(template)

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(
                search_kwargs={"k": 10, "score_threshold": 0.7}
            ),
            combine_docs_chain_kwargs={"prompt": custom_rag_prompt}
        )

    def retrieve(self, state: State) -> dict:
        docs = self.vector_db.similarity_search(state["question"])
        return {"context": docs}

    def generate(self, state: State) -> dict:
        context_str = "\n\n".join(doc.page_content for doc in state["context"])
        messages = [
            {"role": "user", "content": f"{state['question']}\n\n{context_str}"}]
        response = self.llm.invoke(messages)
        return {"answer": response}

    def build_graph(self):
        builder = StateGraph(State)
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("generate", self.generate)
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        graph = builder.compile()
        return graph

    def run_graph(self, question: str):
        graph = self.build_graph()
        state = {"question": question, "context": [], "answer": ""}
        result = graph.invoke(state)

        source = next((doc.metadata["source"]
                       for doc in result['context'] if "source" in doc.metadata), None)

        return {
            "answer": result["answer"],
            "source": source
        }


# Exemplo de uso:
if __name__ == "__main__":
    bot = Embedding_builder()
    resposta = bot.run_graph(
        "qual o Assunto na sentença de numero 0000028-28.2018.8.26.0069?")
    print(f"Resposta: {resposta['answer']}")
    print(f"Fonte: {resposta['source']}")
