from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

from qwen_llm import QwenLLM  # Importa sua classe customizada


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
            encode_kwargs={"normalize_embeddings": True})

    def _load_vector_db(self):
        self.vector_db = FAISS.load_local(
            self.db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def _load_model(self):
        self.llm = QwenLLM(device=self.device)

    def _build_chain(self):
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(
                search_kwargs={"k": 8, "score_threshold": 0.7})
        )

    def ask(self, question: str) -> str:
        try:
            result = self.qa_chain.invoke({
                "question": question,
                "chat_history": self.chat_history
            })
            self.chat_history.append((question, result.get("answer", "")))
            # for doc in result['source_documents']:
            #    print(f"- {doc.page_content}")
            return result.get("answer", "")

        except Exception as e:
            # Aqui você pode logar o erro se quiser
            # print(f"Erro ao processar pergunta: {e}")
            return ""

    def run_cli(self, msg):
        return self.ask(msg)


# Exemplo de uso:
if __name__ == "__main__":
    bot = Embedding_builder()
    print(bot.run_cli("nome do reu ?"))
