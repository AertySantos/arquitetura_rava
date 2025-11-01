from typing_extensions import TypedDict, List
from langgraph.graph import StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import CrossEncoder
from qwen_llm import QwenLLM
import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class Embedding_builder:
    def __init__(self,
                 db_path: str = "datavector/db_faiss",
                 embedding_model: str = "intfloat/multilingual-e5-base",
                 rerank_model: str = "mixedbread-ai/mxbai-rerank-base-v1",
                 device: str = "cuda:1",
                 use_hybrid_score: bool = True):
        """
        use_hybrid_score: combina FAISS + reranker (True) ou usa apenas o reranker (False)
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.device = device
        self.use_hybrid_score = use_hybrid_score
        self.chat_history = []

        # Inicializa componentes
        self._load_embeddings()
        self._load_vector_db()
        self._load_model()
        self._load_reranker()
        self._build_chain()

    # =========================================================
    # Carregamento de componentes
    # =========================================================
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

    def _load_reranker(self):
        print(f"🔁 Carregando modelo de rerank: {self.rerank_model}")
        self.reranker = CrossEncoder(self.rerank_model, device=self.device)

    # =========================================================
    # Prompt e cadeia
    # =========================================================
    def _build_chain(self):
        template = """Utilize o contexto abaixo para responder à pergunta.  
        Se a resposta estiver fora do escopo da Question, informe apenas que não há dados disponíveis sobre o assunto.
        Mantenha a resposta objetiva, com até três frases.

        {context}

        Question: {question}

        Helpful Answer:"""
        custom_rag_prompt = PromptTemplate.from_template(template)

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(
                search_kwargs={"k": 10, "fetch_k": 50, "score_threshold": 0.6}
            ),
            combine_docs_chain_kwargs={"prompt": custom_rag_prompt}
        )

    # =========================================================
    # Recuperação com Rerank e Híbrido
    # =========================================================
    def retrieve(self, state: State) -> dict:
        question = state["question"]

        # 1️⃣ Busca documentos pelo FAISS
        docs_with_scores = self.vector_db.similarity_search_with_score(question, k=50)

        # 2️⃣ Prepara pares (query, doc)
        pairs = [(question, doc.page_content) for doc, _ in docs_with_scores]

        # 3️⃣ Calcula relevância com reranker
        rerank_scores = self.reranker.predict(pairs)

        reranked = []
        for (doc, faiss_score), rerank_score in zip(docs_with_scores, rerank_scores):
            # normaliza FAISS (distância menor = mais próximo)
            faiss_sim = 1 - faiss_score  
            if self.use_hybrid_score:
                # híbrido: média dos dois scores
                final_score = 0.5 * faiss_sim + 0.5 * rerank_score
            else:
                # usa só o reranker
                final_score = rerank_score
            doc.metadata["similarity"] = round(float(faiss_sim) * 100, 2)
            doc.metadata["rerank_score"] = round(float(rerank_score), 3)
            doc.metadata["final_score"] = round(float(final_score), 3)
            reranked.append((doc, final_score))

        # 4️⃣ Ordena e pega top 5 reranqueados
        reranked.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in reranked[:5]]

        return {"context": top_docs}

    # =========================================================
    # Geração de resposta
    # =========================================================
       # =========================================================
    # Geração de resposta (com fontes e peso dinâmico 3× para docs relevantes)
    # =========================================================
    def generate(self, state: State) -> dict:
        if not state["context"]:
            return {"answer": "Nenhum documento relevante encontrado.", "context": state["context"]}

        # Ordena os documentos pelo score e pega os 3 melhores
        top_docs = sorted(state["context"], key=lambda d: float(d.metadata.get("final_score", 0)), reverse=True)[:3]

        context_parts = []
        for i, doc in enumerate(top_docs, start=1):
            sim = float(doc.metadata.get("final_score", 0))
            source = doc.metadata.get("source", "desconhecida")
            text = doc.page_content.strip()
            context_parts.append(f"--- Documento {i} [Fonte: {source} | Score: {sim:.3f}] ---\n{text}\n")

        context_str = "\n\n".join(context_parts)

        messages = [{"role": "user", "content": f"{state['question']}\n\n{context_str}"}]
        response = self.llm.invoke(messages)

        # Fonte do documento mais relevante
        doc_mais_relevante = top_docs[0]
        fonte_top = doc_mais_relevante.metadata.get("source", "desconhecida")

        # Retorna o dict completo esperado pelo StateGraph
        return {
            "answer": f"{response}\n\n📚 Fonte mais relevante: {fonte_top}",
            "context": state["context"],  # mantém o contexto original
            "question": state["question"]  # mantém a pergunta original
        }

    # =========================================================
    # Construção e execução do grafo
    # =========================================================
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

        # Seleciona o documento mais relevante
        if result["context"]:
            doc = result["context"][0]
            source = doc.metadata.get("source")
            similarity = doc.metadata.get("similarity")
            rerank_score = doc.metadata.get("rerank_score")
            final_score = doc.metadata.get("final_score")
            cui = doc.metadata.get("CUI", "desconhecido")
        else:
            source = similarity = rerank_score = final_score = cui = None

        return {
            "answer": result["answer"],
            "source": source,
            "similarity": similarity,
            "rerank_score": rerank_score,
            "final_score": final_score,
            "CUI": cui
        }


# =========================================================
# Exemplo de uso
# =========================================================
if __name__ == "__main__":
    bot = Embedding_builder(
        db_path="datavector/db_faiss",
        embedding_model="intfloat/multilingual-e5-base",
        rerank_model="mixedbread-ai/mxbai-rerank-base-v1",  # multilíngue
        device="cuda:1",
        use_hybrid_score=True  # True = combina FAISS + Rerank
    )

    pergunta = "qual o codigo de quimioterapia adyuvante?"
    resposta = bot.run_graph(pergunta)

    print("\n📘 Resultado:")
    print(f"Pergunta: {pergunta}")
    print(f"Resposta: {resposta['answer']}")
    print(f"Fonte: {resposta['source']}")
    print(f"Similaridade (FAISS): {resposta['similarity']}%")
    print(f"Relevância (Reranker): {resposta['rerank_score']}")
    print(f"Score final combinado: {resposta['final_score']}")
    print(f"CUI identificado: {resposta['CUI']}")  



# =========================================================
# Função de avaliação
# =========================================================
#def evaluate_benchmark(bot: Embedding_builder, benchmark_path: str = "./umls/beck/validation_wo_integrated.json", output_csv: str = "outputs/eval_results.csv"):
    """
    Loop de avaliação: percorre o benchmark JSON, roda o grafo e calcula métricas.
    """
    # 1️ Carrega o benchmark
#    with open(benchmark_path, "r", encoding="utf-8") as f:
#        dataset = json.load(f)

#    results = []

    # 2️ Loop principal
#    for item in tqdm(dataset, desc=" Avaliando termos do benchmark"):
#        termo = item["term"]
#        gold_cuis = item["cuis"]

#        try:
#            result = bot.run_graph(termo)
#            pred_cui = result["CUI"]

#            acerto = pred_cui in gold_cuis
#            results.append({
 #               "term": termo,
 #               "gold_cuis": ",".join(gold_cuis),
 #               "pred_cui": pred_cui,
 #               "acerto": int(acerto),
 #               "similarity": result["similarity"],
 #               "rerank_score": result["rerank_score"],
 #               "final_score": result["final_score"]
 #           })

 #       except Exception as e:
 #           results.append({
 #               "term": termo,
 #               "gold_cuis": ",".join(gold_cuis),
 #               "pred_cui": "ERROR",
 #               "acerto": 0,
 #               "similarity": 0,
 #               "rerank_score": 0,
 #               "final_score": 0
 #           })

    # 3️ Cria dataframe e salva
 #   df = pd.DataFrame(results)
 #   df.to_csv(output_csv, index=False)

    # 4️ Calcula métricas
 #   y_true = [1] * len(df)  # todos os golds são válidos
 #   y_pred = df["acerto"].tolist()

 #   acc = accuracy_score(y_true, y_pred)
 #   prec = precision_score(y_true, y_pred)
 #   rec = recall_score(y_true, y_pred)
 #   f1 = f1_score(y_true, y_pred)

 #   print("\n📊 Resultados do Benchmark:")
 #   print(f"Accuracy:  {acc:.3f}")
 #   print(f"Precision: {prec:.3f}")
 #   print(f"Recall:    {rec:.3f}")
 #   print(f"F1-score:  {f1:.3f}")
 #   print("=============================")

 #   return df


# =========================================================
# Execução do benchmark
# =========================================================
#if __name__ == "__main__":
#    bot = Embedding_builder(
#        db_path="datavector/db_faiss",
#        embedding_model="intfloat/multilingual-e5-base",
#        rerank_model="mixedbread-ai/mxbai-rerank-base-v1",
#        device="cuda:1",
#        use_hybrid_score=True
#    )

#    df_resultados = evaluate_benchmark(bot, "./umls/beck/validation_wo_integrated.json", "outputs/eval_results.csv")



