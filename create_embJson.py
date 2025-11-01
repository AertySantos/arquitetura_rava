import os
import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Caminhos
file_path = './umls/beck/validation_wo_integrated.json'
DB_FAISS_PATH = 'datavector/db_faiss'

# === 1. Carregar o JSON ===
print("Carregando dados JSON...")
data = json.loads(Path(file_path).read_text())

# === 2. Converter JSON em Documentos ===
print("Convertendo para objetos Document...")
docs = []
for item in data:
    term = item.get("term", "")
    context = item.get("context", "")
    cuis = item.get("cuis", [])

    # cada CUI vira um documento separado (importante p/ busca precisa)
    for cui in cuis:
        docs.append(
            Document(
                page_content=term,
                metadata={
                    "CUI": cui,
                    "context": context,
                    "source": "beck"
                }
            )
        )

print(f"Foram criados {len(docs)} documentos com metadados CUI.")

# === 3. Divisão dos textos (opcional) ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=128,  # pode ser menor, pois termos são curtos
    chunk_overlap=0
)
text_chunks = text_splitter.split_documents(docs)

# === 4. Geração dos embeddings ===
print("Gerando embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name='intfloat/multilingual-e5-base',
    encode_kwargs={"normalize_embeddings": True}
)

# === 5. Criação do índice FAISS ===
print("Criando e salvando FAISS...")
docsearch = FAISS.from_documents(text_chunks, embeddings)
docsearch.save_local(DB_FAISS_PATH)

print(f"✅ Banco vetorial salvo em: {DB_FAISS_PATH}")


