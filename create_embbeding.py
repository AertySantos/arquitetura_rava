import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. CONFIGURAÇÃO ---
# Caminho para a pasta onde os vetores serão salvos
DB_FAISS_PATH = "datavector/db_faiss"
# Caminho para a pasta com suas sentenças em .txt

DATA_PATH = "testes/dataset"

# --- 2. CARREGAMENTO DOS DADOS ---
print("Carregando documentos...")
loader = DirectoryLoader(
    DATA_PATH,
    glob="*.txt",
    loader_cls=TextLoader,
    recursive=True,
    loader_kwargs={'encoding': 'utf-8'}
)

docs = loader.load()
print(f"Foram carregados {len(docs)} documento(s).")

# --- 3. DIVISÃO DO TEXTO (ESTRATÉGIA 1) ---
print("Dividindo os textos em pedaços (chunks)...")
text_splitter = RecursiveCharacterTextSplitter(
    # Tamanho do chunk: 1500 caracteres é um bom ponto de partida para textos densos.
    chunk_size=512,
    # Sobreposição: 300 caracteres é um valor robusto para não perder o contexto
    # de argumentos jurídicos que podem se estender por mais de um parágrafo.
    chunk_overlap=100,
    # Tenta usar separadores lógicos antes de forçar a quebra por tamanho.
    separators=["\n\n", "\n", ". ", ", ", " "],
    add_start_index=True
)
text_chunks = text_splitter.split_documents(docs)
print(f"Os documentos foram divididos em {len(text_chunks)} chunks.")

# --- 4. GERAÇÃO DE EMBEDDINGS ---
print("Gerando embeddings para os chunks...")
embeddings = HuggingFaceEmbeddings(
    model_name='intfloat/multilingual-e5-base',
    model_kwargs={'device': 'cuda:1'}, # Use 'cuda' se tiver uma GPU configurada
    encode_kwargs={"normalize_embeddings": True}
)

# --- 5. CRIAÇÃO E SALVAMENTO DO BANCO DE DADOS VETORIAL ---
print("Criando o banco de dados vetorial FAISS...")
docsearch = FAISS.from_documents(text_chunks, embeddings)

print(f"Salvando o banco de dados em '{DB_FAISS_PATH}'...")
docsearch.save_local(DB_FAISS_PATH)

print("\nProcesso concluído com sucesso!")
