# Importação de módulos necessários para processamento de linguagem natural
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredXMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Caminho para onde os dados processados serão salvos
DB_FAISS_PATH = "datavector/db_faiss"
DATA_PATH = "dataset/"

# Carregamento dos dados de um arquivo XML não estruturado
# loader = UnstructuredXMLLoader("data/ptwiki-20230920-pages-articles.xml")
# loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
loader = DirectoryLoader(
    DATA_PATH,
    glob="*.txt",
    loader_cls=TextLoader
)

docs = loader.load()  # minimizar o problema  ctx512

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
text_chunks = text_splitter.split_documents(docs)

#print(f"Split blog post into {len(all_splits)} sub-documents.")
# Impressão do número de partes resultantes após a divisão do texto
print(len(text_chunks))

# Geração de embeddings usando o modelo Hugging Face 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2')

# Conversão dos pedaços de texto em embeddings e criação de um índice de pesquisa FAISS
docsearch = FAISS.from_documents(text_chunks, embeddings)

# Salvamento dos dados processados, incluindo o índice FAISS, em um diretório específico
docsearch.save_local(DB_FAISS_PATH)
