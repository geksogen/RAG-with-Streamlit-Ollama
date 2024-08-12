from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


# Load data
loader = PDFPlumberLoader("test_doc.pdf")
docs = loader.load()


# Split data
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs)

# Create a vector store using Chroma DB, our chunked data from the URLs, and the nomic-embed-text embedding model
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings.ollama.OllamaEmbeddings(model='mxbai-embed-large'),
    )
retriever = vectorstore.as_retriever()

