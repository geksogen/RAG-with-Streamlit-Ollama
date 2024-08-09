from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings

# Load data
loader = WebBaseLoader("https://ru.wikipedia.org/wiki/%D0%93%D1%80%D0%B0%D0%B2%D0%B8%D1%86%D0%B0%D0%BF%D0%BF%D0%B0")
data = loader.load()

# Splitting data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Index data
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Add to ChromaDB vector store
from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(
    documents=all_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

question = "What are Gravicapa?"
docs = vectorstore.similarity_search(question)

print(docs[0])


