from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.document_loaders import WebBaseLoader

model = ChatOllama(model="tinyllama")
prompt = PromptTemplate.from_template(
            """
            [INST]<<SYS>> You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>> 
            Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

query = "what is gravicapa?"

# Load data
#docs = PyPDFLoader(file_path="gravicapa_1.pdf").load()
docs = WebBaseLoader("https://raw.githubusercontent.com/geksogen/RAG-with-Streamlit-Ollama/master/application/RAG_Langchain/gravicapa.md")
docs = docs.load()

# Split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)
chunks = filter_complex_metadata(chunks)

# Create Vectors and embedings
vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        'k': 3,
        'score_threshold': 0.5
        },
    )

# Create promt
chain = ({
            "context" : retriever,
            "question" : RunnablePassthrough()
                       }
                        | prompt
                        | model
                        | StrOutputParser()
                       )

# receive answer
if not chain:
    print("Please ingest a PDF file first.")
else:
    print(chain.invoke(query))  # receive answer

