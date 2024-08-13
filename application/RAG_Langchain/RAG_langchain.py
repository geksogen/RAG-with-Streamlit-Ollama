from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import WebBaseLoader

model = ChatOllama(model="orca-mini:3b")
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
#docs = PyPDFLoader(file_path="test_doc.pdf").load()
#docs = CSVLoader(file_path="gravicapa.csv").load()
docs = WebBaseLoader("https://ru.wikipedia.org/wiki/%D0%93%D1%80%D0%B0%D0%B2%D0%B8%D1%86%D0%B0%D0%BF%D0%BF%D0%B0").load()

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

chain = ({
            "context" : retriever,
            "question" : RunnablePassthrough()
                       }
                        | prompt
                        | model
                        | StrOutputParser()
                       )

if not chain:
    print("Please ingest a PDF file first.")
else:
    print(chain.invoke(query))

