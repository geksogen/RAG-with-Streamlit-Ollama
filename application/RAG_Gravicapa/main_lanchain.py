from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
import ollama
import chromadb

import json
import requests

# Load data
#loader = WebBaseLoader("https://ru.wikipedia.org/wiki/%D0%93%D1%80%D0%B0%D0%B2%D0%B8%D1%86%D0%B0%D0%BF%D0%BF%D0%B0")
#data = loader.load()
data = requests.get("https://ru.wikipedia.org/wiki/%D0%93%D1%80%D0%B0%D0%B2%D0%B8%D1%86%D0%B0%D0%BF%D0%BF%D0%B0").json()



# Splitting data
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#all_splits = text_splitter.split_documents(data)

splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_chunks = splitter.split_json(json_data=data)


documents = splitter.create_documents(texts=[data])

client = chromadb.Client()
collection = client.create_collection(name="docs_1")

# store each document in a vector embedding database
for i, d in enumerate(documents):
  response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[d]
  )

# an example prompt
prompt = "What is Gravicapa?"

# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
  prompt=prompt,
  model="mxbai-embed-large"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=1
)
data = results['documents'][0][0]

# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
  model="orca-mini:3b",
  prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print(output['response'])
