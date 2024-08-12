from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter
import ollama
import chromadb


# Load data
loader = PDFPlumberLoader("test_doc.pdf")
docs = loader.load()


# Split data
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs)


client = chromadb.Client()
collection = client.create_collection(name="docs_1")

# store each document in a vector embedding database
for i, d in enumerate(doc_splits):
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


