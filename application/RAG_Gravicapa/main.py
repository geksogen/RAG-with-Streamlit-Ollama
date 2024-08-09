####
# pull mxbai-embed-large models
####

import ollama
import chromadb

documents = [
  "Gravicapa a fictional device, a key element of the plot of the Soviet science fiction feature film Kin-dza-dza!",
  "The gravitsappa is an egg-shaped metal body about 10-15 cm in size, consisting of two moving parts that easily rotate relative to each other. The lower part of the device is made of yellow metal"
]

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
