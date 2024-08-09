from langchain_text_splitters import RecursiveJsonSplitter
import requests
import json

# This is a large nested json object and will be loaded as a python dict
json_data = requests.get("https://ru.wikipedia.org/wiki/%D0%93%D1%80%D0%B0%D0%B2%D0%B8%D1%86%D0%B0%D0%BF%D0%BF%D0%B0").json()

splitter = RecursiveJsonSplitter(max_chunk_size=300)
json_chunks = splitter.split_json(json_data=json_data)

docs = splitter.create_documents(texts=[json_data])

for doc in docs[:3]:
    print(doc)
