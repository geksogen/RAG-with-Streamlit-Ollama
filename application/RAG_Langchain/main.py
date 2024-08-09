from langchain_community.document_loaders import WebBaseLoader
import os

os.environ['USER_AGENT'] = 'Chrome/127.0.0.0 Safari/537.36'


loader = WebBaseLoader("https://ru.wikipedia.org/wiki/%D0%93%D1%80%D0%B0%D0%B2%D0%B8%D1%86%D0%B0%D0%BF%D0%BF%D0%B0")
data = loader.load()
