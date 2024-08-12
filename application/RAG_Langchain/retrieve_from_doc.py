from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("test_doc.pdf")
docs = loader.load()

# Check the number of pages
print("Number of pages in the PDF:",len(docs))

