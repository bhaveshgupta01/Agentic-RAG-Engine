import os
# Identify your scraper to avoid being blocked
os.environ["USER_AGENT"] = "myagent/1.0"

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. LOAD DATA
# We will "teach" the agent about Agents by loading a technical blog post
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/", # The famous ReAct paper blog
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# 2. SPLIT DATA (Chunking)
# LLMs can't read whole books at once. We cut text into 500-character chunks.
# "overlap=50" ensures we don't cut a sentence in half awkwardly.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# 3. EMBED & STORE (The Vector DB)
# This sends chunks to Ollama -> converts to vectors -> saves to ./chroma_db
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    persist_directory="./chroma_db" 
)

print(f"Successfully saved {len(doc_splits)} chunks to ChromaDB!")