from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. Connect to the DB
# Make sure this matches the path in your ingest.py
CHROMA_PATH = "./chroma_db"

print(f"--- INSPECTING DATABASE AT {CHROMA_PATH} ---")

try:
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        collection_name="local-os-rag", # MUST match the name in ingest.py
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
    )

    # 2. Get Stats
    # We can access the underlying collection object to run standard DB commands
    collection_count = vectorstore._collection.count()
    print(f"📊 Total Documents Found: {collection_count}")

    if collection_count == 0:
        print("⚠️  Database is empty! Did you run ingest.py?")
    else:
        # 3. Peek at the Data
        # Fetch the first 10 items to see what they look like
        print("\n--- PEEKING AT FIRST 10 DOCS ---")
        
        # .get() returns the raw data (ids, metadatas, documents)
        data = vectorstore._collection.get(limit=10)
        
        docs = data['documents']
        metadatas = data['metadatas']
        ids = data['ids']

        for i, (doc, meta, id_) in enumerate(zip(docs, metadatas, ids)):
            print(f"\n[Doc {i}] ID: {id_}")
            print(f"   📂 Source: {meta.get('source', 'Unknown')}")
            print(f"   🏷️  Type:   {meta.get('type', 'Unknown')}")

            # Print just the first 250 characters of content
            clean_content = doc.replace("\n", " ")[:250]
            print(f"   📄 Content: \"{clean_content}...\"")

except Exception as e:
    print(f"❌ Error inspecting DB: {e}")