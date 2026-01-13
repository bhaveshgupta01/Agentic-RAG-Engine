import os
import shutil
import ollama 
from typing import List

# --- LOADERS ---
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader, 
    CSVLoader
)
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- CONFIGURATION ---
DATA_FOLDER = "./my_data"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "local-os-rag"
VISION_MODEL = "llava"       # Make sure you have this pulled (ollama pull llava)
RESET_DB = False             # Toggle to wipe DB

def get_existing_files(vectorstore):
    """Check DB for existing sources to avoid duplicate work."""
    if not os.path.exists(CHROMA_PATH):
        return set()
    try:
        existing_data = vectorstore._collection.get(include=["metadatas"])
        existing_files = set()
        for meta in existing_data["metadatas"]:
            if meta and "source" in meta:
                existing_files.add(meta["source"])
        return existing_files
    except:
        return set()

def process_image(file_path):
    """Uses Vision LLM to caption images."""
    print(f"   👀 Vision-Scanning: {os.path.basename(file_path)}")
    prompt = "Transcribe any text in this image exactly. Then, describe the visual details, layout, and purpose of the image."
    
    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{'role': 'user', 'content': prompt, 'images': [file_path]}]
        )
        description = response['message']['content']
        return Document(
            page_content=description, 
            metadata={"source": file_path, "type": "image"}
        )
    except Exception as e:
        print(f"   ❌ Image Error: {e}")
        return None

def ingest_files():
    # 1. Database Setup
    if RESET_DB and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("✨ Database wiped.")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory=CHROMA_PATH
    )
    
    existing_files = get_existing_files(vectorstore)
    print(f"🔍 Skipping {len(existing_files)} files already in DB...")

    new_documents = []
    
    print(f"--- SCANNING {DATA_FOLDER} ---")
    
    # 2. Universal File Walker
    for root, dirs, files in os.walk(DATA_FOLDER):
        for file in files:
            full_path = os.path.join(root, file)
            
            if full_path in existing_files:
                continue
                
            file_ext = os.path.splitext(file)[1].lower()
            doc_loader = None
            
            # --- SELECT LOADER BASED ON EXTENSION ---
            try:
                # A. PDFs
                if file_ext == ".pdf":
                    print(f"   📚 Loading PDF: {file}")
                    doc_loader = PyPDFLoader(full_path)
                
                # B. Word Docs
                elif file_ext == ".docx":
                    print(f"   📝 Loading Word: {file}")
                    doc_loader = Docx2txtLoader(full_path)
                
                # C. CSV / Excel (Simple)
                elif file_ext == ".csv":
                    print(f"   📊 Loading CSV: {file}")
                    doc_loader = CSVLoader(full_path)

                # D. Text / Code
                elif file_ext in [".txt", ".md", ".py", ".json"]:
                    print(f"   📄 Loading Text: {file}")
                    doc_loader = TextLoader(full_path)
                
                # E. Images (Custom Logic)
                elif file_ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                    img_doc = process_image(full_path)
                    if img_doc:
                        new_documents.append(img_doc)
                    continue # Skip the loader block below for images

                # F. Unknown
                else:
                    # Skip weird system files
                    continue

                # Execute Loader (For A, B, C, D)
                if doc_loader:
                    loaded_docs = doc_loader.load()
                    # Add metadata
                    for d in loaded_docs:
                        d.metadata["source"] = full_path
                        d.metadata["type"] = file_ext
                    new_documents.extend(loaded_docs)

            except Exception as e:
                print(f"   ❌ Failed to load {file}: {e}")

    # 3. Batch Process
    if not new_documents:
        print("✅ No new content found.")
        return

    print(f"\n--- SPLITTING & STORING {len(new_documents)} DOCUMENTS ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    splits = text_splitter.split_documents(new_documents)
    
    # Add to Chroma (Batching happens automatically mostly, but for huge sets consider batching manually)
    vectorstore.add_documents(splits)
    print(f"🎉 Success! Added {len(splits)} chunks to the database.")

if __name__ == "__main__":
    ingest_files()

#ollama realtime logs   : tail -f ~/.ollama/logs/server.log
#ollama logs            : cat ~/.ollama/logs/server.log
# detele database folder : rm -rf ./chroma_db