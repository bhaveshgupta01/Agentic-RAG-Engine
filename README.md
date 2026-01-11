# 🧠 Agentic RAG Engine with Self-Correction

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green?logo=fastapi)
![Ollama](https://img.shields.io/badge/LLM-Mistral%20Local-purple)

A **Self-Correcting Retrieval Augmented Generation (CRAG)** system that autonomously grades its own retrieval quality. If local documents are insufficient, it automatically falls back to web search to ensure accuracy and prevent hallucinations.



## 🌟 Key Features

* **Self-Reflection:** Uses an LLM "Grader" to evaluate if retrieved documents are actually relevant.
* **Conditional Routing:** If the grader votes "No", the system routes the query to **DuckDuckGo Search** instead of hallucinating.
* **Memory Management:** Implements persistent conversation threads using `MemorySaver`.
* **Microservice Architecture:** Deployed as a REST API using **FastAPI** for easy frontend integration.

## 🏗️ Architecture

1.  **Retrieve:** Fetches top-k context from local ChromaDB.
2.  **Grade:** Mistral (via Ollama) scores the relevance of the documents.
3.  **Decide:**
    * *Pass:* -> Generate Answer.
    * *Fail:* -> Web Search -> Generate Answer.
4.  **Response:** Returns a grounded, citation-backed answer.

## 🚀 Quick Start

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) running locally (`ollama pull mistral`)

### Installation

```bash
# 1. Clone the repo
git clone [https://github.com/yourusername/agentic-rag-engine.git](https://github.com/yourusername/agentic-rag-engine.git)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ingest your documents (Place PDF/Text in a folder or edit ingest.py)
python ingest.py

# 4. Start the API Server
python server.py
