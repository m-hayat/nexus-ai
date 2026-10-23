# 🏗️ Nexus-AI Platform

An enterprise-grade, citation-verified AI platform for Betsson Group. Built with FastAPI, LangGraph, Hybrid RAG, and MCP (Model Context Protocol). 

This platform features a complete web-based chat interface, safety guardrails, and local/cloud LLM support.

## ✨ Key Features

- **Hybrid RAG Engine**: Combines BM25 (keyword) and ChromaDB (semantic) using Reciprocal Rank Fusion (RRF) to retrieve highly accurate policy contexts.
- **Citation-Verified Generation**: Every generated answer must include inline `[Source N]` citations mapped directly to the retrieved documents.
- **AI Safety Harness**: Uses an LLM-as-a-judge to score faithfulness. If the score falls below the threshold (0.85), the system safely refuses to answer to prevent hallucination.
- **Distress Protocol**: Detects signs of gambling addiction or user distress. Instantly bypasses the LLM and returns human support resources.
- **MCP Integration**: Persistent Model Context Protocol client for fast, sub-second player limit and self-exclusion database checks.
- **EU AI Act Ready**: Includes a `/compliance` endpoint mapping technical controls to Articles 13, 14, and 15.
- **Interactive Web UI**: A responsive, streaming HTML/JS chat interface to interact with the agent in real-time.

## 🛠️ Tech Stack

- **Backend API**: FastAPI, Uvicorn, Pydantic v2
- **AI Orchestration**: LangGraph, LangChain
- **LLM Engine**: Groq (Cloud - Llama 3.3 / Qwen 3.8), Ollama (Local - Qwen 2.5), DeepSeek, or OpenAI (Switchable via `.env`)
- **Databases**: ChromaDB (Vector), BM25 (Keyword)
- **Protocols**: Model Context Protocol (MCP) 2.x

## 🚀 How to Run

**Prerequisites:** 
- Python 3.12+
- [Ollama](https://ollama.com/) installed (for local LLM), OR a [Groq API Key](https://console.groq.com/) (for ultra-fast cloud LLM).

1. **Install Dependencies:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/nexus-ai.git
   cd nexus-ai
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows
   # source .venv/bin/activate    # On macOS/Linux
   pip install -e ".[dev]"


   Configure Environment:
Copy .env.example to .env.

bash
cp .env.example .env
Open .env and configure your LLM provider:

Option A: Groq (Recommended for Speed)

env
LLM_PROVIDER=groq
LLM_MODEL=qwen/qwen3.8-27b
GROQ_API_KEY=gsk_your_api_key_here
Option B: Ollama (Local)

env
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b


Seed the Vector Database:
This loads the Betsson policy documents (30+ sections) into ChromaDB.

bash
python scripts/seed_chroma.py
Start the Server:

bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
Access the Application:

Chat UI: Open http://localhost:8000/ui/

API Docs (Swagger): Open http://localhost:8000/docs

🧪 Testing & Evaluation
Run Unit Tests:

bash
pytest tests/ -v
Run Golden Dataset Evaluation (CI):

bash
python scripts/run_eval.py
Check EU AI Act Compliance:

bash
curl http://localhost:8000/compliance

📊 Evaluation Metrics (Golden Dataset)
Routing Accuracy: 100%

Citation Accuracy: 96.7%

Average Faithfulness: 0.99