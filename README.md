# 🏗️ Nexus-AI Platform

An enterprise-grade, citation-verified AI platform for Betsson Group. Built with FastAPI, LangGraph, Hybrid RAG, and MCP.

## ✨ Features (Ki Ki Korte Pare)

- **Hybrid RAG Engine**: BM25 + ChromaDB RRF fusion diye uttor khuje ber kore.
- **Citation-Verified**: Prottek uttor e `[Source N]` cite kore, tai bhul information dewar chance kom.
- **Safety Harness**: Faithfulness score 0.85 er niche holei system uttor dite refuse kore.
- **Distress Protocol**: Kono user mental health er kotha bolle LLM bypass kore direct help resource dey.
- **MCP Integration**: Player limits ba self-exclusion data fast khuje ber kore.
- **EU AI Act Ready**: `/compliance` endpoint diye legal requirements map kora ache.

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic
- **AI/LLM**: LangGraph, LangChain, Ollama (qwen2.5:7b)
- **Database**: ChromaDB (Vector), BM25 (Keyword)
- **Tooling**: Model Context Protocol (MCP) 2.x

## 🚀 How to Run (Kivabe Chalaben)

Ei project ta chalate hole apnar PC te Python 3.12+ ar Ollama installed thakte hobe.

1. **Install Dependencies:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/nexus-ai.git
   cd nexus-ai
   python -m venv .venv
   source .venv/Scripts/activate  # Windows er jonno
   pip install -e ".[dev]"


**Setup Local AI Model:**

bash
ollama pull qwen2.5:7b
ollama serve
Configure Environment:
**Copy .env.example to .env and set your preferences:**

**Local LLM**
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
MCP_ENABLED=true

**Seed Vector Database & Run Server:**

python scripts/seed_chroma.py
uvicorn main:app --reload --port 8000
**Access the API:**
Open your browser and go to: http://localhost:8000/docs