# 🤖 HRP AI Assistant & RAG Chatbot

An enterprise-grade, role-aware **Retrieval-Augmented Generation (RAG)** conversational assistant built for the **HRP (Human Resource Planning) Employee Management ERP System**.

The assistant empowers employees, department managers, branch heads, and company executives to query organizational workflows, leaves, payroll policies, onboarding protocols, and system actions in natural language with strict grounding in official operational manuals.

---

## 📑 Table of Contents
- [Overview & Business Context](#-overview--business-context)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Configuration](#-configuration)
- [API Documentation](#-api-documentation)
- [Role-Based Access Scope](#-role-based-access-scope)
- [License](#-license)

---

## 🎯 Overview & Business Context

In large organizations using the **HRP ERP system**, employees across different tiers frequently encounter friction navigating complex administrative workflows:
- How to submit a business trip or vacation request
- How HR managers approve timesheets and generate labor contracts
- How branch directors track team KPIs and approve requisition forms
- How executives review organizational analytics

This system solves this challenge by providing an intelligent, context-aware RAG assistant that retrieves strictly authorized company guidelines, answers questions in real-time, cites official policy sections, and provides direct deep-links to the corresponding pages inside the HRP web platform.

---

## ✨ Key Features

- **Role-Aware Access Control (RBAC)**: Enforces strict data segregation across 4 organizational tiers:
  - `EMPLOYEE` (Xodim)
  - `HR_MANAGER` (Kadrlar bo'limi boshlig'i)
  - `BRANCH_MANAGER` (Filial rahbari)
  - `OWNER` (Kompaniya rahbari / Ta'sischi)
- **Multilingual Support**: Accepts natural language questions in **Uzbek**, **Russian**, and **English**, while seamlessly matching against Uzbek enterprise manuals.
- **Strict Hallucination Prevention**: Integrated action guards and cosine thresholding prevent the LLM from making up undocumented policies or non-existent UI buttons.
- **Direct System Navigation**: Automatically parses and resolves ERP section URLs, returning clickable navigation routes for the user's role.
- **High-Performance Vector Store**: Pre-computed vector store utilizing **BGE-M3** multilingual dense embeddings with role-scoped cosine similarity and title relevance boosting.

---

## 🏗 System Architecture

```
User Query (Question + User Role + Language)
               │
               ▼
┌──────────────────────────────────────────────┐
│  FastAPI Gateway (Scoped CORS + Validation)  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Role Scoping & Action Guard Validation      │
│  - Filters out-of-scope policies             │
│  - Prevents non-existent action fabrication  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Semantic Search Engine (BGE-M3 + NumPy)     │
│  - Dense vector similarity ranking           │
│  - Section title boosting                    │
│  - Top-K chunk retrieval                     │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  LLM Synthesis (Gemma / Qwen via vLLM)       │
│  - Role-specific system instructions         │
│  - Grounded context synthesis                │
│  - Markdown answer formatting                │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
Structured Response (Answer + Deep Link URLs + Cited Chunks)
```

---

## 🛠 Technology Stack

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Async Python 3.11+)
- **ASGI Server**: [Uvicorn](https://www.uvicorn.org/)
- **Data Validation & Settings**: [Pydantic v2](https://docs.pydantic.dev/) & [python-dotenv](https://github.com/theskumar/python-dotenv)
- **Vector Search & Math**: [NumPy](https://numpy.org/) (High-speed Cosine Similarity)
- **Document Ingestion**: [python-docx](https://python-docx.readthedocs.io/) (Structured XML/DOCX parsing)
- **Embedding Model**: `BAAI/bge-m3` (Multilingual 1024-dimensional dense vectors)
- **LLM Inference**: OpenAI-compatible API (`vLLM`, `Ollama`, or standard endpoints running `gemma4-31b` or `qwen2.5:14b-instruct`)

---

## 📂 Project Structure

```
hrp-rag-chatbot/
├── hr_chatbot/                 # Core Chatbot & RAG Engine
│   ├── __init__.py
│   ├── config.py              # Dynamic configuration & env loader
│   ├── document_loader.py     # DOCX manual parser & chunker
│   ├── messages.py            # Multilingual system messages & prompts
│   ├── ollama_client.py       # OpenAI/Ollama/vLLM HTTP client
│   ├── schemas.py             # Pydantic data schemas & Enums
│   ├── section_urls.py        # ERP route mappings & deep-link resolver
│   ├── service.py             # RAG orchestration & guardrails
│   └── vector_store.py        # Vector storage, indexing & cosine search
├── manuals/                   # Official HRP ERP Word manuals (.docx)
│   ├── ERP_HR_BranchManager_qollanma.docx
│   ├── ERP_HR_HRManager_qollanma.docx
│   ├── ERP_HR_Owner_qollanma.docx
│   └── ERP_HR_Xodim_qollanma.docx
├── storage/                   # Pre-indexed vector store data
│   └── vector_store.json
├── .env.example               # Template environment configuration
├── .gitignore                 # Git ignore file (prevents secret leaks)
├── main.py                    # Application entry point & FastAPI routes
├── README.md                  # Comprehensive documentation
└── requirements.txt           # Project dependencies
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.10+ (Recommended: Python 3.12)
- Git
- Access to an LLM / Embedding endpoint (Ollama, vLLM, or OpenAI-compatible server)

### 2. Clone the Repository
```bash
git clone https://github.com/quvonchbek212/hrp-rag-chatbot.git
cd hrp-rag-chatbot
```

### 3. Create a Virtual Environment
```bash
# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your connection details:
```bash
cp .env.example .env
```

Edit `.env`:
```env
PORT=8000
CHAT_BASE_URL=https://your-llm-server.com/v1
CHAT_API_KEY=your_api_key_here
CHAT_MODEL=gemma4-31b
EMBEDDING_BASE_URL=https://your-llm-server.com/v1
EMBEDDING_MODEL=bge-m3
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Port on which the FastAPI application runs |
| `CHAT_BASE_URL` | - | OpenAI-compatible endpoint URL for chat completions |
| `CHAT_API_KEY` | - | Bearer token / API key for the LLM endpoint |
| `CHAT_MODEL` | `gemma4-31b` | Target LLM model name (`gemma4-31b`, `qwen2.5:14b-instruct`) |
| `EMBEDDING_BASE_URL` | - | Endpoint URL for generating text embeddings |
| `EMBEDDING_MODEL` | `bge-m3` | Embedding model identifier |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Fallback Ollama service address |
| `HR_CHATBOT_TOP_K` | `5` | Maximum number of context chunks retrieved per query |
| `HR_CHATBOT_MIN_RELEVANCE_SCORE` | `0.58` | Minimum cosine similarity threshold for same-language queries |
| `HR_CHATBOT_CROSS_LINGUAL_MIN_RELEVANCE_SCORE` | `0.46` | Threshold for cross-lingual queries (Russian/English) |

---

## 📖 API Documentation

Once the server is running, explore interactive Swagger docs at:
👉 **`http://localhost:8000/docs`**

### Core Endpoints

#### 1. Ask Chatbot
`POST /api/v1/chat/ask`

**Request Body:**
```json
{
  "role": "EMPLOYEE",
  "question": "Qanday qilib ta'til uchun ariza topshiraman?",
  "language": "uz"
}
```

**Response Body (200 OK):**
```json
{
  "answer": "Ta'til uchun ariza topshirish uchun:\n1. Tizimda **«Arizalar»** bo'limiga kiring.\n2. **«Yangi ariza»** tugmasini bosing.\n3. Ariza turidan **«Mehnat ta'tili»**ni tanlang va sanalarni belgilab, tasdiqlashga yuboring.",
  "role": "EMPLOYEE",
  "language": "uz",
  "sources": [
    {
      "chunk_id": "EMPLOYEE-0014",
      "document": "ERP_HR_Xodim_qollanma.docx",
      "title_path": "Arizalar > Mehnat ta'tili",
      "score": 0.842
    }
  ],
  "section_url": "/dashboard/applications/create",
  "is_out_of_domain": false
}
```

#### 2. Rebuild Vector Index
`POST /api/v1/index/rebuild`
Re-scans all `.docx` manuals from `manuals/`, recalculates chunk embeddings, and updates `storage/vector_store.json`.

#### 3. Health Check
`GET /health`
Returns `{"status": "ok"}`.

---

## 👥 Role-Based Access Scope

| Role Key | Name | Search Scope |
|---|---|---|
| `EMPLOYEE` | Xodim | Personal profile, attendance, leave requests, payroll slips |
| `HR_MANAGER` | Kadrlar Menejeri | Employee profiles, labor contracts, vacation approvals, hiring workflows |
| `BRANCH_MANAGER` | Filial Rahbari | Branch employee monitoring, branch timesheets, shift approvals, KPI reviews |
| `OWNER` | Rahbariyat / Ta'sischi | Organization-wide executive reports, company financial HR analytics, global settings |

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
