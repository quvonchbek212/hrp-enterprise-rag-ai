"""
HRP AI Assistant & RAG Chatbot API
Role-aware RAG system for the HRP (Human Resource Planning) ERP platform.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from hr_chatbot.ollama_client import OllamaClientError
from hr_chatbot.schemas import ChatbotRequest, ChatbotResponse
from hr_chatbot.service import HRChatbotService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hrp_rag_chatbot")

_service: HRChatbotService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize chatbot service and verify vector store on startup."""
    global _service
    logger.info("Initializing HRP Chatbot service...")
    try:
        _service = HRChatbotService()
        logger.info("HRP Chatbot service initialized successfully.")
    except Exception as exc:
        logger.warning(f"Could not initialize vector store during startup: {exc}. Will retry on request.")
    yield


app = FastAPI(
    title="HRP AI Assistant & RAG Chatbot API",
    description="""
# 🤖 HRP AI Assistant (RAG Chatbot)

Role-aware Retrieval-Augmented Generation (RAG) assistant designed for the **HRP (Human Resource Planning) ERP system**.

### 🌟 Key Capabilities
- **Role-Aware Access Control**: Tailored answers based on user roles (`EMPLOYEE`, `HR_MANAGER`, `BRANCH_MANAGER`, `OWNER`).
- **Semantic Vector Search**: High-precision search using `bge-m3` embeddings with role-scoped filtering and title boosting.
- **Multilingual Support**: Supports queries in Uzbek (O'zbek), Russian (Русский), and English.
- **Strict Grounding & Hallucination Prevention**: Answers are strictly grounded in official ERP operational manuals; out-of-domain queries or undocumented actions are politely intercepted.
- **Direct System Navigation**: Automatically returns deep links and section URLs for the HRP ERP system.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service() -> HRChatbotService:
    global _service
    if _service is None:
        _service = HRChatbotService()
    return _service


@app.get("/", tags=["General"])
async def root():
    return {
        "service": "HRP AI Assistant & RAG Chatbot API",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health", tags=["General"])
async def health():
    return {"status": "ok"}


@app.post(
    "/api/v1/chat/ask",
    response_model=ChatbotResponse,
    summary="Ask HR Chatbot",
    tags=["Chatbot"],
)
@app.post(
    "/ask",
    response_model=ChatbotResponse,
    include_in_schema=False,
)
async def ask_hr_chatbot(payload: ChatbotRequest) -> ChatbotResponse:
    """
    Answer HRP ERP platform usage questions for specific roles:
    - `EMPLOYEE` (Xodim)
    - `HR_MANAGER` (Kadrlar bo'limi menejeri)
    - `BRANCH_MANAGER` (Filial rahbari)
    - `OWNER` (Kompaniya rahbari / Ta'sischi)
    """
    service = get_service()
    try:
        return await run_in_threadpool(
            service.answer,
            role=payload.role,
            question=payload.question,
            language=payload.language,
        )
    except OllamaClientError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"status": "error", "error_code": exc.code, "message": exc.message},
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "error_code": "HR_MANUAL_NOT_FOUND", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.exception("hr_chatbot_failed")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "error_code": "HR_CHATBOT_ERROR", "message": f"HR chatbot error: {exc}"},
        ) from exc


@app.post(
    "/api/v1/index/rebuild",
    summary="Rebuild vector index from DOCX manuals",
    tags=["Indexing"],
)
async def rebuild_index():
    """Forces recalculation of embeddings from manuals in the manuals/ folder."""
    service = get_service()
    try:
        await run_in_threadpool(service.rebuild_index)
        return {"status": "ok", "message": "Vector index rebuilt successfully."}
    except Exception as exc:
        logger.exception("rebuild_index_failed")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Failed to rebuild index: {exc}"},
        )
