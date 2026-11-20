"""Persistent JSON vector store for HR manuals."""

from __future__ import annotations

import difflib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import numpy as np

from hr_chatbot.document_loader import ManualChunk, load_manual_chunks, manuals_fingerprint
from hr_chatbot.ollama_client import OllamaClient
from hr_chatbot.schemas import ROLE_SEARCH_SCOPE, HRRole

STORE_VERSION = 6
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Vector search hit."""

    chunk: ManualChunk
    score: float


class HRVectorStore:
    """Loads, builds and searches role-filtered manual embeddings."""

    def __init__(
        self,
        *,
        store_path: Path,
        manuals_dir: Path,
        ollama_client: OllamaClient,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._store_path = store_path
        self._manuals_dir = manuals_dir
        self._ollama_client = ollama_client
        self._embedding_model = embedding_model
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._entries: list[dict[str, Any]] = []
        self._chunks: list[ManualChunk] = []
        # Satrlari birlik uzunlikka keltirilgan matritsa: kosinus o'xshashlik bitta
        # matritsa ko'paytmasiga aylanadi.
        self._matrix: np.ndarray | None = None
        self._role_rows: dict[str, np.ndarray] = {}
        self._metadata: dict[str, Any] = {}
        self._loaded_signature: tuple[int, int] | None = None
        self._lock = threading.Lock()

    def ensure_ready(self) -> None:
        """Load an up-to-date store or rebuild it from DOCX manuals, caching it in memory."""

        with self._lock:
            payload = self._payload_if_stale()
            if payload is None:
                return
            if payload and self._payload_matches(payload):
                self._apply_payload(payload)
                return
            try:
                self.rebuild(locked=True)
            except Exception:
                if payload:
                    logger.exception("hr_vector_store_rebuild_failed_using_stale_store")
                    self._apply_payload(payload)
                    return
                raise

    def _payload_if_stale(self) -> dict[str, Any] | None:
        """Return the on-disk payload, or None when the cached index is still valid."""

        signature = self._store_signature()
        if (
            self._matrix is not None
            and signature is not None
            and signature == self._loaded_signature
            and self._config_matches(self._metadata)
        ):
            return None
        return self._read_payload()

    def _store_signature(self) -> tuple[int, int] | None:
        try:
            stat = self._store_path.stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def rebuild(self, *, locked: bool = False) -> None:
        """Re-embed manuals and save vectors to disk."""

        if not locked:
            with self._lock:
                self._rebuild_unlocked()
            return
        self._rebuild_unlocked()

    def _rebuild_unlocked(self) -> None:
        chunks = load_manual_chunks(
            self._manuals_dir,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        embeddings = self._embed_in_batches([normalize_for_embedding(chunk.text) for chunk in chunks])
        entries: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            entries.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "role": chunk.role.value,
                    "document": chunk.document,
                    "text": chunk.text,
                    "source_hash": chunk.source_hash,
                    "title_path": chunk.title_path,
                    "embedding": embedding,
                }
            )

        payload = {
            "version": STORE_VERSION,
            "embedding_model": self._embedding_model,
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "manuals_fingerprint": manuals_fingerprint(self._manuals_dir),
            "entries": entries,
        }
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        self._apply_payload(payload)

    def search(self, *, role: HRRole, question: str, top_k: int) -> list[SearchResult]:
        """Return the most relevant chunks for one role."""

        self.ensure_ready()
        rows = self._scope_rows(role)
        if self._matrix is None or rows.size == 0:
            return []

        query = np.asarray(
            self._ollama_client.embed(model=self._embedding_model, texts=[normalize_for_embedding(question)])[0],
            dtype=np.float32,
        )
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return []

        # Matritsa satrlari allaqachon normallashtirilgan, shuning uchun bitta
        # ko'paytma barcha chunklar uchun kosinus o'xshashlikni beradi.
        scores = self._matrix[rows] @ (query / query_norm)
        title_boost = np.fromiter(
            (0.35 * self._title_match_score(question, self._chunks[row].title_path) for row in rows),
            dtype=np.float32,
            count=rows.size,
        )
        scores = scores + title_boost

        limit = min(top_k, rows.size)
        best = np.argpartition(-scores, limit - 1)[:limit]
        best = best[np.argsort(-scores[best])]
        return [SearchResult(chunk=self._chunks[rows[index]], score=float(scores[index])) for index in best]

    def _scope_rows(self, role: HRRole) -> np.ndarray:
        """Rows of every manual this role may be answered from."""

        scope = ROLE_SEARCH_SCOPE.get(role, (role,))
        parts = [self._role_rows[item.value] for item in scope if item.value in self._role_rows]
        if not parts:
            return np.empty(0, dtype=np.intp)
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    @staticmethod
    def _title_match_score(question: str, title_path: str) -> float:
        """Boost short queries that match a manual feature heading."""

        stop_words = {
            "qanday", "qayerda", "qayerdan", "qanaqa", "nima", "uchun",
            "qilaman", "qilsam", "kerak", "bor", "bormi", "men", "bu",
        }
        query_tokens = [token for token in _search_tokens(question) if token not in stop_words]
        title_tokens = _search_tokens(title_path)
        if not query_tokens or not title_tokens:
            return 0.0

        matched = 0
        for query_token in query_tokens:
            if any(_tokens_match(query_token, title_token) for title_token in title_tokens):
                matched += 1
        return matched / len(query_tokens)

    def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        batch_size = 16
        for index in range(0, len(texts), batch_size):
            batch = texts[index : index + batch_size]
            embeddings.extend(self._ollama_client.embed(model=self._embedding_model, texts=batch))
        return embeddings

    def _read_payload(self) -> dict[str, Any]:
        """Return the stored payload, or an empty dict when it is missing or corrupt."""

        try:
            payload = json.loads(self._store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _payload_matches(self, payload: dict[str, Any]) -> bool:
        return (
            payload.get("version") == STORE_VERSION
            and isinstance(payload.get("entries"), list)
            and self._config_matches(payload)
        )

    def _config_matches(self, metadata: dict[str, Any]) -> bool:
        """Check the index against the active config and the manuals on disk."""

        return (
            same_embedding_model(metadata.get("embedding_model"), self._embedding_model)
            and metadata.get("chunk_size") == self._chunk_size
            and metadata.get("chunk_overlap") == self._chunk_overlap
            and metadata.get("manuals_fingerprint") == manuals_fingerprint(self._manuals_dir)
        )

    def _apply_payload(self, payload: dict[str, Any]) -> None:
        """Cache the index in memory as chunks plus one unit-normalized matrix."""

        entries = list(payload["entries"])
        chunks = [
            ManualChunk(
                chunk_id=str(entry["chunk_id"]),
                role=HRRole(str(entry["role"])),
                document=str(entry["document"]),
                text=str(entry["text"]),
                source_hash=str(entry["source_hash"]),
                title_path=str(entry.get("title_path", "")),
            )
            for entry in entries
        ]
        matrix = np.asarray([entry["embedding"] for entry in entries], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise ValueError("Vector store embeddings are malformed.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix /= norms

        role_rows: dict[str, np.ndarray] = {}
        for role_value in {chunk.role.value for chunk in chunks}:
            role_rows[role_value] = np.array(
                [index for index, chunk in enumerate(chunks) if chunk.role.value == role_value],
                dtype=np.intp,
            )

        self._entries = entries
        self._chunks = chunks
        self._matrix = matrix
        self._role_rows = role_rows
        self._metadata = {key: value for key, value in payload.items() if key != "entries"}
        self._loaded_signature = self._store_signature()


# Apostrof turlari (o'g'ri/egri tirnoq, tutuq belgisi, teskari tirnoq). Foydalanuvchilar
# o'zbekcha so'zlarni ko'pincha apostrofsiz yozadi ("tug'ilgan" -> "tugilgan"), shuning
# uchun embedding va so'z mosligida apostrof butunlay e'tiborsiz qoldiriladi.
_APOSTROPHE_TABLE = {ord(ch): None for ch in "'‘’ʻʼ`"}


def same_embedding_model(stored: Any, configured: str) -> bool:
    """Ollama tegi (`bge-m3:latest`) va vLLM nomi (`bge-m3`) bir xil model hisoblanadi.

    Ikkala server bir xil vektor beradi (kosinus ~1.0), shuning uchun faqat
    nom o'zgargani uchun butun bazani qayta embedding qilish shart emas.
    """

    def base(name: Any) -> str:
        text = str(name or "").strip()
        return text[: -len(":latest")] if text.endswith(":latest") else text

    return base(stored) == base(configured)


def normalize_for_embedding(text: str) -> str:
    """Apostroflarni olib tashlaydi — apostrofli va apostrofsiz yozuv bir xil bo'ladi."""

    return text.translate(_APOSTROPHE_TABLE)


def _search_tokens(text: str) -> list[str]:
    normalized = normalize_for_embedding(text.lower())
    tokens = re.findall(r"[a-z0-9]+", normalized)
    # (1) Ketma-ket takror harflarni yig'ish ("davrrrlar" -> "davrlar").
    # (2) O'zbekcha "x"/"h" chalkashligini birlashtirish ("hodim" ~ "xodim") —
    # bu faqat sarlavha so'z mosligiga ta'sir qiladi, embeddingga emas.
    return [re.sub(r"(.)\1+", r"\1", token).replace("x", "h") for token in tokens]


def _tokens_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if min(len(left), len(right)) < 5:
        return False
    if left[:5] == right[:5]:
        return True
    # Imlo xatosiga bardosh: yaqin so'zlarni ham mos deb hisoblaydi
    # (masalan "davrrrlar" ~ "davrlar", "yngiliklar" ~ "yangiliklar").
    short, long = sorted((left, right), key=len)
    if difflib.SequenceMatcher(None, short, long).ratio() >= 0.8:
        return True
    # O'zbekcha qo'shimchalar (masalan "-imni", "-lari") uzun so'zni cho'zib,
    # nisbatni pasaytiradi. Uzun so'zni qisqasining uzunligiga qisqartirib
    # solishtiramiz — "setifikatimni" ~ "sertifikat" shu tarzda topiladi.
    return difflib.SequenceMatcher(None, short, long[: len(short)]).ratio() >= 0.8
