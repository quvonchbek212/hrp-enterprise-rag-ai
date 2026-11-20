"""Pydantic schemas for the HR chatbot API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Language(StrEnum):
    """Javob tili. Qo'llanmalar o'zbekcha, javob esa tanlangan tilda beriladi."""

    UZ = "UZ"
    RU = "RU"
    EN = "EN"


class HRRole(StrEnum):
    """Supported ERP HR website roles."""

    OWNER = "OWNER"
    BRANCH_MANAGER = "BRANCH_MANAGER"
    DEPARTMENT_MANAGER = "DEPARTMENT_MANAGER"
    EMPLOYEE = "EMPLOYEE"


ROLE_LABELS: dict[HRRole, str] = {
    HRRole.OWNER: "Rahbar (Owner)",
    HRRole.BRANCH_MANAGER: "Filial rahbari",
    HRRole.DEPARTMENT_MANAGER: "Bo'lim rahbari (HR)",
    HRRole.EMPLOYEE: "Xodim",
}

# Rol qaysi qo'llanmalardan javob oladi. Filial rahbari ayni paytda xodim ham,
# lekin uning qo'llanmasida shaxsiy bo'limlar (ta'til, davomat, oshxona) yo'q,
# shuning uchun u Xodim qo'llanmasidan ham qidiradi.
ROLE_SEARCH_SCOPE: dict[HRRole, tuple[HRRole, ...]] = {
    HRRole.OWNER: (HRRole.OWNER,),
    HRRole.BRANCH_MANAGER: (HRRole.BRANCH_MANAGER, HRRole.EMPLOYEE),
    HRRole.DEPARTMENT_MANAGER: (HRRole.DEPARTMENT_MANAGER,),
    HRRole.EMPLOYEE: (HRRole.EMPLOYEE,),
}


class ChatbotRequest(BaseModel):
    """User question, role and answer language."""

    role: HRRole = Field(..., description="Foydalanuvchi roli.")
    question: str = Field(..., min_length=2, max_length=4000, description="ERP HR saytidan foydalanish bo'yicha savol.")
    language: Language = Field(default=Language.UZ, description="Javob tili: UZ, RU yoki EN.")


class SourceChunk(BaseModel):
    """Manual fragment used by the answer."""

    role: HRRole
    document: str
    chunk_id: str
    score: float
    preview: str
    url: str | None = Field(default=None, description="Bo'limning ERP HR sahifasi.")


class ChatbotResponse(BaseModel):
    """Answer with traceable manual sources."""

    role: HRRole
    language: Language = Language.UZ
    answer: str
    sources: list[SourceChunk]
    url: str | None = Field(
        default=None,
        description="Javob olingan katta bo'limning ERP HR sahifasi.",
    )

