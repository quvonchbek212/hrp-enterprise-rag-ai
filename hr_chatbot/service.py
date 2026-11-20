"""RAG orchestration for the ERP HR chatbot."""

from __future__ import annotations

import logging
import re

from hr_chatbot.config import HRChatbotConfig, get_hr_chatbot_config
from hr_chatbot.ollama_client import OllamaClient
from hr_chatbot.messages import (
    GREETING,
    LANGUAGE_NAMES,
    NOTE_PREFIXES,
    OUT_OF_DOMAIN,
    UNDOCUMENTED_PAGE,
)
from hr_chatbot.schemas import ROLE_LABELS, ChatbotResponse, HRRole, Language, SourceChunk
from hr_chatbot.section_urls import match_undocumented_page, resolve_section_url
from hr_chatbot.vector_store import HRVectorStore, SearchResult

logger = logging.getLogger(__name__)


class HRChatbotService:
    """Role-aware retrieval and grounded answer generation."""

    # (amal nomi, savoldagi so'rov iboralari, kontekstdagi tasdiq iboralari).
    # Savol amalni so'rasa-yu, kontekstda o'sha amal tasdig'i bo'lmasa, javob to'qib
    # chiqarilmaydi: qo'llanmada nima borligi aytiladi. Savol uch tilda bo'lishi
    # mumkin, kontekst esa doim o'zbekcha — shuning uchun kalit so'zlar ajratilgan.
    _ACTION_GUARD_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
        (
            "yaratish",
            (
                "yarat", "qo'shaman", "qo'shish", "qo'shsam", "qo'shamiz",
                "созда", "добав", "новый", "новую", "новое",
                "create", "add ", "adding", "new ",
            ),
            ("yarat", "qo'sh", "saqlash"),
        ),
        (
            "o'chirish",
            (
                "o'chir", "olib tashla", "bekor qil",
                "удал", "отмен",
                "delete", "remove", "cancel",
            ),
            ("o'chir", "olib tashla", "bekor qil"),
        ),
        (
            "tahrirlash",
            (
                "tahrir", "taxrir", "o'zgartir",
                # "правит" ishlatilmaydi: u "отправить" ichiga ham tushadi.
                "измен", "редактир", "исправ", "обнов",
                "edit", "change", "update", "modify",
            ),
            ("tahrir", "taxrir", "o'zgartir", "saqlash"),
        ),
        (
            "buyurtma berish",
            (
                "buyurtma", "buyurish", "buyursam", "zakaz",
                "заказ",
                # "order" yolg'iz olinmaydi: "in order to" ham mos kelib qolardi.
                "order food", "place an order", "ordering",
            ),
            ("buyurtma", "buyurish", "zakaz"),
        ),
    )

    # Qo'llanmalardagi standart yorliqlar: javob shu tuzilishni saqlab qoladi.
    _OPEN_SECTION_LABEL = "bo'limni qanday ochish"
    _SECTION_LABELS: frozenset[str] = frozenset(
        {"bo'limni qanday ochish", "oynada nima ko'rinadi"}
    )

    def __init__(self, config: HRChatbotConfig | None = None) -> None:
        self._config = config or get_hr_chatbot_config()
        self._ollama = OllamaClient(
            host=self._config.ollama_host,
            timeout_seconds=self._config.timeout_seconds,
            chat_base_url=self._config.chat_base_url,
            api_key=self._config.chat_api_key,
            embedding_base_url=self._config.embedding_base_url,
        )
        self._store = HRVectorStore(
            store_path=self._config.vector_store_path,
            manuals_dir=self._config.manuals_dir,
            ollama_client=self._ollama,
            embedding_model=self._config.embedding_model,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )

    def answer(
        self,
        *,
        role: HRRole,
        question: str,
        language: Language = Language.UZ,
    ) -> ChatbotResponse:
        """Answer one question using only the selected role manual."""

        if self._should_answer_without_retrieval(question):
            return ChatbotResponse(
                role=role,
                language=language,
                answer=self._answer_without_retrieval(question, language=language),
                sources=[],
            )

        # Qo'llanmada tavsiflanmagan sahifa: qidiruvdan javob chiqmaydi, shuning
        # uchun havolaning o'zini beramiz.
        page = match_undocumented_page(role, question)
        if page is not None:
            notice = UNDOCUMENTED_PAGE[language].format(name=page.name)
            return ChatbotResponse(
                role=role,
                language=language,
                answer=notice,
                sources=[],
                url=page.url,
            )

        candidates = self._store.search(
            role=role,
            question=question,
            top_k=max(2, self._config.top_k),
        )
        min_score = self._min_relevance_score(language)
        if not self._has_relevant_hits(candidates, min_score=min_score):
            return ChatbotResponse(
                role=role,
                language=language,
                answer=OUT_OF_DOMAIN[language],
                sources=[],
            )

        hits = self._select_context_hits(
            candidates[: self._config.top_k],
            min_score=min_score,
            margin=self._context_margin(language),
        )
        context = self._format_context(hits)
        # Javob har doim o'zbekcha quriladi: prompt ham, qo'llanma ham o'zbekcha.
        # Rus va ingliz javob shu o'zbekcha matnni tarjima qilish orqali olinadi —
        # model ikkala ishni bir vaqtda qilganda javob sifati keskin pasayadi.
        uzbek_answer = self._guard_unsupported_action(
            question=question,
            context=context,
            evidence=hits[0].chunk.text,
        )
        if uzbek_answer is None:
            system_prompt, user_prompt = self._build_prompts(
                role=role,
                question=question,
                context=context,
            )
            uzbek_answer = self._remove_unsupported_note(self._clean_answer(
                self._ollama.chat(
                    model=self._config.chat_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    num_ctx=self._config.num_ctx,
                    num_predict=self._config.num_predict,
                    temperature=self._config.temperature,
                )
            ), context=context)

        return self._build_response(
            role=role,
            answer=self._translate(uzbek_answer, language=language),
            uzbek_answer=uzbek_answer,
            hits=hits,
            language=language,
        )

    def _min_relevance_score(self, language: Language) -> float:
        """Uzbek manuals score lower against a Russian or English question."""

        if language is Language.UZ:
            return self._config.min_relevance_score
        return self._config.cross_lingual_min_relevance_score

    def _translate(self, text: str, *, language: Language) -> str:
        """Translate a manual-derived answer without changing what it says."""

        if language is Language.UZ:
            return text
        system_prompt = f"""
Siz tarjimonsiz. Berilgan ERP HR yo'riqnomasini {LANGUAGE_NAMES[language]} tiliga tarjima qiling.

Qoidalar:
- Ma'noni o'zgartirmang, yangi ma'lumot qo'shmang, hech narsani tashlab ketmang.
- Matn tuzilishini saqlang: satrlar, raqamli qadamlar, ro'yxat belgilari o'sha holicha qolsin.
- ERP HR bo'lim va tugma nomlarini ham tarjima qiling (tizim interfeysi shu tilda ko'rinadi).
- Matnda o'zbekcha so'z qoldirmang.
- Faqat tarjimani qaytaring, izoh yoki sarlavha qo'shmang.
""".strip()
        translated = self._request_translation(text, system_prompt=system_prompt)
        if _has_foreign_script(translated):
            # Model ba'zan tarjima o'rtasida xitoycha matn chiqarib yuboradi.
            logger.warning("hr_chatbot_translation_retried language=%s", language.value)
            translated = self._request_translation(text, system_prompt=system_prompt)
        return translated if not _has_foreign_script(translated) else text

    def _request_translation(self, text: str, *, system_prompt: str) -> str:
        return self._clean_answer(
            self._ollama.chat(
                model=self._config.chat_model,
                system_prompt=system_prompt,
                user_prompt=text,
                num_ctx=self._config.num_ctx,
                num_predict=self._config.num_predict,
                temperature=0.0,
            )
        )

    @staticmethod
    def _build_response(
        *,
        role: HRRole,
        answer: str,
        hits: list[SearchResult],
        language: Language = Language.UZ,
        uzbek_answer: str | None = None,
    ) -> ChatbotResponse:
        """Attach manual sources and the link of the section the answer came from."""

        sources = [
            SourceChunk(
                role=item.chunk.role,
                document=item.chunk.document,
                chunk_id=item.chunk.chunk_id,
                score=round(item.score, 6),
                preview=item.chunk.text[:350],
                url=resolve_section_url(item.chunk.role, item.chunk.title_path),
            )
            for item in hits
        ]
        # Havola bo'lim nomiga qarab tanlanadi, tarjimada esa nomlar o'zgaradi —
        # shuning uchun tanlash har doim o'zbekcha matn ustida bajariladi.
        url = HRChatbotService._pick_answer_url(
            answer=uzbek_answer if uzbek_answer is not None else answer,
            hits=hits,
            sources=sources,
        )
        # Havola javob matniga qo'shilmaydi — mijoz `url` maydonidan foydalanadi.
        return ChatbotResponse(
            role=role,
            language=language,
            answer=answer,
            sources=sources,
            url=url,
        )

    @staticmethod
    def _pick_answer_url(
        *,
        answer: str,
        hits: list[SearchResult],
        sources: list[SourceChunk],
    ) -> str | None:
        """Link the section the answer was actually written from.

        Havola eng yuqori ballli chunkdan olinmaydi: kontekstda bir nechta bo'lim
        bo'ladi va javob ular ichidan bittasini tanlab yozadi. Bo'lim nomini javob
        matnidan izlash ham ishonchsiz — qo'llanmadagi mavzu nomlari tavsifiy
        ("Xodimga yozish (Xabarlar)") va javobda aynan shunday yozilmaydi.
        Shuning uchun javob qaysi bo'lim matni bilan eng ko'p kesishsa, o'sha
        bo'limning havolasi olinadi.
        """

        answer_tokens = _content_tokens(HRChatbotService._normalize_apostrophes(answer))
        if not answer_tokens:
            return next((source.url for source in sources if source.url), None)

        # Havola faqat eng mos hitlar orasidan tanlanadi: retrieval bali top hitdan
        # ancha past bo'lgan bo'lim (masalan bir necha bo'lim umumiy so'zlarni baham
        # ko'rsa) noto'g'ri havola bermasligi kerak.
        top_score = hits[0].score if hits else 0.0
        best_key = (0, 0.0)
        best_url: str | None = None
        for item, source in zip(hits, sources, strict=True):
            if not source.url:
                continue
            if item.score < top_score - _URL_SCORE_MARGIN:
                continue
            chunk_tokens = _content_tokens(HRChatbotService._normalize_apostrophes(item.chunk.text))
            if not chunk_tokens:
                continue
            shared = chunk_tokens & answer_tokens
            # Asosiy o'lchov — javobga ko'chgan mazmunli so'zlar soni. Nisbat
            # ishlatilsa, qisqa bo'lim g'olib chiqib qolardi: javob asosan boshqa
            # bo'limdan yozilgan bo'lsa ham, uning bir necha so'zi to'liq qoplanardi.
            key = (len(shared), len(shared) / len(chunk_tokens))
            if key > best_key:
                best_key = key
                best_url = source.url

        if best_url is not None and best_key[0] >= _MIN_SHARED_TOKENS:
            return best_url
        return next((source.url for source in sources if source.url), None)

    def rebuild_index(self) -> None:
        """Force a vector store rebuild from manuals."""

        self._store.rebuild()

    @staticmethod
    def _answer_without_retrieval(question: str = "", *, language: Language = Language.UZ) -> str:
        """Salomlashuvga tanishtiruv, qolgan hollarda qat'iy rad javobini qaytaradi."""

        if HRChatbotService._is_greeting(question):
            return GREETING[language]
        return OUT_OF_DOMAIN[language]

    @staticmethod
    def _normalize_message(question: str) -> str:
        """Kichik harf, bitta probel va chekka tinish belgilarisiz ko'rinish.

        Tinish belgisi kesilmasa «Salom!» salomlashuv sifatida tanilmay,
        qidiruvga tushib ketardi va rad javobi qaytardi.
        """

        normalized = " ".join(question.lower().split())
        return normalized.strip(" !?.,;:…-—’'\"")

    @staticmethod
    def _is_greeting(question: str) -> bool:
        """Return True when the message is only a greeting."""

        normalized = HRChatbotService._normalize_message(question)
        greetings = {
            "salom",
            "salom alaykum",
            "assalom",
            "assalom alaykum",
            "assalomu alaykum",
            "assalomu aleykum",
            "xayrli kun",
            "xayrli tong",
            "xayrli kech",
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "privet",
            "привет",
            "здравствуйте",
            "здравствуй",
            "доброе утро",
            "добрый день",
            "добрый вечер",
        }
        small_talk_terms = (
            "ishlaring yaxshimi",
            "ishlar yaxshimi",
            "qalaysan",
            "qalesan",
            "yaxshimisan",
            "nima gap",
            "nima gaplar",
        )
        return normalized in greetings or any(term in normalized for term in small_talk_terms)

    @staticmethod
    def _should_answer_without_retrieval(question: str) -> bool:
        """Skip retrieval only for empty messages and greetings."""

        normalized = HRChatbotService._normalize_message(question)
        if not normalized:
            return True

        if HRChatbotService._is_greeting(question):
            return True
        return False

    @staticmethod
    def _has_relevant_hits(hits: list[SearchResult], *, min_score: float) -> bool:
        """Classify domain relevance from retrieval strength, not a keyword list."""

        if not hits:
            return False
        top_score = hits[0].score
        if top_score >= min_score:
            return True

        # A slightly lower score is accepted only when one manual section is
        # clearly more relevant than all alternatives.
        second_score = hits[1].score if len(hits) > 1 else 0.0
        relaxed_score = max(0.0, min_score - 0.06)
        return top_score >= relaxed_score and (top_score - second_score) >= 0.10

    @staticmethod
    def _select_context_hits(
        hits: list[SearchResult],
        *,
        min_score: float,
        margin: float = 0.18,
    ) -> list[SearchResult]:
        """Exclude weak neighboring topics before building the LLM context."""

        if not hits:
            return []
        cutoff = max(min_score, hits[0].score - margin)
        selected = [hit for hit in hits if hit.score >= cutoff]
        return selected or [hits[0]]

    @staticmethod
    def _context_margin(language: Language) -> float:
        """Cross-lingual scores sit close together, so the context must be tighter."""

        return 0.18 if language is Language.UZ else 0.05

    @staticmethod
    def _build_prompts(
        *,
        role: HRRole,
        question: str,
        context: str,
        language: Language = Language.UZ,
    ) -> tuple[str, str]:
        """Build a strict RAG prompt that avoids unsupported actions."""

        system_prompt = f"""
Siz ERP HR websaytini qanday ishlatishni tushuntiradigan professional RAG yordamchisiz.

Til qoidasi:
- Savol rus yoki ingliz tilida bo'lishi mumkin. Javobni baribir o'zbek tilida yozing — kerak bo'lsa u keyin tarjima qilinadi.

Majburiy qoidalar:
- Faqat berilgan qo'llanma kontekstidagi ma'lumotlarga tayaning.
- Kontekstda aniq yozilmagan tugma, amal, huquq, bo'lim yoki jarayonni o'zingizdan qo'shmang.
- Foydalanuvchi so'ragan amal kontekstda aniq tasdiqlanmagan bo'lsa, aynan shunday ayting: "Qo'llanmada bu amal bo'yicha aniq ko'rsatma berilmagan."
- Kontekstda faqat ko'rish yoki kuzatish yozilgan bo'lsa, uni yaratish, o'chirish yoki tahrirlash yo'riqnomasi sifatida talqin qilmang.
- Kontekstda bir nechta manba bo'lishi mumkin; faqat o'sha manbalardagi ma'lumotdan foydalaning va kontekstda yo'q qo'llanmani o'zingizdan qo'shmang.
- Javob ERP HR tizimini ishlatish bo'yicha bo'lsin; umumiy hayotiy maslahat, taxmin yoki qo'llanmadan tashqari ma'lumot bermang.
- Kontekst matnini uzun ko'chirmang; kerakli qadamlarni qisqa, ravon va amaliy qilib tushuntiring.
- Javob rasmiy, aniq va tushunarli uslubda bo'lsin.
- "va hokazo", "erda", "taxminan" kabi noaniq yoki xato iboralarni ishlatmang.
- Javobni "Xodim roli..." kabi sun'iy jumla bilan boshlamang.

Javob formati:
1. Avval qisqa xulosa bering.
2. Keyin faqat kontekstda bor bo'lgan 2-5 ta amaliy qadamni yozing.
3. Faqat kontekstda "Eslatma:" satri bo'lsa, javob oxirida shu eslatmani javob tilida yozing.
4. Kontekstda yo'q umumiy eslatma, ogohlantirish yoki izohni javobga qo'shmang.
5. Kontekstda "Eslatma:" bo'lmasa, eslatma haqida umuman yozmang — "eslatma yo'q" kabi jumla ham yozmang.
""".strip()
        user_prompt = f"""
Rol: {ROLE_LABELS.get(role, role.value)}
Savol: {question.strip()}

Qo'llanma konteksti:
{context}

Javobni faqat yuqoridagi kontekstga asoslanib yozing. Kontekstda dalil bo'lmasa taxmin qilmang va qo'llanmadan tashqari ma'lumot qo'shmang.
""".strip()
        return system_prompt, user_prompt

    @staticmethod
    def _format_context(hits: list[SearchResult]) -> str:
        parts: list[str] = []
        for index, item in enumerate(hits, start=1):
            parts.append(
                f"[Manba {index}: {item.chunk.document}, {item.chunk.chunk_id}, score={item.score:.3f}]\n"
                f"{item.chunk.text}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _normalize_apostrophes(text: str) -> str:
        """Unify the apostrophe variants used across manuals, questions and models."""

        normalized = text
        for variant in ("‘", "’", "ʻ", "ʼ", "`", "´"):
            normalized = normalized.replace(variant, "'")
        return normalized

    @staticmethod
    def _guard_unsupported_action(
        *,
        question: str,
        context: str,
        evidence: str | None = None,
    ) -> str | None:
        """Return a safe answer when the question asks for an action not present in context.

        Amal tasdig'i faqat savolga eng mos bo'lim matnidan qidiriladi. Kontekstga
        qo'shni bo'limlar ham tushishi mumkin va ularda "Yaratish" so'zi uchrasa,
        himoya noo'rin ravishda o'chib qolardi — ruscha va inglizcha savollarda
        ballar bir-biriga yaqin bo'lgani uchun bu ayniqsa tez-tez sodir bo'lgan.
        """

        question_lower = HRChatbotService._normalize_apostrophes(question).lower()
        evidence_lower = HRChatbotService._normalize_apostrophes(
            evidence if evidence is not None else context
        ).lower()

        for action, question_terms, context_terms in HRChatbotService._ACTION_GUARD_RULES:
            asks_action = any(term in question_lower for term in question_terms)
            if not asks_action:
                continue
            if any(term in evidence_lower for term in context_terms):
                continue
            return HRChatbotService._build_unsupported_action_answer(action=action, context=context)
        return None

    @staticmethod
    def _build_unsupported_action_answer(*, action: str, context: str) -> str:
        """State that the action is undocumented, then quote what the manual does describe."""

        groups, note = HRChatbotService._manual_section_groups(context)
        parts = [f"Qo'llanmada {action} amali bo'yicha aniq ko'rsatma berilmagan."]
        for label, lines in groups:
            if not lines:
                continue
            # Qo'llanmada "ochish" satrlari ketma-ket qadam, qolganlari tavsif.
            if HRChatbotService._normalize_apostrophes(label).lower() == HRChatbotService._OPEN_SECTION_LABEL:
                body = "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))
            else:
                body = "\n".join(f"- {line}" for line in lines)
            parts.append(f"{label}:\n{body}" if label else body)
        if note:
            parts.append(note)
        return "\n\n".join(parts)

    @staticmethod
    def _manual_section_groups(context: str) -> tuple[list[tuple[str, list[str]]], str]:
        """Regroup the top-scoring manual section under its own DOCX labels."""

        section: list[str] = []
        for line in context.splitlines():
            if line.startswith("[Manba "):
                if section:
                    break
                continue
            if line.strip():
                section.append(line.strip())

        groups: list[tuple[str, list[str]]] = []
        current: tuple[str, list[str]] = ("", [])
        note = ""
        for line in section:
            lowered = HRChatbotService._normalize_apostrophes(line).lower()
            if lowered.startswith("eslatma:"):
                note = line
                continue
            if lowered.startswith(("asosiy bo'lim:", "mavzu:")):
                continue
            if lowered in HRChatbotService._SECTION_LABELS:
                if current[1]:
                    groups.append(current)
                current = (line, [])
                continue
            current[1].append(line)
        if current[1]:
            groups.append(current)
        return [(label, lines[:4]) for label, lines in groups], note

    @staticmethod
    def _clean_answer(answer: str) -> str:
        """Normalize occasional mojibake produced by local models."""

        replacements = {
            "Ť": "«",
            "ť": "»",
            "Ň": "O'",
            "ň": "o'",
            "roli erda": "rolida",
            "roli asosida,": "rolida",
            " va hokazo": "",
        }
        cleaned = answer
        for source, target in replacements.items():
            cleaned = cleaned.replace(source, target)
        return cleaned.strip()

    @staticmethod
    def _remove_unsupported_note(answer: str, *, context: str) -> str:
        """Drop model-added notes when the retrieved manual has no note."""

        if "eslatma:" in context.lower():
            return answer.strip()
        # Javob tiliga qarab model eslatmani turlicha nomlaydi.
        pattern = "|".join(NOTE_PREFIXES)
        return re.split(rf"\n\s*(?:{pattern})\s*:", answer, maxsplit=1, flags=re.IGNORECASE)[0].rstrip()


def _has_foreign_script(text: str) -> bool:
    """Detect scripts that never belong in an Uzbek, Russian or English answer."""

    return any("　" <= char <= "鿿" or "가" <= char <= "힯" for char in text)


# Deyarli har bir bo'limda uchraydigan qolip so'zlar: ular bo'limlarni ajratmaydi,
# shuning uchun ustma-ustlikni hisoblashda tashlab yuboriladi.
_BOILERPLATE_TOKENS: frozenset[str] = frozenset(
    {
        "asosiy", "mavzu", "bo'lim", "bo'limni", "bo'limi", "bo'limini", "bo'limlar",
        "qanday", "ochish", "ochilgan", "oynada", "nima", "ko'rinadi", "ko'rish",
        "chap", "menyudan", "menyu", "ichida", "tanlang", "bosing", "tugma",
        "tugmasi", "tugmasini", "yaratish", "saqlash", "eslatma", "formada",
        "forma", "maydon", "maydonlarni", "maydonlar", "to'ldiring", "quyidagi",
        "bilan", "uchun", "mumkin", "kerak", "ham", "yoki", "ustunlari", "jadval",
        "jadvalda", "ro'yxati", "filtri", "holat", "sana",
    }
)


_BOILERPLATE_STEMS: frozenset[str] = frozenset(word[:5] for word in _BOILERPLATE_TOKENS)

# Havolani bo'limga bog'lash uchun javob shu bo'limdan kamida shuncha mazmunli
# so'z olgan bo'lishi kerak; aks holda eng mos chunk havolasi qaytariladi.
_MIN_SHARED_TOKENS = 3
# Havola nomzodi top hitdan shu balldan ko'proq past bo'lsa, hisobga olinmaydi.
_URL_SCORE_MARGIN = 0.12


def _content_tokens(text: str) -> set[str]:
    """Bo'limni ajratib turadigan so'z o'zaklari.

    O'zbekcha qo'shimchalar so'zni o'zgartiradi ("so'rovlar" / "so'rovlarim"),
    shuning uchun so'zlar o'zagi bo'yicha solishtiriladi.
    """

    stems = {word[:5] for word in re.findall(r"[a-z0-9']{4,}", text.lower())}
    return stems - _BOILERPLATE_STEMS
