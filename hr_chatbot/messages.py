"""Chatbot javoblaridagi qat'iy matnlarning uch tildagi variantlari.

Qo'llanmalar faqat o'zbek tilida. Model javobni so'ralgan tilda yozadi, bu
yerdagi matnlar esa modelga bog'liq bo'lmagan qat'iy javoblar uchun.
"""

from __future__ import annotations

from hr_chatbot.schemas import Language

# Promptda modelga qaysi tilda javob berishini aytish uchun.
LANGUAGE_NAMES: dict[Language, str] = {
    Language.UZ: "o'zbek",
    Language.RU: "rus",
    Language.EN: "ingliz",
}

# Salomlashuvga javob: rad etish emas, qisqa tanishtiruv va namuna savol.
GREETING: dict[Language, str] = {
    Language.UZ: (
        "Assalomu alaykum! Men ERP HR tizimi bo'yicha yordamchiman. "
        "Tizimdan qanday foydalanish haqida savolingizni yozing — "
        "masalan, «Ta'til so'rovini qanday yuboraman?»"
    ),
    Language.RU: (
        "Здравствуйте! Я помощник по системе ERP HR. "
        "Напишите вопрос о том, как пользоваться системой — "
        "например, «Как отправить заявку на отпуск?»"
    ),
    Language.EN: (
        "Hello! I am the ERP HR system assistant. "
        "Ask how to use the system — "
        "for example, «How do I submit a leave request?»"
    ),
}

OUT_OF_DOMAIN: dict[Language, str] = {
    Language.UZ: "Iltimos, ERP HR tizimini qanday ishlatish bo'yicha aniqroq savol bering.",
    Language.RU: "Пожалуйста, задайте более конкретный вопрос о том, как пользоваться системой ERP HR.",
    Language.EN: "Please ask a more specific question about how to use the ERP HR system.",
}

UNDOCUMENTED_PAGE: dict[Language, str] = {
    Language.UZ: "Qo'llanmada «{name}» bo'limi tavsiflanmagan, lekin bu bo'lim tizimda mavjud.",
    Language.RU: "В руководстве раздел «{name}» не описан, но этот раздел есть в системе.",
    Language.EN: "The manual does not describe the «{name}» section, but the section exists in the system.",
}

# Modelning javobiga o'zidan qo'shgan eslatmasini kesish uchun (uch tilda).
NOTE_PREFIXES: tuple[str, ...] = ("Eslatma", "Примечание", "Note")
