"""ERP HR bo'limlarining havolalari.

Havola qo'llanmadagi bo'lim sarlavhasiga bog'lanadi. Avval to'liq yo'l
("Katta bo'lim > Ichki bo'lim") tekshiriladi, topilmasa katta bo'limning
havolasi olinadi. Shuning uchun alohida sahifasi bor ichki bo'lim (masalan
"7. Hujjatlar > Imzolarni tekshirish") o'z havolasini, alohida sahifasi
yo'qlari esa katta bo'limning havolasini oladi.
"""

from __future__ import annotations

from dataclasses import dataclass

from hr_chatbot.schemas import ROLE_SEARCH_SCOPE, HRRole

SITE_BASE_URL = "https://hr-erp.tenzorsoft.uz"
APP_BASE_URL = f"{SITE_BASE_URL}/app"
# Login sahifasi ilova ichida emas, alohida yo'lda turadi.
LOGIN_URL = f"{SITE_BASE_URL}/auth/login"


def _url(path: str) -> str:
    return f"{APP_BASE_URL}/{path}"


# Xodim qo'llanmasi guruhli tuzilmada: UMUMIY, MENING KABINETIM, Qo'shimcha.
# Sahifalar guruh ichida H2 bo'lgani uchun kalit to'liq yo'l ("Guruh > Sahifa").
_EMPLOYEE_URLS: dict[str, str | None] = {
    # UMUMIY guruhi
    "1. UMUMIY": _url("employeeDashboard"),
    "1. UMUMIY > Dashboard (yangiliklar, e'lonlar, tug'ilgan kunlar)": _url("employeeDashboard"),
    "1. UMUMIY > Forum": _url("forum"),
    # Umumiy tugmalar izohi — alohida sahifa emas, havola berilmaydi.
    "1. UMUMIY > Barcha bo'limlarda uchraydigan umumiy tugmalar": None,
    # MENING KABINETIM guruhi — har bir sahifa o'z havolasiga ega
    "2. MENING KABINETIM": _url("myProfile"),
    # Profil, Mening KPI, Qarindoshlar, Ta'lim, Hujjatlar, Ish tajribasi —
    # bularning hammasi Profil sahifasi ichidagi tablarda ochiladi.
    "2. MENING KABINETIM > Profil": _url("myProfile"),
    "2. MENING KABINETIM > Mening so'rovlarim": _url("myAffairs"),
    "2. MENING KABINETIM > Mening davomatim": _url("myHikvisionAttendance"),
    "2. MENING KABINETIM > Mening kechikishlarim": _url("myEmployeeLatencies"),
    "2. MENING KABINETIM > Mening kalendarim": _url("myActivityCalendar"),
    "2. MENING KABINETIM > Mening navbatchiligim": _url("myDutySchedule"),
    "2. MENING KABINETIM > Mening ta'tillarim": _url("myVacations"),
    "2. MENING KABINETIM > Xabarlar": _url("myMessaging"),
    "2. MENING KABINETIM > Oshxona": _url("myCanteen"),
    "2. MENING KABINETIM > Suhbatlarim": _url("myInterviews"),
    "2. MENING KABINETIM > Rag'batlantirish": _url("myRecognitions"),
    "2. MENING KABINETIM > Intizomiy choralar": _url("myDisciplinaryActions"),
    "2. MENING KABINETIM > Kompensatsiyalar": _url("compensation"),
    "2. MENING KABINETIM > Imtiyozlar": _url("benefits"),
    "2. MENING KABINETIM > Hisob varaqalari": _url("payslips"),
    "2. MENING KABINETIM > Buyruqlar": _url("hrOrders"),
    # Qo'shimcha bo'limlar — umumiy maslahatlar, yagona sahifasi yo'q.
    "3. Qo'shimcha bo'limlar": None,
}

# Filial rahbari va Owner bitta qo'llanma tuzilmasidan foydalanadi. Bu yerda
# sahifalar ichki bo'lim (H2) darajasida, chunki bitta katta bo'lim bir nechta
# sahifaga bo'linadi (masalan "6. Ish haqi" ichida davrlar, qoidalar, diapazonlar).
_ADMIN_URLS: dict[str, str | None] = {
    "1. UMUMIY": _url("dashboard"),
    "1. UMUMIY > Tizimga kirish": LOGIN_URL,
    "1. UMUMIY > Dashboard (yangiliklar, e'lonlar, tug'ilgan kunlar)": _url("employeeDashboard"),
    "1. UMUMIY > Forum": _url("forum"),
    # Umumiy tugmalar izohi — alohida sahifa emas, havola berilmaydi.
    "1. UMUMIY > Barcha bo'limlarda uchraydigan umumiy tugmalar": None,
    # "Shtat va tuzilma" — barcha ichki bo'limlar bitta workforce sahifasidagi
    # tablar (hash-anchor bilan).
    "2. Kadrlar — Tashkiliy tuzilma (Shtat va tuzilma)": _url("workforce"),
    "2. Kadrlar — Tashkiliy tuzilma (Shtat va tuzilma) > Ierarxiya": _url("workforce#tab=hierarchy"),
    "2. Kadrlar — Tashkiliy tuzilma (Shtat va tuzilma) > Kompaniyalar": _url("workforce#tab=companies"),
    "2. Kadrlar — Tashkiliy tuzilma (Shtat va tuzilma) > Filiallar": _url("workforce#tab=branches"),
    "2. Kadrlar — Tashkiliy tuzilma (Shtat va tuzilma) > Bo'limlar": _url("workforce#tab=departments"),
    "2. Kadrlar — Tashkiliy tuzilma (Shtat va tuzilma) > Lavozimlar": _url("workforce#tab=positions"),
    "2. Kadrlar — Tashkiliy tuzilma (Shtat va tuzilma) > Shtat jadvali": _url("workforce#tab=staffing"),
    "2. Kadrlar — Tashkiliy tuzilma (Shtat va tuzilma) > Ish grafiklari": _url("workforce#tab=work-schedules"),
    "2. Kadrlar — Tashkiliy tuzilma (Shtat va tuzilma) > Oshxona boshqaruvi": _url("workforce#tab=canteen"),
    "3. Kadrlar — Xodimlar": _url("employees"),
    "3. Kadrlar — Xodimlar > Xodimlar kalendari": _url("employeeActivityCalendar"),
    "4. Kadrlar — Ishga olish (Vakansiya)": _url("recruitment"),
    "5. Kadrlar — HR operatsiyalari": _url("operations"),
    "6. Ish haqi > DAVRLAR (hisob davrini ochish)": _url("payrollPeriods"),
    "6. Ish haqi > Hisob varaqasini tekshirish": _url("payslips"),
    "6. Ish haqi > Hisoblash qoidasini sozlash": _url("payrollRules"),
    "6. Ish haqi > Kompensatsiyalarni yuritish": _url("compensation"),
    "6. Ish haqi > Imtiyozlarni yuritish": _url("benefits"),
    "6. Ish haqi > Maosh diapazonini sozlash": _url("salaryBands"),
    "6. Ish haqi > Maosh ko'rib chiqish": _url("salaryCycles"),
    "6. Ish haqi > Tuzatish kiritish": _url("payrollAdjustments"),
    "7. Hujjatlar > Buyruqlarni tekshirish": _url("hrOrders"),
    "7. Hujjatlar > Shablon tayyorlash": _url("templates"),
    "7. Hujjatlar > Hujjat harakatini kuzatish": _url("documentFlow"),
    "7. Hujjatlar > Imzolarni tekshirish": _url("signatures"),
    "7. Hujjatlar > Hisobot tayyorlash": _url("reports"),
    "7. Hujjatlar > Ma'lumot almashinuvi (Import/eksport)": _url("dataExchange"),
    "8. Tizim > Foydalanuvchilarni boshqarish": _url("users"),
    "8. Tizim > Foydalanuvchi profilini tekshirish": _url("users"),
    "8. Tizim > Foydalanuvchi ruxsatlari": _url("companyRoles"),
}

# HR Manager guruhli tuzilmada: UMUMIY, MENING KABINETIM (Xodim bilan bir xil
# shaxsiy sahifalar), MENING JAMOAM (Xodimlar / Shtat va tuzilma / kalendar), TIZIM.
_DEPARTMENT_MANAGER_URLS: dict[str, str | None] = {
    # UMUMIY
    "1. UMUMIY": _url("employeeDashboard"),
    "1. UMUMIY > Dashboard (yangiliklar, e'lonlar, tug'ilgan kunlar)": _url("employeeDashboard"),
    "1. UMUMIY > Forum": _url("forum"),
    # Umumiy tugmalar izohi — alohida sahifa emas, havola berilmaydi.
    "1. UMUMIY > Barcha bo'limlarda uchraydigan umumiy tugmalar": None,
    # MENING KABINETIM — shaxsiy sahifalar (Xodim bilan bir xil havolalar)
    "2. MENING KABINETIM": _url("myProfile"),
    "2. MENING KABINETIM > Profil": _url("myProfile"),
    "2. MENING KABINETIM > Mening so'rovlarim": _url("myAffairs"),
    "2. MENING KABINETIM > Mening davomatim": _url("myHikvisionAttendance"),
    "2. MENING KABINETIM > Mening kechikishlarim": _url("myEmployeeLatencies"),
    "2. MENING KABINETIM > Mening kalendarim": _url("myActivityCalendar"),
    "2. MENING KABINETIM > Mening navbatchiligim": _url("myDutySchedule"),
    "2. MENING KABINETIM > Mening ta'tillarim": _url("myVacations"),
    "2. MENING KABINETIM > Xabarlar": _url("myMessaging"),
    "2. MENING KABINETIM > Oshxona": _url("myCanteen"),
    "2. MENING KABINETIM > Suhbatlarim": _url("myInterviews"),
    "2. MENING KABINETIM > Rag'batlantirish": _url("myRecognitions"),
    "2. MENING KABINETIM > Intizomiy choralar": _url("myDisciplinaryActions"),
    "2. MENING KABINETIM > Kompensatsiyalar": _url("compensation"),
    "2. MENING KABINETIM > Imtiyozlar": _url("benefits"),
    "2. MENING KABINETIM > Hisob varaqalari": _url("payslips"),
    "2. MENING KABINETIM > Buyruqlar": _url("hrOrders"),
    # MENING JAMOAM
    "3. MENING JAMOAM": _url("employees"),
    "3. MENING JAMOAM > Xodimlar (KPI, hujjatlar: pasport, mehnat shartnomasi, sertifikat, tibbiy ma'lumotnoma, harbiy guvohnoma; xizmat safari, so'rovlar, davomat, kechikish, navbatchilik, ta'til)": _url("employees"),
    "3. MENING JAMOAM > Shtat va tuzilma": _url("workforce"),
    "3. MENING JAMOAM > Xodimlar kalendari": _url("employeeActivityCalendar"),
    # TIZIM
    "4. TIZIM": _url("users"),
    "4. TIZIM > Foydalanuvchilar": _url("users"),
}

# Kalit: qo'llanmadagi to'liq yo'l yoki katta bo'lim sarlavhasi.
# Qiymati None bo'lsa, bo'limning yagona sahifasi yo'q va havola berilmaydi.
SECTION_URLS: dict[HRRole, dict[str, str | None]] = {
    HRRole.EMPLOYEE: _EMPLOYEE_URLS,
    HRRole.BRANCH_MANAGER: _ADMIN_URLS,
    HRRole.OWNER: _ADMIN_URLS,
    HRRole.DEPARTMENT_MANAGER: _DEPARTMENT_MANAGER_URLS,
}


@dataclass(frozen=True, slots=True)
class UndocumentedPage:
    """A page that exists in ERP HR but is missing from the manuals."""

    name: str
    keywords: tuple[str, ...]
    url: str


# Qo'llanmada tavsiflanmagan, lekin tizimda mavjud sahifalar. Bularga oid savolga
# qidiruv hech narsa topa olmaydi, shuning uchun havolaning o'zi qaytariladi.
# "Jihoz reyestri" — workforce sahifasidagi tab, qo'llanmada tavsiflanmagan.
_ASSET_REGISTRY = UndocumentedPage(
    name="Jihoz reyestri",
    keywords=("jihoz reyestri", "jihozlar reyestri", "jihoz reestri", "asset registry", "asset-registry"),
    url=_url("workforce#tab=asset-registry"),
)

UNDOCUMENTED_PAGES: dict[HRRole, tuple[UndocumentedPage, ...]] = {
    HRRole.EMPLOYEE: (),
    HRRole.BRANCH_MANAGER: (_ASSET_REGISTRY,),
    HRRole.OWNER: (_ASSET_REGISTRY,),
    HRRole.DEPARTMENT_MANAGER: (_ASSET_REGISTRY,),
}


def resolve_section_url(role: HRRole, title_path: str) -> str | None:
    """Return the app link of the section a manual chunk belongs to."""

    if not title_path:
        return None
    urls = SECTION_URLS.get(role, {})
    if title_path in urls:
        return urls[title_path]
    return urls.get(title_path.split(" > ")[0].strip())


def match_undocumented_page(role: HRRole, question: str) -> UndocumentedPage | None:
    """Find a known ERP HR page the manuals do not describe."""

    lowered = question.lower()
    best: UndocumentedPage | None = None
    best_length = 0
    for scope_role in ROLE_SEARCH_SCOPE.get(role, (role,)):
        for page in UNDOCUMENTED_PAGES.get(scope_role, ()):
            for keyword in page.keywords:
                if keyword in lowered and len(keyword) > best_length:
                    best = page
                    best_length = len(keyword)
    return best
