from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

import os
import logging
import uuid
import bcrypt
import jwt
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Literal

from fastapi import FastAPI, APIRouter, HTTPException, Depends, Request
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, EmailStr

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

app = FastAPI(title="21-ASR Raqamli Xizmatlar Markazi API")
api_router = APIRouter(prefix="/api")

JWT_ALGORITHM = "HS256"
JWT_SECRET = os.environ["JWT_SECRET"]
ACCESS_TOKEN_EXPIRE_DAYS = 30


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: str, email: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS),
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_user(u: dict) -> dict:
    return {
        "id": u["id"],
        "first_name": u.get("first_name", ""),
        "last_name": u.get("last_name", ""),
        "phone": u.get("phone", ""),
        "email": u.get("email", ""),
        "role": u.get("role", "user"),
        "language": u.get("language", "uz"),
        "kyc_status": u.get("kyc_status", "none"),
        "kyc_note": u.get("kyc_note", ""),
        "created_at": u.get("created_at"),
    }


def strip_file_content(files: list) -> list:
    return [
        {"name": f.get("name"), "size": f.get("size"), "mime": f.get("mime"), "field": f.get("field")}
        for f in (files or [])
    ]


def serialize_request(r: dict, include_files: bool = False) -> dict:
    out = {k: v for k, v in r.items() if k != "_id"}
    if not include_files:
        out["files"] = strip_file_content(out.get("files", []))
    return out


async def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth[7:]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await db.users.find_one({"id": payload["sub"]}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    user.pop("password_hash", None)
    return user


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ================= Models =================
class RegisterIn(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    last_name: str = Field(min_length=1, max_length=100)
    phone: str = Field(min_length=5, max_length=30)
    password: str = Field(min_length=6, max_length=128)


class LoginIn(BaseModel):
    phone: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    language: Optional[str] = None


class FileItem(BaseModel):
    name: str
    mime: str
    size: int
    content: str
    field: str = "document"


class KycSubmitIn(BaseModel):
    passport_photo: FileItem
    selfie_photo: FileItem


class KycDecisionIn(BaseModel):
    decision: Literal["approved", "rejected"]
    note: Optional[str] = ""


class ServiceRequestIn(BaseModel):
    category: Literal["single_window", "accounting"]
    service_id: str
    service_title: str
    form_data: dict = {}
    documents: List[FileItem] = []
    ekey_files: List[FileItem] = []


class StatusUpdateIn(BaseModel):
    status: Literal["pending", "in_review", "approved", "rejected"]
    admin_note: Optional[str] = None
    payment_required: Optional[bool] = None
    payment_amount: Optional[float] = None
    payment_note: Optional[str] = None


class PaymentMethodIn(BaseModel):
    name: str
    kind: Literal["qr", "link"]
    value: str  # base64 image OR URL
    description: Optional[str] = ""
    order: int = 0
    active: bool = True


class PaymentMarkIn(BaseModel):
    payment_status: Literal["paid", "none", "required"]


class ContactItemIn(BaseModel):
    label: str
    icon: str = "circle"  # phone, mail, telegram, instagram, whatsapp, facebook, globe, map-pin, clock, etc.
    value: str
    href: Optional[str] = ""  # tel:, mailto:, https://t.me/...
    order: int = 0
    active: bool = True


class RatingIn(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = ""


class SupportMessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class AdminSupportMessageIn(BaseModel):
    user_id: str
    text: str = Field(min_length=1, max_length=2000)


class TeamMemberIn(BaseModel):
    name: str
    role: str
    email: Optional[str] = ""
    phone: Optional[str] = ""
    photo: Optional[str] = ""  # base64
    order: int = 0
    bio: Optional[str] = ""


class ContactInfoIn(BaseModel):
    address_uz: Optional[str] = ""
    address_ru: Optional[str] = ""
    address_en: Optional[str] = ""
    phone: Optional[str] = ""
    short_number: Optional[str] = ""
    email: Optional[str] = ""
    working_hours: Optional[str] = ""
    telegram: Optional[str] = ""
    website: Optional[str] = ""


# ================= Service Catalog =================
SERVICE_CATALOG = {
    "single_window": [
        {"id": "sw_passport", "title_uz": "Pasport olish / almashtirish", "title_ru": "Паспорт: получение / замена", "title_en": "Passport Issuance / Replacement", "icon": "id-card"},
        {"id": "sw_id_card", "title_uz": "ID-karta", "title_ru": "ID-карта", "title_en": "ID Card", "icon": "credit-card"},
        {"id": "sw_birth_cert", "title_uz": "Tug'ilganlik guvohnomasi", "title_ru": "Свидетельство о рождении", "title_en": "Birth Certificate", "icon": "baby"},
        {"id": "sw_marriage", "title_uz": "Nikoh guvohnomasi", "title_ru": "Свидетельство о браке", "title_en": "Marriage Certificate", "icon": "heart"},
        {"id": "sw_divorce", "title_uz": "Ajrashish guvohnomasi", "title_ru": "Свидетельство о разводе", "title_en": "Divorce Certificate", "icon": "heart-off"},
        {"id": "sw_address", "title_uz": "Propiska / Manzilni ro'yxatdan o'tkazish", "title_ru": "Прописка / Регистрация адреса", "title_en": "Address Registration", "icon": "map-pin"},
        {"id": "sw_license", "title_uz": "Haydovchilik guvohnomasi", "title_ru": "Водительские права", "title_en": "Driver License", "icon": "car"},
        {"id": "sw_auto_reg", "title_uz": "Avtomobil ro'yxatdan o'tkazish", "title_ru": "Регистрация автомобиля", "title_en": "Vehicle Registration", "icon": "car-front"},
        {"id": "sw_pension", "title_uz": "Pensiya tayinlash", "title_ru": "Назначение пенсии", "title_en": "Pension Application", "icon": "wallet"},
        {"id": "sw_child_support", "title_uz": "Bolalar nafaqasi", "title_ru": "Пособие на детей", "title_en": "Child Support Allowance", "icon": "baby"},
        {"id": "sw_social_aid", "title_uz": "Ijtimoiy yordam", "title_ru": "Социальная помощь", "title_en": "Social Aid", "icon": "hand-helping"},
        {"id": "sw_med_cert", "title_uz": "Tibbiy sertifikat (Ma'lumotnoma 086)", "title_ru": "Мед. справка (086)", "title_en": "Medical Certificate (Form 086)", "icon": "heart-pulse"},
        {"id": "sw_edu_docs", "title_uz": "Ta'lim hujjatlari tasdiqlash (Apostille)", "title_ru": "Апостиль / заверение документов об образовании", "title_en": "Education Document Apostille", "icon": "graduation-cap"},
        {"id": "sw_property", "title_uz": "Ko'chmas mulk ro'yxati", "title_ru": "Регистрация недвижимости", "title_en": "Real Estate Registration", "icon": "home"},
        {"id": "sw_migration", "title_uz": "Migratsiya xizmatlari (viza / ruxsatnoma)", "title_ru": "Миграционные услуги", "title_en": "Migration Services", "icon": "plane"},
        {"id": "sw_court_ref", "title_uz": "Sud qaroridan nusxa", "title_ru": "Копия судебного решения", "title_en": "Court Decision Copy", "icon": "scale"},
        {"id": "sw_notary", "title_uz": "Notarial xizmatlar", "title_ru": "Нотариальные услуги", "title_en": "Notary Services", "icon": "stamp"},
        {"id": "sw_military", "title_uz": "Harbiy hisob guvohnomasi", "title_ru": "Военный билет", "title_en": "Military ID", "icon": "shield"},
    ],
    "accounting": [
        {"id": "ac_tax_report", "title_uz": "Soliq hisoboti topshirish", "title_ru": "Налоговая отчётность", "title_en": "Tax Report Submission", "icon": "file-text"},
        {"id": "ac_financial_report", "title_uz": "Moliyaviy hisobot", "title_ru": "Финансовая отчётность", "title_en": "Financial Report", "icon": "trending-up"},
        {"id": "ac_corp_open", "title_uz": "Korxona ochish (MChJ, XK va b.)", "title_ru": "Открытие предприятия (ООО и др.)", "title_en": "Enterprise Registration", "icon": "building-2"},
        {"id": "ac_corp_close", "title_uz": "Korxona yopish (likvidatsiya)", "title_ru": "Ликвидация предприятия", "title_en": "Enterprise Liquidation", "icon": "building"},
        {"id": "ac_charter", "title_uz": "Ustav o'zgartirish", "title_ru": "Изменение устава", "title_en": "Charter Amendment", "icon": "file-edit"},
        {"id": "ac_yatt_open", "title_uz": "YaTT ochish", "title_ru": "Открытие ИП (ЯТТ)", "title_en": "Sole Proprietor (YaTT) Registration", "icon": "user-plus"},
        {"id": "ac_yatt_close", "title_uz": "YaTT yopish", "title_ru": "Закрытие ИП (ЯТТ)", "title_en": "Sole Proprietor (YaTT) Closure", "icon": "user-x"},
        {"id": "ac_yatt_freeze", "title_uz": "YaTT faoliyatini muzlatish", "title_ru": "Приостановка ИП (ЯТТ)", "title_en": "Sole Proprietor (YaTT) Suspension", "icon": "pause-circle"},
        {"id": "ac_cash_sell", "title_uz": "Kassa apparat sotish", "title_ru": "Продажа кассового аппарата", "title_en": "Cash Register Sale", "icon": "calculator"},
        {"id": "ac_cash_support", "title_uz": "Kassa apparat texnik xizmati", "title_ru": "Тех. обслуживание кассового аппарата", "title_en": "Cash Register Tech Support", "icon": "wrench"},
        {"id": "ac_salary", "title_uz": "Ish haqi hisob-kitob xizmati", "title_ru": "Расчёт заработной платы", "title_en": "Payroll Services", "icon": "banknote"},
        {"id": "ac_consult", "title_uz": "Buxgalterlik maslahat", "title_ru": "Бухгалтерские консультации", "title_en": "Accounting Consultation", "icon": "message-circle"},
        {"id": "ac_audit", "title_uz": "Audit", "title_ru": "Аудит", "title_en": "Audit", "icon": "search-check"},
        {"id": "ac_ekey", "title_uz": "ERI (elektron raqamli imzo)", "title_ru": "ЭЦП (электронная цифровая подпись)", "title_en": "Digital Signature (ERI)", "icon": "key"},
    ],
}


# ================= Auth =================
@api_router.post("/auth/register", response_model=TokenOut)
async def register(body: RegisterIn):
    if await db.users.find_one({"phone": body.phone}):
        raise HTTPException(status_code=400, detail="Phone already registered")
    user_doc = {
        "id": str(uuid.uuid4()),
        "first_name": body.first_name.strip(),
        "last_name": body.last_name.strip(),
        "phone": body.phone.strip(),
        "email": "",
        "password_hash": hash_password(body.password),
        "role": "user",
        "language": "uz",
        "kyc_status": "none",
        "kyc_note": "",
        "created_at": now_iso(),
    }
    await db.users.insert_one(user_doc)
    user_doc.pop("_id", None); user_doc.pop("password_hash", None)
    token = create_access_token(user_doc["id"], user_doc["phone"], user_doc["role"])
    return TokenOut(access_token=token, user=serialize_user(user_doc))


@api_router.post("/auth/login", response_model=TokenOut)
async def login(body: LoginIn):
    phone = body.phone.strip()
    user = await db.users.find_one({"phone": phone})
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user["id"], user["phone"], user.get("role", "user"))
    return TokenOut(access_token=token, user=serialize_user(user))


@api_router.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return serialize_user(user)


@api_router.patch("/auth/me")
async def update_me(body: UserUpdate, user: dict = Depends(get_current_user)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    if updates:
        await db.users.update_one({"id": user["id"]}, {"$set": updates})
    fresh = await db.users.find_one({"id": user["id"]}, {"_id": 0, "password_hash": 0})
    return serialize_user(fresh)


# ================= KYC =================
@api_router.post("/kyc/submit")
async def kyc_submit(body: KycSubmitIn, user: dict = Depends(get_current_user)):
    now = now_iso()
    await db.users.update_one(
        {"id": user["id"]},
        {"$set": {
            "kyc_status": "pending",
            "kyc_note": "",
            "kyc_submitted_at": now,
        }}
    )
    await db.kyc_records.update_one(
        {"user_id": user["id"]},
        {"$set": {
            "user_id": user["id"],
            "passport_photo": body.passport_photo.dict(),
            "selfie_photo": body.selfie_photo.dict(),
            "status": "pending",
            "submitted_at": now,
            "note": "",
        }},
        upsert=True,
    )
    return {"status": "pending", "submitted_at": now}


@api_router.get("/kyc/me")
async def kyc_me(user: dict = Depends(get_current_user)):
    rec = await db.kyc_records.find_one({"user_id": user["id"]}, {"_id": 0})
    return {
        "status": user.get("kyc_status", "none"),
        "note": user.get("kyc_note", ""),
        "submitted_at": rec.get("submitted_at") if rec else None,
        "has_passport": bool(rec and rec.get("passport_photo")),
        "has_selfie": bool(rec and rec.get("selfie_photo")),
    }


@api_router.get("/admin/kyc")
async def admin_kyc_list(status: Optional[str] = None, admin: dict = Depends(require_admin)):
    q = {"kyc_status": {"$in": ["pending", "approved", "rejected"]}}
    if status:
        q = {"kyc_status": status}
    cursor = db.users.find(q, {"_id": 0, "password_hash": 0}).sort("kyc_submitted_at", -1)
    users = await cursor.to_list(500)
    return [serialize_user(u) | {"kyc_submitted_at": u.get("kyc_submitted_at")} for u in users]


@api_router.get("/admin/kyc/{user_id}")
async def admin_kyc_detail(user_id: str, admin: dict = Depends(require_admin)):
    rec = await db.kyc_records.find_one({"user_id": user_id}, {"_id": 0})
    u = await db.users.find_one({"id": user_id}, {"_id": 0, "password_hash": 0})
    if not rec or not u:
        raise HTTPException(status_code=404, detail="Not found")
    return {"user": serialize_user(u), "record": rec}


@api_router.patch("/admin/kyc/{user_id}")
async def admin_kyc_decide(user_id: str, body: KycDecisionIn, admin: dict = Depends(require_admin)):
    u = await db.users.find_one({"id": user_id})
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    await db.users.update_one({"id": user_id}, {"$set": {"kyc_status": body.decision, "kyc_note": body.note or ""}})
    await db.kyc_records.update_one({"user_id": user_id}, {"$set": {"status": body.decision, "note": body.note or "", "decided_at": now_iso()}})
    return {"ok": True, "status": body.decision}


# ================= Services =================
@api_router.get("/services/catalog")
async def get_catalog():
    return SERVICE_CATALOG


# ================= Requests =================
@api_router.post("/requests")
async def create_request(body: ServiceRequestIn, user: dict = Depends(get_current_user)):
    if user.get("kyc_status") != "approved":
        raise HTTPException(status_code=403, detail="KYC not approved. Please complete identification first.")
    files = [f.dict() for f in body.documents] + [f.dict() | {"field": "ekey"} for f in body.ekey_files]
    now = now_iso()
    req = {
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "user_name": f"{user['first_name']} {user['last_name']}",
        "user_email": user.get("email", ""),
        "user_phone": user["phone"],
        "category": body.category,
        "service_id": body.service_id,
        "service_title": body.service_title,
        "form_data": body.form_data,
        "files": files,
        "status": "pending",
        "admin_note": "",
        "rating": None,
        "rating_comment": "",
        "closed": False,
        "payment_required": False,
        "payment_amount": 0,
        "payment_note": "",
        "payment_status": "none",  # none | required | paid
        "created_at": now,
        "updated_at": now,
    }
    await db.requests.insert_one(req)
    return serialize_request(req)


@api_router.get("/requests/mine")
async def my_requests(user: dict = Depends(get_current_user)):
    cursor = db.requests.find({"user_id": user["id"]}, {"_id": 0}).sort("created_at", -1)
    items = await cursor.to_list(500)
    return [serialize_request(r) for r in items]


@api_router.get("/requests/{request_id}")
async def get_request(request_id: str, user: dict = Depends(get_current_user)):
    r = await db.requests.find_one({"id": request_id}, {"_id": 0})
    # 436-qatordan boshlab shu 5 qatorni yozing (eskisini ustidan):
    if not r:
        raise HTTPException(status_code=404, detail="Request not found")
    if user.get("role") != "admin" and r["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return r


@api_router.post("/requests/{request_id}/rate")
async def rate_request(request_id: str, body: RatingIn, user: dict = Depends(get_current_user)):
    r = await db.requests.find_one({"id": request_id})
    if not r:
        raise HTTPException(status_code=404, detail="Not found")
    if r["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    if r.get("status") != "approved":
        raise HTTPException(status_code=400, detail="Can rate only approved requests")
    if r.get("rating"):
        raise HTTPException(status_code=400, detail="Already rated")
    await db.requests.update_one(
        {"id": request_id},
        {"$set": {
            "rating": body.rating,
            "rating_comment": body.comment or "",
            "closed": True,
            "rated_at": now_iso(),
            "updated_at": now_iso(),
        }},
    )
    fresh = await db.requests.find_one({"id": request_id}, {"_id": 0})
    return serialize_request(fresh)


# ================= Admin Requests =================
@api_router.get("/admin/requests")
async def admin_list(
    status: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
    admin: dict = Depends(require_admin),
):
    query = {}
    if status: query["status"] = status
    if category: query["category"] = category
    if search:
        query["$or"] = [
            {"user_name": {"$regex": search, "$options": "i"}},
            {"user_email": {"$regex": search, "$options": "i"}},
            {"user_phone": {"$regex": search, "$options": "i"}},
            {"service_title": {"$regex": search, "$options": "i"}},
        ]
    cursor = db.requests.find(query, {"_id": 0}).sort("created_at", -1)
    items = await cursor.to_list(1000)
    return [serialize_request(r) for r in items]


@api_router.patch("/admin/requests/{request_id}/status")
async def admin_update_status(request_id: str, body: StatusUpdateIn, admin: dict = Depends(require_admin)):
    r = await db.requests.find_one({"id": request_id})
    if not r:
        raise HTTPException(status_code=404, detail="Request not found")
    update = {"status": body.status, "updated_at": now_iso()}
    if body.admin_note is not None:
        update["admin_note"] = body.admin_note
    if body.payment_required is not None:
        update["payment_required"] = body.payment_required
        update["payment_status"] = "required" if body.payment_required else "none"
    if body.payment_amount is not None:
        update["payment_amount"] = body.payment_amount
    if body.payment_note is not None:
        update["payment_note"] = body.payment_note
    await db.requests.update_one({"id": request_id}, {"$set": update})
    # Notification
    title_map = {"pending": "Kutilmoqda", "in_review": "Ko'rib chiqilmoqda", "approved": "Tasdiqlandi", "rejected": "Rad etildi"}
    kind = "payment" if body.payment_required else "status"
    title = f"{r.get('service_title', 'Ariza')} — {title_map.get(body.status, body.status)}"
    if body.payment_required:
        title = f"To'lov kerak: {body.payment_amount or 0} so'm"
    notif = {
        "id": str(uuid.uuid4()),
        "user_id": r["user_id"],
        "title": title,
        "body": body.admin_note or body.payment_note or "",
        "request_id": request_id,
        "kind": kind,
        "read": False,
        "created_at": now_iso(),
    }
    await db.notifications.insert_one(notif)
    fresh = await db.requests.find_one({"id": request_id}, {"_id": 0})
    return serialize_request(fresh)


@api_router.get("/notifications/mine")
async def my_notifications(user: dict = Depends(get_current_user)):
    cursor = db.notifications.find({"user_id": user["id"]}, {"_id": 0}).sort("created_at", -1)
    items = await cursor.to_list(200)
    return items


@api_router.get("/notifications/unread-count")
async def unread_count(user: dict = Depends(get_current_user)):
    n = await db.notifications.count_documents({"user_id": user["id"], "read": False})
    return {"count": n}


@api_router.post("/notifications/{nid}/read")
async def mark_notif_read(nid: str, user: dict = Depends(get_current_user)):
    await db.notifications.update_one({"id": nid, "user_id": user["id"]}, {"$set": {"read": True}})
    return {"ok": True}


@api_router.post("/notifications/read-all")
async def mark_all_read(user: dict = Depends(get_current_user)):
    await db.notifications.update_many({"user_id": user["id"], "read": False}, {"$set": {"read": True}})
    return {"ok": True}


@api_router.post("/requests/{request_id}/mark-paid")
async def user_mark_paid(request_id: str, user: dict = Depends(get_current_user)):
    r = await db.requests.find_one({"id": request_id})
    if not r:
        raise HTTPException(status_code=404, detail="Not found")
    if r["user_id"] != user["id"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not r.get("payment_required"):
        raise HTTPException(status_code=400, detail="No payment required")
    await db.requests.update_one(
        {"id": request_id},
        {"$set": {"payment_status": "paid", "paid_at": now_iso(), "updated_at": now_iso()}},
    )
    fresh = await db.requests.find_one({"id": request_id}, {"_id": 0})
    return serialize_request(fresh)


@api_router.patch("/admin/requests/{request_id}/payment")
async def admin_mark_payment(request_id: str, body: PaymentMarkIn, admin: dict = Depends(require_admin)):
    r = await db.requests.find_one({"id": request_id})
    if not r:
        raise HTTPException(status_code=404, detail="Not found")
    await db.requests.update_one(
        {"id": request_id},
        {"$set": {"payment_status": body.payment_status, "updated_at": now_iso()}},
    )
    fresh = await db.requests.find_one({"id": request_id}, {"_id": 0})
    return serialize_request(fresh)


# ================= Payment Methods =================
@api_router.get("/payments/methods")
async def list_payments(user: dict = Depends(get_current_user)):
    methods = await db.payment_methods.find({"active": True}, {"_id": 0}).sort("order", 1).to_list(100)
    return methods


@api_router.get("/admin/payments")
async def admin_list_payments(admin: dict = Depends(require_admin)):
    methods = await db.payment_methods.find({}, {"_id": 0}).sort("order", 1).to_list(100)
    return methods


@api_router.post("/admin/payments")
async def admin_add_payment(body: PaymentMethodIn, admin: dict = Depends(require_admin)):
    item = body.dict() | {"id": str(uuid.uuid4()), "created_at": now_iso()}
    await db.payment_methods.insert_one(item)
    item.pop("_id", None)
    return item


@api_router.patch("/admin/payments/{pid}")
async def admin_edit_payment(pid: str, body: PaymentMethodIn, admin: dict = Depends(require_admin)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    r = await db.payment_methods.update_one({"id": pid}, {"$set": updates})
    if r.matched_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    item = await db.payment_methods.find_one({"id": pid}, {"_id": 0})
    return item


@api_router.delete("/admin/payments/{pid}")
async def admin_del_payment(pid: str, admin: dict = Depends(require_admin)):
    r = await db.payment_methods.delete_one({"id": pid})
    return {"ok": r.deleted_count > 0}


# ================= Dynamic Contact Items =================
@api_router.get("/contact-items")
async def list_contact_items():
    items = await db.contact_items.find({"active": True}, {"_id": 0}).sort("order", 1).to_list(100)
    return items


@api_router.get("/admin/contact-items")
async def admin_list_contact_items(admin: dict = Depends(require_admin)):
    items = await db.contact_items.find({}, {"_id": 0}).sort("order", 1).to_list(100)
    return items


@api_router.post("/admin/contact-items")
async def admin_add_contact_item(body: ContactItemIn, admin: dict = Depends(require_admin)):
    item = body.dict() | {"id": str(uuid.uuid4()), "created_at": now_iso()}
    await db.contact_items.insert_one(item)
    item.pop("_id", None)
    return item


@api_router.patch("/admin/contact-items/{cid}")
async def admin_edit_contact_item(cid: str, body: ContactItemIn, admin: dict = Depends(require_admin)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    r = await db.contact_items.update_one({"id": cid}, {"$set": updates})
    if r.matched_count == 0:
        raise HTTPException(status_code=404, detail="Not found")
    item = await db.contact_items.find_one({"id": cid}, {"_id": 0})
    return item


@api_router.delete("/admin/contact-items/{cid}")
async def admin_del_contact_item(cid: str, admin: dict = Depends(require_admin)):
    r = await db.contact_items.delete_one({"id": cid})
    return {"ok": r.deleted_count > 0}


@api_router.get("/admin/stats")
async def admin_stats(admin: dict = Depends(require_admin)):
    total = await db.requests.count_documents({})
    pending = await db.requests.count_documents({"status": "pending"})
    in_review = await db.requests.count_documents({"status": "in_review"})
    approved = await db.requests.count_documents({"status": "approved"})
    rejected = await db.requests.count_documents({"status": "rejected"})
    users_count = await db.users.count_documents({"role": "user"})
    kyc_pending = await db.users.count_documents({"kyc_status": "pending"})
    support_open = await db.support_threads.count_documents({"unread_admin": {"$gt": 0}})
    return {
        "total": total, "pending": pending, "in_review": in_review,
        "approved": approved, "rejected": rejected, "users": users_count,
        "kyc_pending": kyc_pending, "support_open": support_open,
    }


# ================= Support Chat =================
async def _ensure_thread(user_id: str, user_name: str):
    await db.support_threads.update_one(
        {"user_id": user_id},
        {"$setOnInsert": {"user_id": user_id, "user_name": user_name, "unread_admin": 0, "unread_user": 0, "created_at": now_iso()}},
        upsert=True,
    )


@api_router.post("/support/messages")
async def post_support_msg(body: SupportMessageIn, user: dict = Depends(get_current_user)):
    await _ensure_thread(user["id"], f"{user['first_name']} {user['last_name']}")
    msg = {
        "id": str(uuid.uuid4()),
        "user_id": user["id"],
        "sender": "user",
        "text": body.text,
        "created_at": now_iso(),
    }
    await db.support_messages.insert_one(msg)
    await db.support_threads.update_one(
        {"user_id": user["id"]},
        {"$set": {"last_message": body.text, "last_at": msg["created_at"]}, "$inc": {"unread_admin": 1}},
    )
    msg.pop("_id", None)
    return msg


@api_router.get("/support/messages/mine")
async def my_support_msgs(user: dict = Depends(get_current_user)):
    cursor = db.support_messages.find({"user_id": user["id"]}, {"_id": 0}).sort("created_at", 1)
    msgs = await cursor.to_list(500)
    # Clear user unread
    await db.support_threads.update_one({"user_id": user["id"]}, {"$set": {"unread_user": 0}})
    return msgs


@api_router.get("/admin/support/threads")
async def admin_support_threads(admin: dict = Depends(require_admin)):
    cursor = db.support_threads.find({}, {"_id": 0}).sort("last_at", -1)
    return await cursor.to_list(500)


@api_router.get("/admin/support/messages/{user_id}")
async def admin_support_msgs(user_id: str, admin: dict = Depends(require_admin)):
    cursor = db.support_messages.find({"user_id": user_id}, {"_id": 0}).sort("created_at", 1)
    msgs = await cursor.to_list(1000)
    await db.support_threads.update_one({"user_id": user_id}, {"$set": {"unread_admin": 0}})
    return msgs


@api_router.post("/admin/support/messages")
async def admin_post_support_msg(body: AdminSupportMessageIn, admin: dict = Depends(require_admin)):
    target = await db.users.find_one({"id": body.user_id})
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    await _ensure_thread(body.user_id, f"{target['first_name']} {target['last_name']}")
    msg = {
        "id": str(uuid.uuid4()),
        "user_id": body.user_id,
        "sender": "admin",
        "text": body.text,
        "created_at": now_iso(),
    }
    await db.support_messages.insert_one(msg)
    await db.support_threads.update_one(
        {"user_id": body.user_id},
        {"$set": {"last_message": body.text, "last_at": msg["created_at"]}, "$inc": {"unread_user": 1}},
    )
    msg.pop("_id", None)
    return msg


# ================= About Us (Team + Contact) =================
@api_router.get("/about")
async def get_about():
    team = await db.team_members.find({}, {"_id": 0}).sort("order", 1).to_list(100)
    contact = await db.contact_info.find_one({"_id": "singleton"}, {"_id": 0})
    return {"team": team, "contact": contact or {}}


@api_router.post("/admin/team")
async def admin_add_member(body: TeamMemberIn, admin: dict = Depends(require_admin)):
    item = body.dict() | {"id": str(uuid.uuid4()), "created_at": now_iso()}
    await db.team_members.insert_one(item)
    item.pop("_id", None)
    return item


@api_router.patch("/admin/team/{member_id}")
async def admin_update_member(member_id: str, body: TeamMemberIn, admin: dict = Depends(require_admin)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    r = await db.team_members.update_one({"id": member_id}, {"$set": updates})
    if r.matched_count == 0:
        raise HTTPException(status_code=404, detail="Member not found")
    item = await db.team_members.find_one({"id": member_id}, {"_id": 0})
    return item


@api_router.delete("/admin/team/{member_id}")
async def admin_delete_member(member_id: str, admin: dict = Depends(require_admin)):
    r = await db.team_members.delete_one({"id": member_id})
    return {"ok": r.deleted_count > 0}


@api_router.patch("/admin/contact")
async def admin_update_contact(body: ContactInfoIn, admin: dict = Depends(require_admin)):
    updates = {k: v for k, v in body.dict().items() if v is not None}
    await db.contact_info.update_one({"_id": "singleton"}, {"$set": updates}, upsert=True)
    fresh = await db.contact_info.find_one({"_id": "singleton"}, {"_id": 0})
    return fresh or {}


@api_router.get("/")
async def root():
    return {"message": "21-ASR Raqamli Xizmatlar Markazi API"}


app.include_router(api_router)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def on_startup():
    await db.users.create_index("phone", unique=True, sparse=True)
    await db.requests.create_index("user_id")
    await db.requests.create_index("status")
    await db.requests.create_index("category")
    await db.support_messages.create_index([("user_id", 1), ("created_at", 1)])

    admin_phone = os.environ.get("ADMIN_PHONE", "+998901112233")
    admin_password = os.environ.get("ADMIN_PASSWORD", "Admin@123")
    existing = await db.users.find_one({"phone": admin_phone})
    if not existing:
        await db.users.insert_one({
            "id": str(uuid.uuid4()),
            "first_name": "System", "last_name": "Admin",
            "phone": admin_phone, "email": "",
            "password_hash": hash_password(admin_password),
            "role": "admin", "language": "uz",
            "kyc_status": "approved", "kyc_note": "auto-approved",
            "created_at": now_iso(),
        })
        logger.info(f"Seeded admin: {admin_phone}")
    else:
        updates = {"role": "admin", "kyc_status": "approved"}
        if not verify_password(admin_password, existing["password_hash"]):
            updates["password_hash"] = hash_password(admin_password)
        await db.users.update_one({"phone": admin_phone}, {"$set": updates})

    # Remove old email-based admin account if exists (cleanup)
    await db.users.delete_many({"email": "admin@swsp.uz", "phone": {"$ne": admin_phone}})

    # Seed default team if empty
    count = await db.team_members.count_documents({})
    if count == 0:
        defaults = [
            {"id": str(uuid.uuid4()), "name": "Director / Ta'sischi", "role": "Direktor va Ta'sischi", "email": "", "phone": "", "photo": "", "order": 1, "bio": "", "created_at": now_iso()},
            {"id": str(uuid.uuid4()), "name": "Bosh Buxgalter", "role": "Bosh Buxgalter", "email": "", "phone": "", "photo": "", "order": 2, "bio": "", "created_at": now_iso()},
            {"id": str(uuid.uuid4()), "name": "Buxgalter", "role": "Buxgalter", "email": "", "phone": "", "photo": "", "order": 3, "bio": "", "created_at": now_iso()},
            {"id": str(uuid.uuid4()), "name": "Moliya bo'limi boshlig'i", "role": "Moliya bo'limi boshlig'i", "email": "", "phone": "", "photo": "", "order": 4, "bio": "", "created_at": now_iso()},
        ]
        await db.team_members.insert_many(defaults)

    existing_contact = await db.contact_info.find_one({"_id": "singleton"})
    if not existing_contact:
        await db.contact_info.update_one(
            {"_id": "singleton"},
            {"$set": {
                "address_uz": "Samarqand shahri, O'zbekiston",
                "address_ru": "г. Samarqand, Узбекистан",
                "address_en": "Samarqand, Uzbekistan",
                "phone": "+998 (55) 701-21-00",
                "email": "info@21asr.uz",
                "working_hours": "Dush-Shanba 09:00-18:00",
                "telegram": "asrbux_21",
                "website": "",
            }},
            upsert=True,
        )

    # Seed default payment methods if empty
    if await db.payment_methods.count_documents({}) == 0:
        defaults_pm = [
            {"id": str(uuid.uuid4()), "name": "Payme", "kind": "link", "value": "https://payme.uz", "description": "Payme orqali", "order": 1, "active": True, "created_at": now_iso()},
            {"id": str(uuid.uuid4()), "name": "Click", "kind": "link", "value": "https://click.uz", "description": "Click orqali", "order": 2, "active": True, "created_at": now_iso()},
            {"id": str(uuid.uuid4()), "name": "Xazna", "kind": "link", "value": "https://xazna.uz", "description": "Xazna orqali", "order": 3, "active": True, "created_at": now_iso()},
            {"id": str(uuid.uuid4()), "name": "Paynet", "kind": "link", "value": "https://paynet.uz", "description": "Paynet orqali", "order": 4, "active": True, "created_at": now_iso()},
        ]
        await db.payment_methods.insert_many(defaults_pm)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
