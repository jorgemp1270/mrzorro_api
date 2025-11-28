"""
MongoDB Document Models for Mr.Zorro API
Using Beanie ODM for async MongoDB operations
"""

from datetime import datetime
from typing import Optional
from beanie import Document
from pydantic import Field, BaseModel
from bson import ObjectId

class Therapist(Document):
    """MongoDB document model for therapists"""
    therapist_id: str = Field(..., unique=True)
    name: str
    email: str = Field(..., unique=True)
    address: Optional[str] = None
    phone: Optional[str] = None
    specialization: Optional[str] = None
    profile_img: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "therapists"


class UserSettings(BaseModel):
    age: str = "kids"
    personality: str = "default"
    considerations: Optional[str] = ""
    about_me: Optional[str] = ""

class Contact(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None


class User(Document):
    """MongoDB document model for users"""
    user_id: str = Field(..., unique=True)
    email: str = Field(..., unique=True)
    password: str
    nickname: str
    last_login: Optional[datetime] = None
    streak: int = 1
    best_streak: int = 1
    points: int = 0
    contacts: Optional[list[Contact]] = []
    danger_level: int = 0
    themes: Optional[list[str]] = []
    fonts: Optional[list[str]] = []
    settings: Optional[UserSettings] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "users"


class DiaryEntryDoc(Document):
    """MongoDB document model for diary entries"""
    user_id: str
    date: str
    overview: dict
    mood: str
    title: Optional[str] = None
    note: Optional[str] = None
    img: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "diary_entries"


class ChatMessage(BaseModel):
    role: str  # "user" or "model"
    content: str


class ChatSession(Document):
    user_id: str
    history: list[ChatMessage] = []
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "chat_sessions"