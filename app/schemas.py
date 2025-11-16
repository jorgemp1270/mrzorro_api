"""
Esquemas Pydantic para validación de datos de la API Mr.Zorro
"""

from pydantic import BaseModel, Field


class DiaryEntry(BaseModel):
    """Modelo para entrada de diario del usuario"""
    mood: str  # Estado de ánimo del usuario
    note: str | None  # Nota opcional del usuario
    img: str | None  # Imagen en base64 (opcional)


class GeminiResponseModel(BaseModel):
    """Modelo para respuesta estructurada de Gemini AI"""
    message: str = Field(description="Mensaje motivador general")
    recommendation: str = Field(description="Recomendación para el usuario")
    interesting_fact: str = Field(description="Dato curioso para el usuario relacionado con su día")


class GeminiBaseResponse(BaseModel):
    """Modelo base para respuesta de Gemini AI"""
    response: str = Field(description="Mensaje motivador general")


class ImageInput(BaseModel):
    """Modelo para entrada de predicción de imagen"""
    date: str  # Fecha en formato YYYY-MM-DD
    img: str   # Imagen codificada en base64


class PromptInput(BaseModel):
    """Modelo para entrada de prompt"""
    prompt: str  # Texto del prompt a enviar
