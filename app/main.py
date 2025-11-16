# ============================================================================
# IMPORTACIONES
# ============================================================================

# Librerías estándar de Python
import datetime
import logging
import base64
import json
import os
from io import BytesIO

# Librerías de terceros - FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Librerías de terceros - Base de datos
from tinydb import TinyDB, Query

# Esquemas de la aplicación
from schemas import (
    DiaryEntry,
    GeminiResponseModel,
    GeminiBaseResponse,
    ImageInput,
    PromptInput
)

# Librerías de terceros - Procesamiento de imágenes y ML
from PIL import Image
import torch
from torchvision import models, transforms

# Librerías de terceros - IA Generativa
from google import genai
from dotenv import load_dotenv

# ============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================================

# Obtener el directorio base del proyecto (directorio padre de 'app')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cargar variables de entorno desde archivo .env
load_dotenv(os.path.join(BASE_DIR, ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Inicializar cliente de GenAI
client = genai.Client(api_key=GEMINI_API_KEY)

# Configurar logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FUNCIONES DE IA GENERATIVA
# ============================================================================

def prompt_gemini(prompt: str, model: BaseModel) -> str:
    """
    Envía un prompt a Gemini AI y retorna una respuesta estructurada.

    Args:
        prompt (str): El texto del prompt a enviar
        model (BaseModel): Modelo Pydantic para estructurar la respuesta

    Returns:
        str: Respuesta validada según el modelo especificado
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "temperature": 1.0,
            "response_mime_type": "application/json",
            "response_json_schema": model.model_json_schema()
        }
    )
    model_response = model.model_validate_json(response.text)
    # logger.info(f"Respuesta de Gemini: {response.text}")
    return model_response

# ============================================================================
# CONFIGURACIÓN DEL MODELO DE CLASIFICACIÓN DE IMÁGENES
# ============================================================================

# Obtener el directorio base del proyecto (directorio padre de 'app')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "resnet50")

# Cargar modelo ResNet-50 pre-entrenado
resnet50 = models.resnet50()
resnet50.load_state_dict(torch.load(os.path.join(MODELS_DIR, "resnet50-0676ba61.pth")))
resnet50.eval()  # Establecer en modo evaluación

# Definir pipeline de preprocesamiento de imágenes
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar a 224x224 píxeles
    transforms.ToTensor(),          # Convertir a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalización ImageNet
])

# Cargar etiquetas de clases de ImageNet
with open(os.path.join(MODELS_DIR, "imagenet_class_index.json"), "r") as f:
    idx_to_label = {int(key): value for key, value in json.load(f).items()}

# ============================================================================
# FUNCIONES DE PROCESAMIENTO DE IMÁGENES
# ============================================================================

def predict_image_label(image_base64: str) -> str:
    """
    Predice la etiqueta de una imagen codificada en base64 usando ResNet-50.

    Args:
        image_base64 (str): Imagen codificada en base64

    Returns:
        str: Etiqueta predicha de la imagen
    """
    # Decodificar imagen base64
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data)).convert("RGB")

    # Preprocesar imagen
    img_tensor = preprocess(image).unsqueeze(0)

    # Realizar predicción
    with torch.no_grad():
        outputs = resnet50(img_tensor)
        _, predicted = outputs.max(1)
        label = idx_to_label[predicted.item()]
        return label

# ============================================================================
# CONFIGURACIÓN DE FASTAPI Y BASE DE DATOS
# ============================================================================

# Crear aplicación FastAPI
app = FastAPI(media_type="application/json; charset=utf-8")

# Inicializar base de datos TinyDB (los datos se almacenan en 'db.json')
db = TinyDB(os.path.join(BASE_DIR, 'db', 'db.json'), encoding='utf-8')

# ============================================================================
# ENDPOINTS DE LA API
# ============================================================================

@app.get("/")
async def read_root():
    """
    Endpoint raíz que retorna información básica de la aplicación.

    Returns:
        dict: Información de la aplicación Mr.Zorro
    """
    return {"app": "Mr.Zorro",
            "version": "1.0.0",
            "description": "API para la app Mr.Zorro"}

@app.get("/diary")
async def get_diary_data():
    """
    Obtiene todas las entradas del diario almacenadas en la base de datos.

    Returns:
        JSONResponse: Lista completa de entradas del diario
    """
    return JSONResponse(content=db.all())

@app.get("/diary/{date}")
async def get_diary_by_date(date: str):
    """
    Obtiene las entradas del diario para una fecha específica.

    Args:
        date (str): La fecha en formato 'YYYY-MM-DD'

    Returns:
        JSONResponse: Entradas del diario para la fecha especificada o error 404
    """
    Entry = Query()
    result = db.search(Entry.date == date)
    if result:
        return JSONResponse(content=result)
    return JSONResponse(content={"error": "No se encontraron datos para la fecha especificada"}, status_code=404)

@app.post("/diary")
async def add_diary_entry(entry: DiaryEntry):
    """
    Agrega una nueva entrada al diario del usuario.

    Procesa la entrada del usuario, analiza cualquier imagen adjunta,
    genera recomendaciones personalizadas usando IA y almacena todo en la base de datos.

    Args:
        entry (DiaryEntry): Entrada del diario con estado de ánimo, nota e imagen opcional

    Returns:
        JSONResponse: Confirmación de que la entrada fue agregada exitosamente
    """

    # Procesar imagen si está presente
    if entry.img is not None and entry.img != "":
        entry.img = predict_image_label(entry.img)

    # Crear prompt para Gemini AI
    prompt = """
    Eres un acompañante emocional llamado Mr. Zorro
    que genera recomendaciones diarias positivas y motivadoras.
    Basándote en la emoción del usuario: {mood}, en su nota agregada: {note} y
    en la etiqueta de la imagen que han guardado como recuerdo del día: {img},
    crea una recomendación breve y alentadora, un dato curioso o especial para el día.
    Tu respuesta debe ser en español y no debe exceder 100 palabras."""

    # Llenar el prompt con los datos del usuario
    filled_prompt = prompt.format(mood=entry.mood, note=entry.note or "None", img=entry.img or "None")

    # Generar respuesta personalizada con IA
    overview = prompt_gemini(filled_prompt, GeminiResponseModel)

     # Obtener fecha actual
    date = datetime.date.today().isoformat()

    # Crear estructura de datos para almacenar
    entry_data = {
        "date": date,
        "overview": overview.model_dump(),
        "mood": entry.mood,
        "note": entry.note,
        "img": entry.img
    }
    # Verificar si ya existe una entrada para la fecha actual, actualizar si es así
    if db.search(Query().date == date):
        db.update(entry_data, Query().date == date)
        return JSONResponse(content={"message": "Entrada actualizada exitosamente"})

    # Insertar en base de datos
    db.insert(entry_data)
    return JSONResponse(content={"message": "Entrada agregada exitosamente"})

@app.post("/predict-image")
async def predict_image(input: ImageInput):
    """
    Predice la etiqueta de una imagen y actualiza la entrada del diario correspondiente.

    Args:
        input (ImageInput): Objeto con fecha e imagen en base64

    Returns:
        JSONResponse: Fecha y etiqueta predicha de la imagen

    Raises:
        HTTPException: Error 400 si la imagen/fecha es inválida o hay error en predicción
    """
    try:
        # Buscar entrada existente para la fecha
        data = db.search(Query().date == input.date)[0]
        data = json.loads(json.dumps(data))  # Convertir TinyDB Result a dict estándar
        if not data:
            raise HTTPException(status_code=404, detail="No se encontró entrada para la fecha especificada")

        # Predecir etiqueta de la imagen
        label = predict_image_label(input.img)

        #Actualizar overview con nuevo contenido
        # Crear prompt para Gemini AI
        prompt = """
        Eres un acompañante emocional llamado Mr. Zorro
        que genera recomendaciones diarias positivas y motivadoras.
        Basándote en la emoción del usuario: {mood}, en su nota agregada: {note} y
        en la etiqueta de la imagen que han guardado como recuerdo del día: {img},
        crea una recomendación breve y alentadora, un dato curioso o especial para el día.
        Tu respuesta debe ser en español y no debe exceder 100 palabras."""

        # Llenar el prompt con los datos del usuario
        filled_prompt = prompt.format(mood=data['mood'], note=data['note'] or "None", img=label or "None")

        # Generar respuesta personalizada con IA
        overview = prompt_gemini(filled_prompt, GeminiResponseModel)

        # Crear estructura de datos para almacenar
        updated_data = {
            "date": input.date,
            "overview": overview.model_dump(),
            "mood": data['mood'],
            "note": data['note'],
            "img": label
        }

        db.update(updated_data, Query().date == input.date)
        return JSONResponse(content={"date": input.date, "predicted_label": label})
    except Exception as e:
        logger.error(f"Error de predicción: {e}")
        raise HTTPException(status_code=400, detail="Imagen/fecha inválida o error en predicción")

@app.post("/prompt")
async def generate_prompt_response(prompt: PromptInput):
    """
    Genera una respuesta estructurada usando Gemini AI basada en el prompt proporcionado.

    Args:
        prompt (str): El texto del prompt a enviar

    Returns:
        JSONResponse: Respuesta generada por Gemini AI

    Raises:
        HTTPException: Error 400 si hay un error en la generación de la respuesta
    """
    try:
        # Obtener fecha de inicio de la semana (lunes)
        today = datetime.date.today()
        start_of_week = today - datetime.timedelta(days=today.weekday())

        # Filtrar entradas de la semana actual y extraer solo mood, note e img
        diary_entries = [
            {
                "mood": entry.get("mood"),
                "note": entry.get("note"),
                "img": entry.get("img")
            }
            for entry in db.all()
            if entry.get("date") and datetime.date.fromisoformat(entry.get("date")) >= start_of_week
        ]

        structured_prompt = """
        Eres un acompañante emocional llamado Mr. Zorro
        que genera respuestas motivadoras y positivas.
        Basándote en el siguiente prompt del usuario y las entradas a su diario,
        genera una respuesta breve y alentadora. Las entradas del diario están en formato JSON,
        corresponden a la semana acutal
        y contienen el estado de ánimo, nota e imagen etiquetada del usuario.
        Tu respuesta debe ser en español y no debe exceder 100 palabras.
        Usuario: {user_prompt}
        Entradas del diario: {diary_entries}
        """.format(user_prompt=prompt, diary_entries=json.dumps(diary_entries, ensure_ascii=False))
        response = prompt_gemini(structured_prompt, GeminiBaseResponse)
        return JSONResponse(content=response.model_dump())
    except Exception as e:
        logger.error(f"Error de predicción: {e}")
        raise HTTPException(status_code=400, detail="Error en generación de respuesta")
# ============================================================================
# PUNTO DE ENTRADA DE LA APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
