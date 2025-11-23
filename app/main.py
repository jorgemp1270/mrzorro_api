# ============================================================================
# IMPORTACIONES
# ============================================================================

# Librerías estándar de Python
import datetime
import logging
import base64
import json
import os
import wave
import shutil
import subprocess
from io import BytesIO

# Librerías de terceros - FastAPI
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Librerías de terceros - Base de datos
from tinydb import TinyDB, Query
from beanie import PydanticObjectId
from bson import ObjectId

# Esquemas de la aplicación
from .schemas import (
    DiaryEntry,
    GeminiResponseModel,
    GeminiBaseResponse,
    ImageInput,
    ImagePrediction,
    PurchaseInput,
    LoginInput,
    PromptInput,
    SignupInput
)

# MongoDB models and database
from .database import init_database
from .models import User, DiaryEntryDoc

# Librerías de terceros - Procesamiento de imágenes y ML
from PIL import Image
import torch
from torchvision import models, transforms
import whisper
from gtts import gTTS

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

async def validate_user_exists(user_id: str) -> bool:
    """
    Valida si un usuario existe en la base de datos.

    Args:
        user_id (str): ID del usuario a validar

    Returns:
        bool: True si el usuario existe, False en caso contrario
    """
    user = await User.find_one(User.user_id == user_id)
    return user is not None

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
    try:
        # Eliminar encabezado de data URI si existe (ej: data:image/jpeg;base64,...)
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

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
            # Si la etiqueta es una lista (ej: ['tench', 'Tinca_tinca']), tomar el primer elemento
            if isinstance(label, list):
                label = label[1] # Usar el nombre común si está disponible
            return str(label)
    except Exception as e:
        logger.error(f"Error en predicción de imagen: {e}")
        return "imagen no identificada"

# ============================================================================
# CONFIGURACIÓN DE AUDIO
# ============================================================================
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
whisper_model = None
active_recordings = {}

# ============================================================================
# CONFIGURACIÓN DE FASTAPI Y BASE DE DATOS
# ============================================================================

# Crear aplicación FastAPI
app = FastAPI(media_type="application/json; charset=utf-8")

# MongoDB database initialization on startup
@app.on_event("startup")
async def startup_event():
    """Initialize MongoDB connection and load ML models on application startup"""
    await init_database()
    logger.info("MongoDB connection initialized successfully")

    global whisper_model
    try:
        logger.info("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Whisper model: {e}")

# ============================================================================
# FUNCIONES DE PROCESAMIENTO DE AUDIO
# ============================================================================

def convert_mp3_to_wav(mp3_path, wav_path):
    command = [
        "ffmpeg",
        "-y",
        "-i", mp3_path,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        wav_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

async def process_audio_with_ai(user_id: str):
    try:
        if user_id not in active_recordings or not active_recordings[user_id]['chunks']:
            return JSONResponse(status_code=400, content={'status': 'error', 'message': 'No audio data'})

        recordings_dir = os.path.join(BASE_DIR, 'recordings')
        os.makedirs(recordings_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        input_filename = os.path.join(recordings_dir, f'input_{user_id}_{timestamp}.wav')

        # Combine chunks
        audio_data = b''.join(active_recordings[user_id]['chunks'])

        # Save WAV
        with wave.open(input_filename, 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_data)

        # Transcribe
        if whisper_model is None:
             return JSONResponse(status_code=500, content={'status': 'error', 'message': 'Whisper model not loaded'})

        result = whisper_model.transcribe(input_filename, language='es')
        user_text = result.get("text", "").strip()
        if not user_text:
            user_text = "No pude entender."

        # Get Diary Context
        today = datetime.date.today()
        week_ago = today - datetime.timedelta(days=7)
        # Assuming date is stored as YYYY-MM-DD string
        entries = await DiaryEntryDoc.find(DiaryEntryDoc.user_id == user_id).to_list()

        context_text = ""
        for entry in entries:
            try:
                # Handle potential date format variations if necessary, assuming YYYY-MM-DD
                entry_date = datetime.datetime.strptime(entry.date, "%Y-%m-%d").date()
                if week_ago <= entry_date <= today:
                    context_text += f"Date: {entry.date}, Title: {entry.title or 'No Title'}, Note: {entry.note or 'No Note'}, Mood: {entry.mood}\n"
            except ValueError:
                continue

        # Call Gemini
        prompt = f"""Contexto:
          Entradas del diario del usuario de la última semana:\n{context_text}\n\n
          El usuario dice: {user_text}\n\n
          Responde en español de forma breve y comprensible como un acompañante emocional,
          tu nombre es Mr. Zorro."""

        try:
            gemini_response = prompt_gemini(prompt, GeminiBaseResponse)
            ai_response = gemini_response.response
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            ai_response = "Lo siento, tuve un problema al pensar mi respuesta."

        # TTS
        mp3_name = f"output_{user_id}_{timestamp}.mp3"
        mp3_path = os.path.join(recordings_dir, mp3_name)
        wav_name = f"output_{user_id}_{timestamp}.wav"
        wav_path = os.path.join(recordings_dir, wav_name)

        tts = gTTS(text=ai_response, lang='es', slow=False)
        tts.save(mp3_path)

        # Convert to WAV for ESP32
        convert_mp3_to_wav(mp3_path, wav_path)

        # Clear recording state
        del active_recordings[user_id]

        return JSONResponse(content={
            'status': 'ok',
            'user_text': user_text,
            'ai_response': ai_response,
            'filename': wav_name
        })

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})

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

@app.get("/diary/{user}")
async def get_diary_data(user: str):
    """
    Obtiene todas las entradas del diario para un usuario específico.

    Args:
        user (str): ID único del usuario

    Returns:
        JSONResponse: Lista de entradas del diario del usuario
    """
    # Validar que el usuario existe
    if not await validate_user_exists(user):
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    entries = await DiaryEntryDoc.find(DiaryEntryDoc.user_id == user).to_list()
    # Convert documents to dict format for JSON response
    result = [entry.dict(exclude={"id", "created_at"}) for entry in entries]
    return JSONResponse(content=result)

@app.get("/diary/{user}/{date}")
async def get_diary_by_date(user: str, date: str):
    """
    Obtiene las entradas del diario para un usuario y fecha específica.

    Args:
        user (str): ID único del usuario
        date (str): La fecha en formato 'YYYY-MM-DD'

    Returns:
        JSONResponse: Entradas del diario para la fecha especificada o error 404
    """
    # Validar que el usuario existe
    if not await validate_user_exists(user):
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    entries = await DiaryEntryDoc.find(DiaryEntryDoc.user_id == user, DiaryEntryDoc.date == date).to_list()
    if entries:
        result = [entry.dict(exclude={"id"}) for entry in entries]
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

    # Validar que el usuario existe
    if not await validate_user_exists(entry.user):
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    # Procesar imagen si está presente
    predicted_label = ""
    if entry.img and entry.img.strip():
        logger.info(f"Analizando imagen para usuario {entry.user}...")
        predicted_label = predict_image_label(entry.img)
        logger.info(f"Imagen analizada: {predicted_label}")

    # Crear prompt para Gemini AI
    prompt = """
    Eres un acompañante emocional llamado Mr. Zorro
    que genera recomendaciones diarias positivas y motivadoras.
    Basándote en la emoción del usuario: {mood}, en su nota agregada: {note} y
    en la etiqueta de la imagen que han guardado como recuerdo del día: {img},
    crea una recomendación breve y alentadora, un dato curioso o especial para el día.
    Tu respuesta debe ser en español y no debe exceder 100 palabras."""

    # Llenar el prompt con los datos del usuario
    filled_prompt = prompt.format(mood=entry.mood, note=entry.note or "None", img=predicted_label or "None")

    # Generar respuesta personalizada con IA
    overview = prompt_gemini(filled_prompt, GeminiResponseModel)

     # Obtener fecha actual
    date = datetime.date.today().isoformat()

    # Crear estructura de datos para almacenar
    entry_data = {
        "user_id": entry.user,
        "date": date,
        "overview": overview.model_dump(),
        "mood": entry.mood,
        "title": entry.title,
        "note": entry.note,
        "img": predicted_label
    }

    # Verificar si ya existe una entrada para la fecha actual y usuario, actualizar si es así
    existing_entry = await DiaryEntryDoc.find_one(DiaryEntryDoc.user_id == entry.user, DiaryEntryDoc.date == date)
    if existing_entry:
        await existing_entry.update({"$set": entry_data})
        return JSONResponse(content={"message": "Entrada actualizada exitosamente"})

    # Crear nueva entrada
    new_entry = DiaryEntryDoc(**entry_data)
    await new_entry.insert()
    logger.info(f"✓ Created diary entry for user {entry.user} on {date}")

    # Otorgar 5 puntos al usuario por agregar una entrada
    user = await User.find_one(User.user_id == entry.user)
    new_points = 5  # Default if user not found
    if user:
        current_points = user.points
        new_points = current_points + 5
        await user.update({"$set": {"points": new_points}})
        logger.info(f"✓ Updated user {entry.user} points: {current_points} → {new_points}")

    return JSONResponse(content={
        "message": "Entrada agregada exitosamente",
        "points_earned": 5,
        "total_points": new_points
    })

@app.post("/update-image")
async def update_image_prediction(input: ImageInput):
    """
    Predice la etiqueta de una imagen y actualiza la entrada del diario correspondiente.

    Args:
        input (ImageInput): Objeto con fecha e imagen en base64

    Returns:
        JSONResponse: Fecha y etiqueta predicha de la imagen

    Raises:
        HTTPException: Error 400 si la imagen/fecha es inválida o hay error en predicción
    """
    # Validar que el usuario existe
    if not await validate_user_exists(input.user):
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    try:
        # Buscar entrada existente para el usuario y fecha
        existing_entry = await DiaryEntryDoc.find_one(DiaryEntryDoc.user_id == input.user, DiaryEntryDoc.date == input.date)
        if not existing_entry:
            raise HTTPException(status_code=404, detail="No se encontró entrada para la fecha especificada")

        # Predecir etiqueta de la imagen
        label = predict_image_label(input.img)

        # Crear prompt para Gemini AI
        prompt = """
        Eres un acompañante emocional llamado Mr. Zorro
        que genera recomendaciones diarias positivas y motivadoras.
        Basándote en la emoción del usuario: {mood}, en su nota agregada: {note} y
        en la etiqueta de la imagen que han guardado como recuerdo del día: {img},
        crea una recomendación breve y alentadora, un dato curioso o especial para el día.
        Tu respuesta debe ser en español y no debe exceder 100 palabras."""

        # Llenar el prompt con los datos del usuario
        filled_prompt = prompt.format(mood=existing_entry.mood, note=existing_entry.note or "None", img=label or "None")

        # Generar respuesta personalizada con IA
        overview = prompt_gemini(filled_prompt, GeminiResponseModel)

        # Actualizar la entrada con nueva imagen y overview
        await existing_entry.update({
            "$set": {
                "overview": overview.model_dump(),
                "img": label
            }
        })

        return JSONResponse(content={"date": input.date, "predicted_label": label})
    except Exception as e:
        logger.error(f"Error de predicción: {e}")
        raise HTTPException(status_code=400, detail="Imagen/fecha inválida o error en predicción")

@app.post("/predict-image")
async def predict_image_endpoint(input: ImagePrediction):
    """
    Predice la etiqueta de una imagen codificada en base64.

    Args:
        input (ImagePrediction): Objeto con usuario e imagen en base64

    Returns:
        JSONResponse: Etiqueta predicha de la imagen

    Raises:
        HTTPException: Error 400 si hay un error en la predicción
    """
    # Validar que el usuario existe
    if not await validate_user_exists(input.user):
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    try:
        label = predict_image_label(input.img)
        # Obtener fecha de inicio de la semana (lunes)
        today = datetime.date.today()
        start_of_week = today - datetime.timedelta(days=today.weekday())

        # Filtrar entradas de la semana actual para el usuario específico
        user_entries = await DiaryEntryDoc.find(DiaryEntryDoc.user_id == input.user).to_list()
        diary_entries = [
            {
                "mood": entry.mood,
                "note": entry.note,
                "img": entry.img
            }
            for entry in user_entries
            if entry.date and datetime.date.fromisoformat(entry.date) >= start_of_week
        ]

        # Generar descripción con Gemini AI
        prompt = """
        Eres un acompañante emocional llamado Mr. Zorro.
        Estas ayudando a un usuario a describir la imagen que ha subido a su diario.
        Basándote en la etiqueta reconocida por ResNet-50: {img},
        crea una breve descripción positiva y motivadora relacionada con esa imagen.
        Basate también en las entradas previas de la semana del usuario a su diario:
        {diary_entries}
        Tu respuesta debe ser en español y no debe exceder 50 palabras.
        """.format(img=label or "None", diary_entries=json.dumps(diary_entries, ensure_ascii=False))

        # Generar respuesta personalizada con IA
        response = prompt_gemini(prompt, GeminiResponseModel)

        # Crear estructura de datos para almacenar
        data = {
            "overview": response.model_dump(),
            "img": label
        }

        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Error de predicción: {e}")
        raise HTTPException(status_code=400, detail="Error en predicción de imagen")

@app.post("/prompt")
async def generate_prompt_response(prompt: PromptInput):
    """
    Genera una respuesta estructurada usando Gemini AI basada en el prompt proporcionado.

    Args:
        prompt (PromptInput): El texto del prompt a enviar

    Returns:
        JSONResponse: Respuesta generada por Gemini AI

    Raises:
        HTTPException: Error 400 si hay un error en la generación de la respuesta
    """
    # Validar que el usuario existe
    if not await validate_user_exists(prompt.user):
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    try:
        # Obtener fecha de inicio de la semana (lunes)
        today = datetime.date.today()
        start_of_week = today - datetime.timedelta(days=today.weekday())

        # Filtrar entradas de la semana actual para el usuario específico
        user_entries = await DiaryEntryDoc.find(DiaryEntryDoc.user_id == prompt.user).to_list()
        diary_entries = [
            {
                "mood": entry.mood,
                "note": entry.note,
                "img": entry.img
            }
            for entry in user_entries
            if entry.date and datetime.date.fromisoformat(entry.date) >= start_of_week
        ]

        structured_prompt = """
        Eres un acompañante emocional llamado Mr. Zorro
        que genera respuestas motivadoras y positivas.
        Basándote en el siguiente prompt del usuario y las entradas a su diario,
        genera una respuesta breve y alentadora. Las entradas del diario están en formato JSON,
        corresponden a la semana actual
        y contienen el estado de ánimo, nota e imagen etiquetada del usuario.
        Tu respuesta debe ser en español y no debe exceder 100 palabras.
        Usuario: {user_prompt}
        Entradas del diario: {diary_entries}
        """.format(user_prompt=prompt.prompt, diary_entries=json.dumps(diary_entries, ensure_ascii=False))
        response = prompt_gemini(structured_prompt, GeminiBaseResponse)
        return JSONResponse(content=response.model_dump())
    except Exception as e:
        logger.error(f"Error de predicción: {e}")
        raise HTTPException(status_code=400, detail="Error en generación de respuesta")

@app.post("/make-purchase")
async def make_purchase(input: PurchaseInput):
    """
    Actualiza los puntos del usuario tras una compra.

    Args:
        input (PurchaseInput): Objeto con ID del usuario, precio en puntos
        y el tema o fuente comprada

    Returns:
        JSONResponse: Puntos actualizados del usuario
    """
    # Validar que el usuario existe
    if not await validate_user_exists(input.user):
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    try:
        # Convert price from string to int
        try:
            price = int(float(input.price))  # Handle both int and float strings
        except ValueError:
            raise HTTPException(status_code=400, detail="Precio inválido")

        user = await User.find_one(User.user_id == input.user)
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

        current_points = user.points
        new_points = current_points - price

        if new_points < 0:
            raise HTTPException(status_code=400, detail="Puntos insuficientes para la compra")

        # Must specify exactly one: theme OR font
        if not input.theme and not input.font:
            raise HTTPException(status_code=400, detail="Debe especificar un tema o una fuente para la compra")

        if input.theme and input.font:
            raise HTTPException(status_code=400, detail="Solo puede comprar un tema o una fuente por transacción")

        # Initialize themes and fonts if None
        user_themes = user.themes or []
        user_fonts = user.fonts or []

        # Process theme purchase
        if input.theme:
            if input.theme in user_themes:
                raise HTTPException(status_code=400, detail="El tema ya ha sido comprado previamente")
            await user.update({"$addToSet": {"themes": input.theme}, "$set": {"points": new_points}})
            logger.info(f"✓ User {input.user} purchased theme '{input.theme}' for {price} points")

        # Process font purchase
        elif input.font:
            if input.font in user_fonts:
                raise HTTPException(status_code=400, detail="La fuente ya ha sido comprada previamente")
            await user.update({"$addToSet": {"fonts": input.font}, "$set": {"points": new_points}})
            logger.info(f"✓ User {input.user} purchased font '{input.font}' for {price} points")

        return JSONResponse(content={
            "message": "Compra realizada exitosamente",
            "new_points": new_points
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error de compra: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/purchases/{user}")
async def get_user_purchases(user: str):
    """
    Obtiene los puntos actuales de un usuario.

    Args:
        user (str): ID único del usuario

    Returns:
        JSONResponse: Puntos actuales del usuario
    """
    # Validar que el usuario existe
    if not await validate_user_exists(user):
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    user_obj = await User.find_one(User.user_id == user)
    if user_obj:
        return JSONResponse(content={"points": user_obj.points,
                                     "themes": list(user_obj.themes) if user_obj.themes else [],
                                     "fonts": list(user_obj.fonts) if user_obj.fonts else []})
    else:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

@app.post("/signup")
async def signup_user(SignupInput: SignupInput):
    """
    Registra un nuevo usuario en la base de datos.

    Args:
        SignupInput: Datos del nuevo usuario (email, password, nickname)

    Returns:
        JSONResponse: Confirmación de registro exitoso o error si el usuario ya existe
    """
    try:
        # Verificar si el usuario ya existe
        existing_user = await User.find_one(User.email == SignupInput.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="El usuario ya existe")

        # Crear nuevo usuario
        new_user = User(
            user_id=f"user_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(SignupInput.email) % 10000}",
            email=SignupInput.email,
            password=SignupInput.password,
            nickname=SignupInput.nickname,
            last_login=None,
            streak=1,
            best_streak=1,
            points=0
        )
        await new_user.insert()
        logger.info(f"✓ Created new user: {new_user.user_id} ({SignupInput.email})")
        return JSONResponse(content={
            "message": "Usuario creado exitosamente",
            "user": new_user.user_id
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error de registro: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.post("/login")
async def login_user(LoginInput: LoginInput):
    """
    Verifica las credenciales del usuario para iniciar sesión y actualiza el streak.
    Args:
        LoginInput: Credenciales del usuario (email y password)
    """
    try:
        # Buscar usuario por email y password
        user = await User.find_one(User.email == LoginInput.email, User.password == LoginInput.password)
        if user:
            current_time = datetime.datetime.now()

            # Obtener la última fecha de login (si existe)
            last_login = user.last_login
            current_streak = user.streak
            points = user.points
            best_streak = user.best_streak

            # Calcular nuevo streak
            if last_login:
                time_diff = current_time - last_login

                # Verificar si es un día diferente
                last_login_date = last_login.date()
                current_date = current_time.date()
                is_different_day = last_login_date != current_date

                # Incrementar streak solo si es día diferente Y menos de 24 horas
                if is_different_day and time_diff < datetime.timedelta(hours=24):
                    current_streak += 1
                    points += 1  # Otorgar 1 punto por login diario
                elif time_diff >= datetime.timedelta(hours=24):
                    # Si pasaron más de 24 horas, resetear streak
                    current_streak = 1
                    points += 1  # Otorgar 1 punto por login diario
                # Si es el mismo día, mantener el streak actual (no incrementar)
            else:
                # Primera vez que se loguea, streak = 1
                current_streak = 1
                points += 1  # Otorgar 1 punto por login diario

            # Actualizar best_streak si es necesario
            if current_streak > best_streak:
                best_streak = current_streak

            # Actualizar datos del usuario
            await user.update({
                "$set": {
                    "last_login": current_time,
                    "streak": current_streak,
                    "best_streak": best_streak,
                    "points": points
                }
            })

            result = {
                "user": user.user_id,
                "nickname": user.nickname,
                "streak": current_streak,
                "best_streak": best_streak,
                "points": points,
                "themes": list(user.themes) if user.themes else [],
                "fonts": list(user.fonts) if user.fonts else []
            }
            return JSONResponse(content={
                "message": "Inicio de sesión exitoso",
                "user": result
            })
        else:
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error de inicio de sesión: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

# ============================================================================
# ENDPOINTS DE ASISTENTE DE VOZ (EDWIN)
# ============================================================================

@app.post("/audio")
async def receive_audio(
    request: Request,
    x_chunk_number: int = Header(alias="X-Chunk-Number", default=0),
    x_last_chunk: str = Header(alias="X-Last-Chunk", default="false"),
    x_user_id: str = Header(alias="X-User-Id", default=None)
):
    """
    Recibe chunks de audio desde el ESP32, los procesa con Whisper y Gemini,
    y retorna una respuesta de audio sintetizada.
    """
    if not x_user_id:
        return JSONResponse(status_code=400, content={'status': 'error', 'message': 'X-User-Id header is required'})

    try:
        audio_data = await request.body()
        is_last = x_last_chunk.lower() == 'true'

        if x_chunk_number == 1:
            active_recordings[x_user_id] = {
                'chunks': [],
                'total_bytes': 0,
                'start_time': datetime.datetime.now()
            }
            logger.info(f"New recording started for user {x_user_id}")

        if x_user_id not in active_recordings:
             # Handle case where chunk 1 was missed or server restarted
             if not is_last:
                 active_recordings[x_user_id] = {
                    'chunks': [],
                    'total_bytes': 0,
                    'start_time': datetime.datetime.now()
                }
             else:
                 return JSONResponse(status_code=400, content={'status': 'error', 'message': 'No active recording found'})

        if not is_last and len(audio_data) > 0:
            active_recordings[x_user_id]['chunks'].append(audio_data)
            active_recordings[x_user_id]['total_bytes'] += len(audio_data)
            logger.info(f"User {x_user_id} Chunk {x_chunk_number}: {len(audio_data)} bytes")

        if is_last:
            logger.info(f"User {x_user_id} Last chunk received. Processing...")
            return await process_audio_with_ai(x_user_id)

        return JSONResponse(content={'status': 'ok', 'chunk': x_chunk_number})

    except Exception as e:
        logger.error(f"Error in /audio: {e}")
        return JSONResponse(status_code=500, content={'status': 'error', 'message': str(e)})

@app.get("/get_response/{filename}")
async def get_response(filename: str):
    """
    Sirve el archivo de audio generado (WAV) para que el ESP32 lo reproduzca.
    """
    recordings_dir = os.path.join(BASE_DIR, 'recordings')
    filepath = os.path.join(recordings_dir, filename)

    if os.path.exists(filepath):
        return FileResponse(filepath, media_type='audio/wav')

    raise HTTPException(status_code=404, detail="File not found")

@app.get("/last_response")
async def last_response():
    """
    Retorna el nombre del último archivo de respuesta generado.
    """
    recordings_dir = os.path.join(BASE_DIR, 'recordings')
    if not os.path.exists(recordings_dir):
         raise HTTPException(status_code=404, detail="No recordings found")

    files = [f for f in os.listdir(recordings_dir) if f.startswith('output_') and f.endswith('.wav')]

    if not files:
        raise HTTPException(status_code=404, detail="No responses found")

    files.sort(key=lambda x: os.path.getmtime(os.path.join(recordings_dir, x)), reverse=True)
    latest = files[0]

    return JSONResponse(content={'status': 'ok', 'filename': latest})

# ============================================================================
# PUNTO DE ENTRADA DE LA APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
