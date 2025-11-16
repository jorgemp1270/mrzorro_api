# API Mr. Zorro

Backend desarrollado en Python con FastAPI para la aplicación móvil Mr. Zorro. Esta API funciona como un acompañante emocional que procesa entradas de diario, analiza imágenes usando IA y genera recomendaciones personalizadas.

## 🚀 Características

- **Procesamiento de imágenes**: Clasificación automática usando ResNet-50 pre-entrenado
- **IA Generativa**: Integración con Google Gemini AI para recomendaciones personalizadas
- **Base de datos**: Almacenamiento de entradas de diario con TinyDB
- **API RESTful**: Endpoints completos para gestión de diario

## 📋 Requisitos

- Python 3.8+
- PyTorch
- FastAPI
- Google Gemini API Key

## 🛠️ Instalación

1. Clona el repositorio:
```bash
git clone <repository-url>
cd backend
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Configura las variables de entorno:
Crea un archivo `.env` en el directorio raíz con:
```
GEMINI_API_KEY=tu_api_key_aqui
```

4. Ejecuta la aplicación:
```bash
fastapi dev app/main.py
```

## 📁 Estructura del Proyecto

```
backend/
├── app/
│   └── main.py          # Aplicación principal FastAPI
├── db/
│   └── db.json          # Base de datos TinyDB
├── models/
│   └── resnet50/        # Modelo ResNet-50 y archivos relacionados
├── requirements.txt     # Dependencias Python
└── README.md           # Este archivo
```

## 🔧 Endpoints de la API

### 1. Información de la API
- **URL**: `/`
- **Método**: `GET`
- **Descripción**: Retorna información básica de la aplicación
- **Respuesta**:
```json
{
    "app": "Mr.Zorro",
    "version": "1.0.0",
    "description": "API para la app Mr.Zorro"
}
```

### 2. Obtener todas las entradas del diario
- **URL**: `/diary`
- **Método**: `GET`
- **Descripción**: Obtiene todas las entradas del diario almacenadas
- **Respuesta**: Array de entradas del diario
```json
[
    {
        "date": "2025-11-15",
        "overview": {
            "message": "Mensaje motivador",
            "recommendation": "Recomendación personalizada",
            "interesting_fact": "Dato curioso del día"
        },
        "mood": "feliz",
        "note": "Nota del usuario",
        "img": "etiqueta_imagen"
    }
]
```

### 3. Obtener entrada por fecha
- **URL**: `/diary/{date}`
- **Método**: `GET`
- **Descripción**: Obtiene las entradas del diario para una fecha específica
- **Parámetros**:
  - `date` (string): Fecha en formato YYYY-MM-DD
- **Respuesta exitosa**: Array de entradas para la fecha especificada
- **Respuesta error (404)**:
```json
{
    "error": "No se encontraron datos para la fecha especificada"
}
```

### 4. Agregar nueva entrada al diario
- **URL**: `/diary`
- **Método**: `POST`
- **Descripción**: Agrega una nueva entrada al diario con procesamiento de IA
- **Cuerpo de la petición**:
```json
{
    "mood": "feliz",
    "note": "Mi nota del día (opcional)",
    "img": "imagen_en_base64 (opcional)"
}
```
- **Campos**:
  - `mood` (string, requerido): Estado de ánimo del usuario
  - `note` (string, opcional): Nota personal del usuario
  - `img` (string, opcional): Imagen codificada en base64
- **Respuesta exitosa**:
```json
{
    "message": "Entrada agregada exitosamente"
}
```
- **Si ya existe entrada para la fecha**:
```json
{
    "message": "Entrada actualizada exitosamente"
}
```

### 5. Predecir etiqueta de imagen
- **URL**: `/predict-image`
- **Método**: `POST`
- **Descripción**: Analiza una imagen y actualiza la entrada del diario correspondiente
- **Cuerpo de la petición**:
```json
{
    "date": "2025-11-15",
    "img": "imagen_en_base64"
}
```
- **Campos**:
  - `date` (string, requerido): Fecha en formato YYYY-MM-DD
  - `img` (string, requerido): Imagen codificada en base64
- **Respuesta exitosa**:
```json
{
    "date": "2025-11-15",
    "predicted_label": "etiqueta_predicha"
}
```
- **Respuesta error (400)**:
```json
{
    "detail": "Imagen/fecha inválida o error en predicción"
}
```
- **Respuesta error (404)**:
```json
{
    "detail": "No se encontró entrada para la fecha especificada"
}
```

## 🤖 Integración con IA

### Google Gemini AI
La API utiliza Google Gemini AI para generar:
- Mensajes motivadores personalizados
- Recomendaciones basadas en el estado de ánimo
- Datos curiosos relacionados con el día del usuario

### ResNet-50 para Clasificación de Imágenes
- Modelo pre-entrenado en ImageNet
- Clasifica imágenes en 1000 categorías diferentes
- Procesa imágenes automáticamente cuando se suben al diario

## 📊 Base de Datos

La aplicación utiliza TinyDB, una base de datos JSON ligera que almacena:
- Entradas diarias del usuario
- Estados de ánimo y notas
- Etiquetas de imágenes procesadas
- Respuestas generadas por IA

## 🔐 Configuración de Seguridad

Asegúrate de:
- Mantener tu `GEMINI_API_KEY` segura en el archivo `.env`
- No subir el archivo `.env` al control de versiones
- Configurar CORS apropiadamente para producción

## 🚀 Despliegue

Para desplegar en producción:

1. Configura las variables de entorno en tu servidor
2. Usa un servidor WSGI como Gunicorn:
```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📝 Notas Adicionales

- La API procesa imágenes en formato base64
- Las fechas deben estar en formato ISO (YYYY-MM-DD)
- Las respuestas de IA están limitadas a 100 palabras
- La base de datos se crea automáticamente en la primera ejecución