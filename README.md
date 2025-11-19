# API Mr. Zorro

Backend desarrollado en Python con FastAPI para la aplicación móvil Mr. Zorro. Esta API funciona como un acompañante emocional que procesa entradas de diario, analiza imágenes usando IA y genera recomendaciones personalizadas con sistema de autenticación y streak de usuarios.

## 🚀 Características

- **Sistema de Usuarios**: Registro, login y gestión de streak diario
- **Procesamiento de imágenes**: Clasificación automática usando ResNet-50 pre-entrenado
- **IA Generativa**: Integración con Google Gemini AI para recomendaciones personalizadas
- **Base de datos multi-usuario**: Almacenamiento separado por usuario con TinyDB
- **API RESTful**: Endpoints completos para gestión de diario con autenticación
- **Sistema de Streak**: Seguimiento de días consecutivos de login

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
│   ├── main.py          # Aplicación principal FastAPI
│   └── schemas.py       # Modelos Pydantic para validación
├── db/
│   ├── db.json          # Base de datos de entradas de diario
│   └── users.json       # Base de datos de usuarios
├── models/
│   └── resnet50/        # Modelo ResNet-50 y archivos relacionados
│       ├── resnet50-0676ba61.pth
│       └── imagenet_class_index.json
├── .env                 # Variables de entorno (no versionado)
├── requirements.txt     # Dependencias Python
└── README.md           # Este archivo
```

## 🔧 Endpoints de la API

### 🔐 Autenticación de Usuarios

### 1. Registrar nuevo usuario
- **URL**: `/signup`
- **Método**: `POST`
- **Descripción**: Registra un nuevo usuario en el sistema
- **Cuerpo de la petición**:
```json
{
    "email": "usuario@email.com",
    "password": "contraseña123",
    "nickname": "MiApodo"
}
```
- **Respuesta exitosa**:
```json
{
    "message": "Usuario creado exitosamente",
    "user": "user_20251117203959_8322"
}
```

### 2. Iniciar sesión
- **URL**: `/login`
- **Método**: `POST`
- **Descripción**: Inicia sesión y actualiza el streak del usuario
- **Cuerpo de la petición**:
```json
{
    "email": "usuario@email.com",
    "password": "contraseña123"
}
```
- **Respuesta exitosa**:
```json
{
    "message": "Inicio de sesión exitoso",
    "user": {
        "email": "usuario@email.com",
        "streak": 5,
        "best_streak": 10,
        "last_login": "2025-11-17T20:30:00"
    }
}
```
- **Lógica de Streak**:
  - Incrementa streak si el login es en día diferente y < 24h del último login
  - Resetea streak a 1 si han pasado > 24h
  - Actualiza best_streak si streak actual > mejor streak histórico

### 📚 Gestión de Diario

### 3. Información de la API
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

### 4. Obtener entradas del diario por usuario
- **URL**: `/diary/{user}`
- **Método**: `GET`
- **Descripción**: Obtiene todas las entradas del diario para un usuario específico
- **Parámetros**:
  - `user` (string): ID único del usuario
- **Respuesta**: Array de entradas del diario del usuario
```json
[
    {
        "user": "user_20251117203959_8322",
        "date": "2025-11-15",
        "overview": {
            "message": "Mensaje motivador",
            "recommendation": "Recomendación personalizada",
            "interesting_fact": "Dato curioso del día"
        },
        "mood": "feliz",
        "title": "Mi día especial",
        "note": "Nota del usuario",
        "img": "etiqueta_imagen"
    }
]
```

### 5. Obtener entrada por usuario y fecha
- **URL**: `/diary/{user}/{date}`
- **Método**: `GET`
- **Descripción**: Obtiene las entradas del diario para un usuario y fecha específica
- **Parámetros**:
  - `user` (string): ID único del usuario
  - `date` (string): Fecha en formato YYYY-MM-DD
- **Respuesta exitosa**: Array de entradas para la fecha especificada
- **Respuesta error (404)**:
```json
{
    "error": "No se encontraron datos para la fecha especificada"
}
```
- **Respuesta error (404) si usuario no existe**:
```json
{
    "detail": "Usuario no encontrado"
}
```

### 6. Agregar nueva entrada al diario
- **URL**: `/diary`
- **Método**: `POST`
- **Descripción**: Agrega una nueva entrada al diario con procesamiento de IA
- **Cuerpo de la petición**:
```json
{
    "user": "user_20251117203959_8322",
    "mood": "feliz",
    "title": "Mi día especial",
    "note": "Mi nota del día (opcional)",
    "img": "imagen_en_base64 (opcional)"
}
```
- **Campos**:
  - `user` (string, requerido): ID único del usuario
  - `mood` (string, requerido): Estado de ánimo del usuario
  - `title` (string, opcional): Título del día
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

### 7. Actualizar imagen en entrada existente
- **URL**: `/update-image`
- **Método**: `POST`
- **Descripción**: Analiza una imagen y actualiza la entrada del diario correspondiente con nueva IA
- **Cuerpo de la petición**:
```json
{
    "user": "user_20251117203959_8322",
    "date": "2025-11-15",
    "img": "imagen_en_base64"
}
```
- **Campos**:
  - `user` (string, requerido): ID único del usuario
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

### 8. Predecir etiqueta de imagen independiente
- **URL**: `/predict-image`
- **Método**: `POST`
- **Descripción**: Predice la etiqueta de una imagen sin actualizar ningún diario
- **Cuerpo de la petición**:
```json
{
    "user": "user_20251117203959_8322",
    "img": "imagen_en_base64"
}
```
- **Campos**:
  - `user` (string, requerido): ID único del usuario
  - `img` (string, requerido): Imagen codificada en base64
- **Respuesta exitosa**:
```json
{
    "predicted_label": "etiqueta_predicha",
    "diary_context": {
        "recent_entries": [...],
        "ai_description": "Descripción generada por IA basada en entradas recientes"
    }
}
```

### 9. Generar respuesta con prompt personalizado
- **URL**: `/prompt`
- **Método**: `POST`
- **Descripción**: Genera una respuesta personalizada usando IA basada en las entradas del diario de la semana actual del usuario
- **Cuerpo de la petición**:
```json
{
    "user": "user_20251117203959_8322",
    "prompt": "¿Cómo estuvo mi semana?"
}
```
- **Campos**:
  - `user` (string, requerido): ID único del usuario
  - `prompt` (string, requerido): Pregunta o prompt del usuario
- **Funcionalidad**:
  - Analiza las entradas del diario de la semana actual (desde el lunes) del usuario específico
  - Envía solo los campos `mood`, `note` e `img` a la IA
  - Genera una respuesta motivadora y personalizada
- **Respuesta exitosa**:
```json
{
    "response": "Respuesta motivadora basada en tu semana..."
}
```
- **Respuesta error (400)**:
```json
{
    "detail": "Error en generación de respuesta"
}
```

## 🤖 Integración con IA

### Google Gemini AI (gemini-2.5-flash)
La API utiliza Google Gemini AI para generar:
- **Mensajes motivadores personalizados** basados en el estado de ánimo
- **Recomendaciones diarias** adaptadas al contexto del usuario
- **Datos curiosos** relacionados con las actividades del día
- **Respuestas a prompts personalizados** analizando las entradas de la semana

### ResNet-50 para Clasificación de Imágenes
- Modelo pre-entrenado en ImageNet con 1000 clases
- Clasifica imágenes automáticamente cuando se suben al diario
- Las etiquetas predichas se integran en las recomendaciones de IA
- Procesa imágenes en formato base64

## 📊 Base de Datos

La aplicación utiliza TinyDB, una base de datos JSON ligera con dos archivos principales:

### **users.json**
- **Usuarios registrados** con credenciales y datos de streak
- **Campos**: `user` (ID único), `email`, `password`, `nickname`, `last_login`, `streak`, `best_streak`
- **Sistema de Streak**: Seguimiento automático de días consecutivos de login

### **db.json**
- **Entradas diarias** filtradas por usuario con fecha como identificador
- **Estados de ánimo, notas y títulos** del usuario
- **Etiquetas de imágenes** procesadas por ResNet-50
- **Respuestas generadas por IA** (overview con mensaje, recomendación y dato curioso)
- Los datos se almacenan por usuario y se actualizan automáticamente si ya existe una entrada para la fecha actual

## 🔐 Sistema de Autenticación

### **Validación de Usuario**
- Todos los endpoints que requieren `user` validan que el usuario existe en `users.json`
- Retorna error `404 - Usuario no encontrado` si el ID no existe

### **Registro de Usuarios**
- Genera ID único con timestamp: `user_YYYYMMDDHHMMSS_XXXX`
- Valida emails únicos y almacena credenciales

### **Sistema de Streak**
- **Incremento**: Solo en días diferentes y < 24h del último login
- **Reset**: A 1 si han pasado > 24h del último login
- **Mejor Streak**: Se actualiza automáticamente cuando se supera el récord

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
- Las bases de datos se crean automáticamente en la primera ejecución
- Todos los endpoints con `user` validan la existencia del usuario
- **Diferencia entre endpoints de imágenes**:
  - `/update-image`: Actualiza una entrada de diario existente con nueva imagen y regenera IA
  - `/predict-image`: Solo predice etiqueta de imagen y proporciona contexto del diario
- El endpoint `/prompt` solo analiza entradas de la semana actual del usuario específico
- Se utiliza configuración absoluta de rutas para archivos `.env` y modelos
- Los modelos Pydantic están organizados en `app/schemas.py` para mejor mantenibilidad

## 🧪 Desarrollo

### Ejecutar en modo desarrollo
```bash
fastapi dev app/main.py
```

### Estructura de Esquemas
Los modelos de datos están definidos en `app/schemas.py`:
- `DiaryEntry`: Entrada de diario del usuario (incluye campo `user`)
- `GeminiResponseModel`: Respuesta estructurada con mensaje, recomendación y dato curioso
- `GeminiBaseResponse`: Respuesta simple para prompts personalizados
- `ImageInput`: Entrada para predicción y actualización de imágenes en diario (incluye `user`, `date`, `img`)
- `ImagePrediction`: Entrada para predicción independiente de imágenes (incluye `user`, `img`)
- `PromptInput`: Entrada para prompts personalizados (incluye campo `user`)
- `LoginInput`: Credenciales de inicio de sesión
- `SignupInput`: Datos de registro de nuevo usuario

### Flujo de Autenticación
1. **Registro**: `/signup` → genera ID único y almacena usuario
2. **Login**: `/login` → valida credenciales y actualiza streak
3. **Operaciones**: Todos los endpoints validan que el `user` existe
4. **Datos**: Cada usuario solo accede a sus propios datos de diario