import datetime
from fastapi.responses import JSONResponse
from tinydb import TinyDB, Query
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import base64
from io import BytesIO
from PIL import Image
import torch
from torchvision import models, transforms
from google import genai
from fastapi import HTTPException

client = genai.Client()


# Load the pre-trained ResNet-50 model
resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

# Define the image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create FastAPI app
app = FastAPI(media_type="application/json; charset=utf-8")

# Initialize the database (data will be stored in 'db.json')
db = TinyDB('db.json', encoding='utf-8')

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiaryEntry(BaseModel):
    date: str
    overview: str
    note: str
    img: str | None  # Base64 encoded image

@app.get("/")
def read_root():
    return {"app": "Mr.Zorro",
            "version": "1.0.0",
            "description": "API para la app Mr.Zorro"}

@app.get("/diary")
async def get_diary_data():
    """
    Obtiene todos los datos del diario,
    que incluyen la fecha, resumen, nota e imagen
    codificada en base64.
    """
    return JSONResponse(content=db.all())

@app.get("/diary/{date}")
async def get_diary_by_date(date: str):
    """
    Obtiene los datos del diario para una fecha específica.
    Args:
        date (str): La fecha en formato 'YYYY-MM-DD'.
    """
    Entry = Query()
    result = db.search(Entry.date == date)
    if result:
        return JSONResponse(content=result)
    return JSONResponse(content={"error": "No data found for the given date"}, status_code=404)

@app.post("/diary")
async def add_diary_entry(entry: DiaryEntry):
    date = datetime.date.today().isoformat()
    entry.date = date
    db.insert(entry.model_dump())
    return JSONResponse(content={"message": "Entry added successfully"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)