"""
Database connection and initialization for MongoDB
"""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from .models import Therapist, User, DiaryEntryDoc, ChatSession
import logging

logger = logging.getLogger(__name__)

async def init_database():
    """Initialize MongoDB connection and Beanie ODM"""
    try:
        # MongoDB connection
        MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        DATABASE_NAME = os.getenv("DATABASE_NAME", "mrzorro_db")

        logger.info(f"Connecting to MongoDB at {MONGODB_URL}")

        # Create MongoDB client
        client = AsyncIOMotorClient(MONGODB_URL)
        database = client[DATABASE_NAME]

        # Initialize beanie with document models
        await init_beanie(database=database, document_models=[Therapist, User, DiaryEntryDoc, ChatSession])

        logger.info("Successfully connected to MongoDB and initialized Beanie")
        return database

    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise