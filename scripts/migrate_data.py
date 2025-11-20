"""
Data Migration Script: TinyDB to MongoDB
Migrates existing user and diary data from TinyDB JSON files to MongoDB
"""

import json
import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import init_database
from app.models import User, DiaryEntryDoc


async def migrate_tinydb_to_mongodb():
    """Migrate existing TinyDB data to MongoDB"""
    print("Starting migration from TinyDB to MongoDB...")

    try:
        # Initialize MongoDB connection
        await init_database()
        print("✓ Connected to MongoDB")

        # Load TinyDB data
        users_file_path = os.path.join('db', 'users.json')
        diary_file_path = os.path.join('db', 'db.json')

        if not os.path.exists(users_file_path):
            print(f"❌ Users file not found: {users_file_path}")
            return

        if not os.path.exists(diary_file_path):
            print(f"❌ Diary file not found: {diary_file_path}")
            return

        with open(users_file_path, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
        print(f"✓ Loaded users data from {users_file_path}")

        with open(diary_file_path, 'r', encoding='utf-8') as f:
            diary_data = json.load(f)
        print(f"✓ Loaded diary data from {diary_file_path}")

        # Migrate users
        migrated_users = 0
        for user_data in users_data.get('_default', {}).values():
            if isinstance(user_data, dict) and user_data.get('user'):
                try:
                    # Check if user already exists
                    existing_user = await User.find_one(User.user_id == user_data['user'])
                    if existing_user:
                        print(f"⚠ User {user_data['user']} already exists, skipping...")
                        continue

                    user = User(
                        user_id=user_data['user'],
                        email=user_data['email'],
                        password=user_data['password'],
                        nickname=user_data['nickname'],
                        last_login=datetime.fromisoformat(user_data['last_login']) if user_data.get('last_login') else None,
                        streak=user_data.get('streak', 1),
                        best_streak=user_data.get('best_streak', 1),
                        points=user_data.get('points', 0)
                    )
                    await user.insert()
                    migrated_users += 1
                    print(f"✓ Migrated user: {user_data['user']}")
                except Exception as e:
                    print(f"❌ Error migrating user {user_data.get('user', 'unknown')}: {e}")

        print(f"✓ Migrated {migrated_users} users")

        # Migrate diary entries
        migrated_entries = 0
        for entry_data in diary_data.get('_default', {}).values():
            if isinstance(entry_data, dict) and entry_data.get('user') and entry_data.get('date'):
                try:
                    # Check if entry already exists
                    existing_entry = await DiaryEntryDoc.find_one(
                        (DiaryEntryDoc.user_id == entry_data['user']) &
                        (DiaryEntryDoc.date == entry_data['date'])
                    )
                    if existing_entry:
                        print(f"⚠ Diary entry for {entry_data['user']} on {entry_data['date']} already exists, skipping...")
                        continue

                    diary_entry = DiaryEntryDoc(
                        user_id=entry_data['user'],
                        date=entry_data['date'],
                        overview=entry_data.get('overview', {}),
                        mood=entry_data.get('mood', ''),
                        title=entry_data.get('title'),
                        note=entry_data.get('note'),
                        img=entry_data.get('img')
                    )
                    await diary_entry.insert()
                    migrated_entries += 1
                    print(f"✓ Migrated diary entry: {entry_data['user']} - {entry_data['date']}")
                except Exception as e:
                    print(f"❌ Error migrating diary entry for {entry_data.get('user', 'unknown')}: {e}")

        print(f"✓ Migrated {migrated_entries} diary entries")
        print("🎉 Migration completed successfully!")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise


if __name__ == "__main__":
    print("=== TinyDB to MongoDB Migration Tool ===")
    print("This script will migrate your existing data from TinyDB to MongoDB")
    print("Make sure MongoDB is running and accessible")
    print("")

    confirmation = input("Do you want to proceed with the migration? (y/N): ")
    if confirmation.lower() not in ['y', 'yes']:
        print("Migration cancelled")
        exit(0)

    asyncio.run(migrate_tinydb_to_mongodb())