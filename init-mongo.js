// MongoDB initialization script
db = db.getSiblingDB('mrzorro_db');

// Create collections with indexes
db.createCollection('users');
db.createCollection('diary_entries');

// Create indexes for better performance
db.users.createIndex({ "user_id": 1 }, { unique: true });
db.users.createIndex({ "email": 1 }, { unique: true });
db.diary_entries.createIndex({ "user_id": 1, "date": 1 }, { unique: true });
db.diary_entries.createIndex({ "user_id": 1 });
db.diary_entries.createIndex({ "date": 1 });

print('MongoDB initialized with collections and indexes');