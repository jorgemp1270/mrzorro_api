from tinydb import TinyDB, Query

# Initialize the database (data will be stored in 'db.json')
db = TinyDB('db.json')
# Insert a document into the database
db.insert({'name': 'Alice', 'age': 25})
db.insert({'name': 'Bob', 'age': 30})

# Query the database
# User = Query()
# result = db.search(User.name == 'Alice')
# print(result)  # Output: [{'name': 'Alice', 'age': 25}]

# Update a document
# db.update({'age': 26}, User.name == 'Alice')

# Delete a document
# db.remove(User.name == 'Bob')

# Fetch all documents
all_data = db.all()
print(all_data)  # Output: [{'name': 'Alice', 'age': 26}]