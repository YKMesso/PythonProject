import json
import numpy as np
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb+srv://emmanuelojomo7:UEHOnRNZTOeew4h4@autoproducts.kkxex9t.mongodb.net/?retryWrites=true&w=majority&appName=autoProducts")
db = client["autoProducts"]  # Replace with your DB name
collection = db["products"]  # Replace with your collection name

# Load JSON file
with open("data/auto_products.json", "r") as file:
    data = json.load(file)  # Load JSON content

# Insert into MongoDB
if isinstance(data, list):  # Check if JSON contains a list of objects
    collection.insert_many(data)
else:
    collection.insert_one(data)

print("Data inserted successfully!")
