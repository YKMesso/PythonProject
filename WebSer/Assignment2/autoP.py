import csv
import json
from pymongo import MongoClient

uri = "mongodb+srv://emmanuelojomo7:UEHOnRNZTOeew4h4@autoproducts.kkxex9t.mongodb.net/?retryWrites=true&w=majority&appName=autoProducts"
# Create a new client and connect to the server
client = MongoClient(uri)
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)