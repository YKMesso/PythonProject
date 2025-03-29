from fastapi import FastAPI
from pymongo import MongoClient
from pydantic import BaseModel

app = FastAPI()

# Connect to MongoDB
client = MongoClient("mmongodb+srv://emmanuelojomo7:UEHOnRNZTOeew4h4@autoproducts.kkxex9t.mongodb.net/?retryWrites=true&w=majority&appName=autoProducts")
db = client["inventory_db"]
collection = db["products"]


class Product(BaseModel):
    name: str
    price: float
    description: str = None
    category: str = None

@app.get("/getAllProd")
def get_all_products():
    products = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB _id
    return {"products": products}

@app.get("/getSingleProduct/{product_name}")
def get_single_product(product_name: str):
    product = collection.find_one({"name": product_name}, {"_id": 0})
    return product if product else {"message": "Product not found"}

@app.post("/insertNewProduct")
def insert_new_product(product: Product):
    if collection.find_one({"name": product.name}):
        return {"message": "Product already exists"}
    collection.insert_one(product.dict())
    return {"message": "Product added successfully"}

@app.get("/search")
def search_products(category: str = None, min_price: float = 0, max_price: float = None):
    query = {"price": {"$gte": min_price}}
    if max_price:
        query["price"]["$lte"] = max_price
    if category:
        query["category"] = category
    products = list(collection.find(query, {"_id": 0}))
    return {"products": products}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
