from typing import Union
from fastapi import FastAPI
from pymongo import MongoClient
from pydantic import BaseModel
app = FastAPI()

# Connect to MongoDB
client = MongoClient("mmongodb+srv://emmanuelojomo7:UEHOnRNZTOeew4h4@autoproducts.kkxex9t.mongodb.net/?retryWrites=true&w=majority&appName=autoProducts")
db = client["autoProducts"]
collection = db["data"]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


class Product(BaseModel):
    name: str
    price: float
    description: str = None
    category: str = None


@app.get("/getSingleProduct/{product_id}")
def get_single_product(product_id: int):
    product = collection.find_one({"id": product_id}, {"_id": 0})
    return product if product else {"message": "Product not found"}


@app.get("/getAll")
def get_all_products():
    products = list(collection.find({}, {"_id": 0}))
    return {"products": products}


@app.post("/addNew")
def add_new_product(product: Product):
    if collection.find_one({"id": product.id}):
        return {"message": "Product ID already exists"}
    collection.insert_one(product.dict())
    return {"message": "Product added successfully"}


@app.delete("/deleteOne/{product_id}")
def delete_one(product_id: int):
    result = collection.delete_one({"id": product_id})
    return {"message": "Product deleted"} if result.deleted_count else {"message": "Product not found"}


@app.get("/startsWith/{letter}")
def starts_with(letter: str):
    products = list(collection.find({"name": {"$regex": f"^{letter}", "$options": "i"}}, {"_id": 0}))
    return {"products": products}


@app.get("/paginate/{start_id}/{end_id}")
def paginate(start_id: int, end_id: int):
    products = list(collection.find({"id": {"$gte": start_id, "$lte": end_id}}, {"_id": 0}).limit(10))
    return {"products": products}


@app.get("/convert/{product_id}")
def convert_price(product_id: int):
    product = collection.find_one({"id": product_id}, {"_id": 0})
    if not product:
        return {"message": "Product not found"}

    # Get exchange rate from an online API
    response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
    if response.status_code == 200:
        exchange_rate = response.json().get("rates", {}).get("EUR", 1)
        price_in_euro = round(product["price"] * exchange_rate, 2)
        return {"product_id": product_id, "price_in_euro": price_in_euro}

    return {"message": "Failed to fetch exchange rate"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
