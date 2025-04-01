import requests
from typing import Union
from fastapi import FastAPI
from pymongo import MongoClient
from pydantic import BaseModel
import uvicorn
app = FastAPI()

# Connect to MongoDB
client = MongoClient("mongodb+srv://emmanuelojomo7:UEHOnRNZTOeew4h4@autoproducts.kkxex9t.mongodb.net/?retryWrites=true&w=majority&appName=autoProducts")
db = client["autoProducts"]
collection = db["products"]


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


class Product(BaseModel):
    Product_ID: str
    Name: str
    Unit_Price: float
    Stock_Quantity: int
    Description: str = None
    start_id: str
    end_id: str



@app.get("/getSingleProduct/{product_id}")
def get_single_product(product_id: str):
    product = collection.find_one({"Product ID": str(product_id)}, {"_id": 0})
    return product if product else {"message": "Product not found"}


@app.get("/getAll")
def get_all_products():
    products = list(collection.find({}, {"_id": 0}))

    # Convert numeric fields safely
    for product in products:
        try:
            product["Unit Price"] = float(product["Unit Price"])
            product["Stock Quantity"] = int(product["Stock Quantity"])
        except (ValueError, KeyError, TypeError):
            pass  # Skip conversion if the value is missing or incorrect

    return {"CARS": products}


@app.post("/addNew")
def add_new_product(product: Product):
    # Check if the product already exists (by Product ID)
    existing_product = collection.find_one({"Product ID": product.Product_ID})
    if existing_product:
        return {"message": "Product ID already exists"}

    # Convert the product to the correct format
    new_product = {
        "Product ID": product.Product_ID,  # Ensure correct key
        "Name": product.Name,
        "Unit Price": product.Unit_Price,
        "Stock Quantity": product.Stock_Quantity,
        "Description": product.Description
    }

    # Insert into MongoDB
    collection.insert_one(new_product)
    return {"message": "Product added successfully"}


@app.delete("/deleteOne/{product_id}")
def delete_one(product: Product):
    result = collection.delete_one({"id": product.product_id})
    return {"message": "Product deleted"} if result.deleted_count else {"message": "Product not found"}


@app.get("/startsWith/{letter}")
def starts_with(letter: str):
    products = list(collection.find({"Name": {"$regex": f"^{letter}", "$options": "i"}}, {"_id": 0}))
    return {"products": products}


@app.get("/paginate/{start_id}/{end_id}")
def paginate(params: Product):
    products = list(collection.find({"Product ID": {"$gte": params.start_id, "$lte": params.end_id}}, {"_id": 0}).limit(10))
    return {"products": products}


@app.get("/convert/{product_id}")
def convert_price(product: Product):
    product = collection.find_one({"Product ID": product.product_id}, {"_id": 0})
    if not product:
        return {"message": "Product not found"}

    # Get exchange rate from an online API
    response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
    if response.status_code == 200:
        exchange_rate = response.json().get("rates", {}).get("EUR", 1)
        price_in_euro = round(float(product["Unit Price"]) * exchange_rate, 2)
        return {"product_id": product.product_id, "price_in_euro": price_in_euro}

    return {"message": "Failed to fetch exchange rate"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
