import sys

import pytest
from fastapi.testclient import TestClient
import requests
import sys
from fpdf import FPDF
from main import app  # Ensure 'main' is the name of your FastAPI script

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

def test_get_single_product():
    product_id = "AUTO999"  # Ensure this exists in your DB for a valid test
    response = client.get(f"/getSingleProduct/{product_id}")
    assert response.status_code == 200
    assert "message" not in response.json() or response.json()["Product ID"] == product_id

def test_get_all_products():
    response = client.get("/getAll")
    assert response.status_code == 200
    assert "CARS" in response.json()

def test_add_new_product():
    product_data = {
        "Product_ID": "AUTO123",
        "Name": "Test Car",
        "Unit_Price": 50000.0,
        "Stock_Quantity": 5,
        "Description": "A test vehicle"
    }
    response = client.post("/addNew", json=product_data)
    assert response.status_code == 200
    assert "message" in response.json()

def test_delete_one():
    product_id = "AUTO123"
    response = client.delete(f"/deleteOne/{product_id}")
    assert response.status_code == 200
    assert "message" in response.json()

def test_starts_with():
    response = client.get("/startsWith/T")
    assert response.status_code == 200
    assert "products" in response.json()

def test_paginate():
    response = client.get("/paginate/AUTO001/AUT030")
    assert response.status_code == 200
    assert "products" in response.json()

def test_convert_price():
    product_id = "AUTO90"  # Ensure this exists in your DB
    response = client.get(f"/convert/{product_id}")
    assert response.status_code == 200
    assert "price_in_euro" in response.json() or "message" in response.json()

def generate_pdf():
    with open("test_results.txt", "r") as f:
        test_results = f.readlines()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Unit Test Results", ln=True, align="C")
    pdf.ln(10)  # Add space

    pdf.set_font("Arial", size=10)
    for line in test_results:
        pdf.multi_cell(0, 5, line)

    pdf.output("unit_test_results.pdf")

# Run tests and save results
if __name__ == "__main__":
    with open("unit_test_results.txt", "w") as f:
        # Redirect stdout temporarily
        sys.stdout = f
        try:
            pytest.main(["-v"])
        finally:
            sys.stdout = sys.__stdout__

    generate_pdf()