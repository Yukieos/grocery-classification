import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from app.utils import normalize, compute_similarity

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT", 5432)

app = FastAPI(
    title="Grocery Price Search API",
    description="Search for grocery prices by name."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# price_api.py
import os
import psycopg2
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.utils import normalize, compute_similarity

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT", 5432)

app = FastAPI(title="Grocery Price Search API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/search_price")
def search_price(q: str = Query(..., description="Please input the name of product")):
    norm_q = normalize(q)  # e.g. "whole milk"
    if not norm_q:
        return []
    conn = psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
        port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("SELECT full_name, vendor, unit_price FROM products;")
    all_rows = cur.fetchall()
    cur.close()
    conn.close()

    scored_list = []
    for full_name, vendor, price in all_rows:
        norm_name = normalize(full_name)  # e.g. "organic whole milk"
        sim_score = compute_similarity(norm_q, norm_name)
        if sim_score > 0:
            scored_list.append({
                "product_name": full_name,
                "vendor":       vendor,
                "price":        price,
                "score":        sim_score
            })
    if not scored_list:
        return []
    scored_list.sort(key=lambda x: -x["score"])
    top12_by_score = scored_list[:7]
    top12_by_score.sort(key=lambda x: x["price"])
    top6_cheapest = top12_by_score[:5]
    return [
        {
            "product_name": item["product_name"],
            "vendor":       item["vendor"],
            "price":        item["price"],
            "score":        item["score"]
        }
        for item in top6_cheapest
    ]
