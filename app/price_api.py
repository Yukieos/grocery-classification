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

@app.get("/search_price")
def search_price(q: str = Query(..., description="请输入商品名称")):
    norm_q = normalize(q)

    conn = psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME,
        user=DB_USER, password=DB_PASS,
        port=DB_PORT
    )
    
    cur = conn.cursor()
    cur.execute(
        "SELECT full_name, vendor, unit_price FROM products "
        "WHERE normalized_name LIKE %s",
        (f"%{norm_q}%",)
    )
    exact_rows = cur.fetchall()

    results = []
    if exact_rows:
        for full_name, vendor, price in exact_rows:
            results.append({
                "product_name": full_name,
                "vendor": vendor,
                "price": price,
                "similarity": 1.0
            })
    else:
        # fetch everything (you could add a LIMIT if your table is huge)
        cur.execute(
            "SELECT full_name, vendor, unit_price, normalized_name FROM products"
        )
        all_rows = cur.fetchall()
        for full_name, vendor, price, norm_name in all_rows:
            sim = compute_similarity(norm_q, norm_name)
            results.append({
                "product_name": full_name,
                "vendor": vendor,
                "price": price,
                "similarity": round(sim, 2)
            })
        # sort by descending similarity, then ascending price
        results.sort(key=lambda x: (-x["similarity"], x["price"]))
        results = results[:5]

    cur.close()
    conn.close()
    return results
