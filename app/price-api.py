from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from utils import normalize, compute_similarity

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
        host="db-foodprice.cs76a4esi9a9.us-east-1.rds.amazonaws.com",
        dbname="postgres",
        user="yukieos",
        password="drinkmoretea1",
        port=5432
    )
    cur = conn.cursor()
    cur.execute("SELECT full_name, vendor, unit_price, normalized_name FROM products;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for full_name, vendor, price, norm_name in rows:
        sim = compute_similarity(norm_q, norm_name)
        results.append((sim, price, full_name, vendor))

    results.sort(key=lambda x: (-x[0], x[1]))

    return [
        {
            "product_name": r[2],
            "vendor": r[3],
            "price": r[1],
            "similarity": round(r[0], 2)
        }
        for r in results[:5]
    ] 
