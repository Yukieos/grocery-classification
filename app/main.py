from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from model_utils import infer_category
from utils import normalize, compute_similarity
import uvicorn

app = FastAPI(
    title="Grocery Photo Search API",
    description="Upload a photo, get back a predicted category or OCR text."
)

# （根据前端域名设置）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST","GET","OPTIONS"],
    allow_headers=["*"],
)

@app.post("/infer")
async def infer(photo: UploadFile = File(...)):
    """
    接收 multipart/form-data 的图片文件
    返回 JSON:{"category":..., "raw_text":..., "method":...}
    """
    if photo.content_type.split('/')[0] != 'image':
        raise HTTPException(400, "Only image uploads are supported.")
    img_bytes = await photo.read()
    result = infer_category(img_bytes)
    return result

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
