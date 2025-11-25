import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import shutil
import json

from backend.feh_sds_autobuilder import generate_fe_teams

# ---------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------
app = FastAPI()

# Allow your frontend to call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# STATIC FILES (index.html, app.js, style.css)
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found in backend/static")
    return FileResponse(index_path)


# ---------------------------------------------------------------------
# UPLOAD CSV
# ---------------------------------------------------------------------
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"csv_path": save_path}


# ---------------------------------------------------------------------
# GENERATE TEAMS
# ---------------------------------------------------------------------
from fastapi import Body, HTTPException

@app.post("/generate")
async def generate(payload: dict = Body(...)):
    """
    Accepts pure JSON:
      {
        available_units: [...],
        seed_units: [...],
        csv_paths: [...],
        etc.
      }
    """
    try:
        results = generate_fe_teams(payload)
        return {"teams": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error building teams: {e}")




# ---------------------------------------------------------------------
# HEALTH CHECK (Render requirement)
# ---------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
