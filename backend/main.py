from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from backend.feh_sds_autobuilder import generate_fe_teams
import os

app = FastAPI(title="FEH Team Generator")

# Serve frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_frontend():
    index_path = os.path.join(static_dir, "index.html")
    return FileResponse(index_path)


class GenerateRequest(BaseModel):
    available_units: List[str]
    forbidden_pairs: Optional[List[List[str]]] = None
    required_pairs: Optional[List[List[str]]] = None
    seed_units: Optional[List[List[str]]] = None
    must_use_units: Optional[List[str]] = None
    csv_paths: Optional[List[str]] = None
    excluded_units_per_team: Optional[List[List[str]]] = None
    current_teams: Optional[List[List[str]]] = None
    original_teams: Optional[List[List[str]]] = None
    removed_units_per_team: Optional[List[List[str]]] = None
    banned_units_per_team: Optional[List[List[str]]] = None
    re_run: Optional[bool] = False


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        payload = req.dict()
        teams = generate_fe_teams(payload)
        return {"teams": teams}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error building teams: {e}")
