# backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from backend.feh_sds_autobuilder import generate_fe_teams

app = FastAPI(title="FEH Team Generator API")

class GenerateRequest(BaseModel):
    available_units: List[str]
    forbidden_pairs: Optional[List[List[str]]] = None
    required_pairs: Optional[List[List[str]]] = None
    seed_units: Optional[List[List[str]]] = None
    must_use_units: Optional[List[str]] = None
    csv_paths: Optional[List[str]] = None
    re_run: Optional[bool] = False
    current_teams: Optional[List[List[str]]] = None
    original_teams: Optional[List[List[str]]] = None
    removed_units_per_team: Optional[List[List[str]]] = None
    banned_units_per_team: Optional[List[List[str]]] = None

@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        payload = req.dict()
        teams = generate_fe_teams(payload)
        return {"teams": teams}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
