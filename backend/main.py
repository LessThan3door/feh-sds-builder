from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, conlist
from typing import List, Optional, Any
import os, shutil, traceback

from backend.feh_sds_autobuilder import generate_fe_teams


app = FastAPI(title="FEH Team Generator API")

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

app.mount("/frontend/static", StaticFiles(directory="frontend/static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "datasets")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mount frontend static so Render can serve /
app.mount("/frontend/static", StaticFiles(directory="frontend/static"), name="static-files")
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

class GenerateRequest(BaseModel):
    available_units: conlist(str, min_items=1)
    forbidden_pairs: Optional[List[List[str]]] = Field(default_factory=list)
    required_pairs: Optional[List[List[str]]] = Field(default_factory=list)
    seed_units: Optional[List[List[str]]] = Field(default_factory=lambda: [[], [], [], []])
    must_use_units: Optional[List[str]] = Field(default_factory=list)
    csv_paths: Optional[List[str]] = None

class RegenerateRequest(BaseModel):
    edited_teams: conlist(list, min_items=4, max_items=4)
    banned_assignments: Optional[List[dict]] = Field(default_factory=list)
    all_available_units: conlist(str, min_items=1)
    must_use_units: Optional[List[str]] = Field(default_factory=list)
    csv_paths: Optional[List[str]] = None

def normalize_banned_assignments(banned_assignments: List[dict], num_teams: int = 4):
    b = [set() for _ in range(num_teams)]
    for ban in banned_assignments:
        unit = ban.get("unit")
        team = ban.get("team")
        try:
            team_i = int(team)
        except Exception:
            continue
        if unit and 0 <= team_i < num_teams:
            b[team_i].add(unit)
    return b

@app.post("/generate")
async def generate(req: GenerateRequest):
    csv_paths = req.csv_paths or []
    for f in os.listdir(UPLOAD_FOLDER):
        if f.lower().endswith(".csv"):
            csv_paths.append(os.path.join(UPLOAD_FOLDER, f))
    try:
        results = generate_fe_teams(
            available_units=req.available_units,
            forbidden_pairs=req.forbidden_pairs,
            required_pairs=req.required_pairs,
            seed_units=req.seed_units,
            must_use_units=req.must_use_units,
            csv_paths=csv_paths or None,
            debug=False,
        )
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Server error building teams: " + str(e))

@app.post("/regenerate")
async def regenerate(req: RegenerateRequest):
    placed_units = set(u for team in req.edited_teams for u in team)
    available_pool = [u for u in req.all_available_units if u not in placed_units]
    banned_by_team = normalize_banned_assignments(req.banned_assignments, num_teams=4)
    csv_paths = req.csv_paths or []
    for f in os.listdir(UPLOAD_FOLDER):
        if f.lower().endswith(".csv"):
            csv_paths.append(os.path.join(UPLOAD_FOLDER, f))
    try:
        results = generate_fe_teams(
            available_units=available_pool,
            forbidden_pairs=[],
            required_pairs=[],
            seed_units=req.edited_teams,
            must_use_units=req.must_use_units,
            csv_paths=csv_paths or None,
            debug=False,
            banned_units_per_team=banned_by_team,
        )
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Server error regenerating teams: " + str(e))

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV allowed")
    safe_name = os.path.basename(file.filename)
    dest = os.path.join(UPLOAD_FOLDER, safe_name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": safe_name, "path": dest}
