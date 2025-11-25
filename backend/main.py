from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
from pathlib import Path

# Import the team builder
import sys
sys.path.append(str(Path(__file__).parent.parent))
from backend.team_builder import FEHTeamBuilder

app = FastAPI(title="FEH Team Generator")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent  # This is /app/backend
STATIC_DIR = BASE_DIR / "static"  # This is /app/backend/static
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"  # This is /app/uploads
UPLOAD_DIR.mkdir(exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    return FileResponse(str(index_path))


class GenerateRequest(BaseModel):
    available_units: List[str]
    seed_units: Optional[List[List[str]]] = None
    must_use_units: Optional[List[str]] = None
    forbidden_pairs: Optional[List[List[str]]] = None
    required_pairs: Optional[List[List[str]]] = None
    csv_filename: Optional[str] = None


class RegenerateRequest(BaseModel):
    edited_teams: List[List[str]]
    banned_assignments: List[dict]  # [{"unit": str, "team": int}, ...]
    all_available_units: List[str]
    must_use_units: Optional[List[str]] = None
    csv_filename: Optional[str] = None


@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate initial teams from scratch"""
    try:
        # Get CSV path if provided
        csv_paths = []
        if req.csv_filename:
            csv_path = UPLOAD_DIR / req.csv_filename
            if csv_path.exists():
                csv_paths = [str(csv_path)]
        
        # Use default CSV if exists and none provided
        if not csv_paths:
            default_csv = BASE_DIR / "dataset1.csv"
            if default_csv.exists():
                csv_paths = [str(default_csv)]
        
        # Initialize builder
        # Pass list of CSV paths or empty list
        builder = FEHTeamBuilder(
            csv_file_path=csv_paths if csv_paths else [],
            priority_weights=[1.0] if csv_paths else [],
            skip_header_rows=3
        )
        
        # Convert forbidden/required pairs from lists to tuples
        forbidden_pairs = [tuple(pair) for pair in (req.forbidden_pairs or [])]
        required_pairs = [tuple(pair) for pair in (req.required_pairs or [])]
        
        # Build teams
        teams = builder.build_multiple_teams(
            available_units=req.available_units,
            num_teams=4,
            team_size=5,
            seed_units_per_team=req.seed_units or [[], [], [], []],
            forbidden_pairs=forbidden_pairs,
            required_pairs=required_pairs,
            must_use_units=req.must_use_units or [],
            unit_quality_weight=0.8,
            debug=False
        )
        
        # Format response
        results = []
        for team in teams:
            captain_skill = builder.suggest_captain_skill(team) if csv_paths else None
            results.append({
                "team": team,
                "captain_skill": captain_skill
            })
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating teams: {str(e)}")


@app.post("/regenerate")
def regenerate(req: RegenerateRequest):
    """Regenerate teams after user edits (removals/moves)"""
    try:
        # Get CSV path if provided
        csv_paths = []
        if req.csv_filename:
            csv_path = UPLOAD_DIR / req.csv_filename
            if csv_path.exists():
                csv_paths = [str(csv_path)]
        
        # Use default CSV if exists
        if not csv_paths:
            default_csv = BASE_DIR / "dataset1.csv"
            if default_csv.exists():
                csv_paths = [str(default_csv)]
        
        # Initialize builder
        builder = FEHTeamBuilder(
            csv_file_path=csv_paths if csv_paths else [],
            priority_weights=[1.0] if csv_paths else [],
            skip_header_rows=3
        )
        
        # Process banned assignments into excluded_units_per_team
        excluded_units_per_team = [[], [], [], []]
        for ban in req.banned_assignments:
            team_idx = ban.get("team")
            unit = ban.get("unit")
            if team_idx is not None and unit:
                excluded_units_per_team[team_idx].append(unit)
        
        # Use edited teams as seeds (remaining units after removals)
        seed_units = [team[:] for team in req.edited_teams]
        
        # Build teams with exclusions
        teams = builder.build_multiple_teams(
            available_units=req.all_available_units,
            num_teams=4,
            team_size=5,
            seed_units_per_team=seed_units,
            must_use_units=req.must_use_units or [],
            unit_quality_weight=0.8,
            excluded_units_per_team=excluded_units_per_team,
            debug=False
        )
        
        # Format response
        results = []
        for team in teams:
            captain_skill = builder.suggest_captain_skill(team) if csv_paths else None
            results.append({
                "team": team,
                "captain_skill": captain_skill
            })
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error regenerating teams: {str(e)}")


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file for synergy data"""
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"filename": file.filename, "message": "CSV uploaded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading CSV: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "healthy"}
