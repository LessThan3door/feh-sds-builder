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
    num_teams: int = 4



class RegenerateRequest(BaseModel):
    edited_teams: List[List[str]]
    banned_assignments: List[dict]  # [{"unit": str, "team": int}, ...]
    all_available_units: List[str]
    must_use_units: Optional[List[str]] = None
    csv_filename: Optional[str] = None
    num_teams: int

@app.get("/top-units")
def get_top_units(n: int = 50):
    """Return the top N units by usage."""
    try:
        # --- Find CSV path just like in /debug-csv ---
        csv_paths = []
        upload_files = list(UPLOAD_DIR.glob("*.csv"))
        if upload_files:
            csv_paths = [str(upload_files[0])]

        if not csv_paths:
            possible_paths = [
                BASE_DIR.parent / "dataset1.csv",
                BASE_DIR / "dataset1.csv",
                Path("/app/dataset1.csv"),
            ]
            for path in possible_paths:
                if path.exists():
                    csv_paths = [str(path)]
                    break

        if not csv_paths:
            return {"error": "No CSV found"}

        # --- Build dataset ---
        builder = FEHTeamBuilder(
            csv_file_path=csv_paths,
            priority_weights=[1.0],
            skip_header_rows=3
        )

        # --- Sort all units by usage ---
        sorted_units = sorted(builder.unit_counts.items(),
                              key=lambda x: x[1],
                              reverse=True)

        # --- Return ONLY the unit names (not counts) ---
        return {
            "units": [u for (u, c) in sorted_units[:n]],
            "total_available": len(sorted_units),
            "csv_path": csv_paths[0]
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}



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
            # Correct dataset location: backend/.. = project root
            default_csv = BASE_DIR.parent / "dataset1.csv"

            if default_csv.exists():
                csv_paths = [str(default_csv)]
            else:
                raise RuntimeError(f"dataset1.csv not found at {default_csv}")

        
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
            num_teams=req.num_teams,
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
            captain = builder.choose_best_captain(team)
            skill = builder.suggest_captain_skill(team)

            results.append({
                "team": team,
                "captain": captain,
                "captain_skill": skill
            })

        return results

        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating teams: {str(e)}")


@app.post("/regenerate")
def regenerate(req: RegenerateRequest):
    """Rebuild teams after manual edits"""
    try:
        # --------------------------
        # Load CSV
        # --------------------------
        csv_paths = []
        if req.csv_filename:
            csv_path = UPLOAD_DIR / req.csv_filename
            if csv_path.exists():
                csv_paths = [str(csv_path)]

        if not csv_paths:
            default_csv = BASE_DIR.parent / "dataset1.csv"
            if not default_csv.exists():
                raise RuntimeError("dataset1.csv not found")
            csv_paths = [str(default_csv)]

        builder = FEHTeamBuilder(
            csv_file_path=csv_paths,
            priority_weights=[1.0] * len(csv_paths),
            skip_header_rows=3
        )

        # --------------------------
        # 1. Collect removed units
        # --------------------------
        removed_units = set()
        for ban in req.banned_assignments:
            u = ban.get("unit")
            if u:
                removed_units.add(u.strip().lower())

        # --------------------------
        # 2. Filter available pool
        # --------------------------
        available_units = [
            u for u in req.all_available_units
            if u.strip().lower() not in removed_units
        ]

        # --------------------------
        # 3. Seeds from surviving units
        # --------------------------
        seed_units = []
        for team in req.edited_teams:
            cleaned = [
                u for u in team
                if u.strip().lower() not in removed_units
            ]
            seed_units.append(cleaned)

        # --------------------------
        # 4. Rebuild intelligently
        # --------------------------
        teams = builder.build_multiple_teams(
            available_units=available_units,
            num_teams=req.num_teams,
            team_size=5,
            seed_units_per_team=seed_units,
            must_use_units=req.must_use_units or [],
            unit_quality_weight=0.8,
            excluded_units_per_team=[[] for _ in range(4)],
            fill_all_slots=True,
            debug=False
        )

        # --------------------------
        # 5. Format output cleanly
        # --------------------------
        results = []
        for team in teams:
            if isinstance(team, dict):
                team_list = team["team"]
            else:
                team_list = team

            captain = builder.choose_best_captain(team_list)
            skill = builder.suggest_captain_skill(team_list)

            results.append({
                "team": team_list,
                "captain": captain,
                "captain_skill": skill
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


@app.get("/debug-csv")
def debug_csv():
    """Debug endpoint to check CSV loading and unit counts"""
    try:
        # Try to find and load CSV
        csv_paths = []
        upload_files = list(UPLOAD_DIR.glob("*.csv"))
        if upload_files:
            csv_paths = [str(upload_files[0])]
        
        if not csv_paths:
            possible_paths = [
                BASE_DIR.parent / "dataset1.csv",
                BASE_DIR / "dataset1.csv",
                Path("/app/dataset1.csv"),
            ]
            for path in possible_paths:
                if path.exists():
                    csv_paths = [str(path)]
                    break
        
        if not csv_paths:
            return {"error": "No CSV found"}
        
        # Load and parse
        builder = FEHTeamBuilder(
            csv_file_path=csv_paths,
            priority_weights=[1.0],
            skip_header_rows=3
        )
        
        # Get top 20 units by usage
        top_units = sorted(
            builder.unit_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
       
        # DEBUG: show top captain usage if available
        captain_usage = getattr(builder, "captain_usage", {})
        top_captains = sorted(captain_usage.items(), key=lambda x: x[1], reverse=True)[:20]
        print("DEBUG top_captains:", top_captains)
        
        debug_data = {}
        debug_data["captain_debug"] = []

        for t in results:
            debug_data["captain_debug"].append({
            "captain": t["team"][0] if t["team"] else None,
            "captain_skill": t["captain_skill"]
            })
        
        return {
            "DEBUG top_captains:": top_captains,
            "debug_csv": debug_data
            
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}
