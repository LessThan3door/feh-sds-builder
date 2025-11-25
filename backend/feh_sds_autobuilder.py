# backend/feh_sds_autobuilder.py

from typing import Dict, Any, List, Set
from backend.feh_sds_autobuilder import FEHTeamBuilder   # <-- YOUR REAL CLASS

def generate_fe_teams(payload: Dict[str, Any]):
    """
    Entry point for backend/main.py.
    
    It uses your FEHTeamBuilder class *directly*, without trying to load
    any external .py files.
    """

    # Extract fields from payload
    available_units = payload.get("available_units", [])
    forbidden_pairs = payload.get("forbidden_pairs", [])
    required_pairs = payload.get("required_pairs", [])
    seed_units = payload.get("seed_units", [[] for _ in range(4)])
    must_use_units = payload.get("must_use_units", [])
    excluded_units_per_team = payload.get("excluded_units_per_team", [[], [], [], []])
    csv_paths = payload.get("csv_paths", [])

    # Convert lists to sets
    excluded_units_per_team = [
        set(team) for team in excluded_units_per_team
    ]

    # Initialize builder
    builder = FEHTeamBuilder(csv_paths=csv_paths, skip_header_rows=3)

    # Call your real logic
    teams_raw = builder.build_multiple_teams(
        available_units=available_units,
        forbidden_pairs=forbidden_pairs,
        required_pairs=required_pairs,
        seed_units_per_team=seed_units,
        excluded_units_per_team=excluded_units_per_team,
        required_units_per_team=[[] for _ in range(4)],
        must_use_units=must_use_units,
    )

    # Normalize output for frontend
    results = []
    for team in teams_raw:
        results.append({
            "team": list(team),
            "captain_skill": None
        })

    return results
