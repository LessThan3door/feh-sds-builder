# backend/feh_sds_autobuilder.py

from typing import Dict, Any, List

# IMPORTANT:
# Your real script must exist in the same folder:
# backend/FEH_SDS_Autobuilder.py
#
# And the filename MUST NOT contain spaces.
#
# Final structure:
#   backend/
#       feh_sds_autobuilder.py     ← this adapter
#       FEH_SDS_Autobuilder.py     ← your real script (renamed)


# Import your real FEH SDS builder class
from backend.FEH_SDS_Autobuilder import FEHTeamBuilder


def generate_fe_teams(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Adapter that calls the FEHTeamBuilder from the user's actual script.

    Expects the payload shape sent by the frontend:
      {
        "available_units": [...],
        "forbidden_pairs": [...],
        "required_pairs": [...],
        "seed_units": [...],
        "must_use_units": [...],
        "excluded_units_per_team": [...],
        "csv_paths": [...]
      }

    Returns:
      [
         {"team": [...], "captain_skill": None},
         ...
      ]
    """

    # Extract fields from payload
    available_units = payload.get("available_units", [])
    forbidden_pairs = payload.get("forbidden_pairs", [])
    required_pairs = payload.get("required_pairs", [])
    seed_units = payload.get("seed_units", [[] for _ in range(4)])
    must_use_units = payload.get("must_use_units", [])
    excluded_units_per_team = payload.get("excluded_units_per_team", [[], [], [], []])
    csv_paths = payload.get("csv_paths", [])

    # Convert excluded lists to sets
    excluded_sets = [set(team) for team in excluded_units_per_team]

    # Initialize your real builder
    # (Your script supports csv_paths and skip_header_rows)
    builder = FEHTeamBuilder(csv_paths=csv_paths, skip_header_rows=3)

    # Call your real generation logic
    teams_raw = builder.build_multiple_teams(
        available_units=available_units,
        forbidden_pairs=forbidden_pairs,
        required_pairs=required_pairs,
        seed_units_per_team=seed_units,
        excluded_units_per_team=excluded_sets,
        required_units_per_team=[[] for _ in range(4)],
        must_use_units=must_use_units,
    )

    # Normalize output for the frontend
    formatted = []
    for team in teams_raw:
        formatted.append({
            "team": list(team),
            "captain_skill": None
        })

    return formatted
