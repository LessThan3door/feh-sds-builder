# backend/feh_sds_autobuilder.py
#
# This adapter converts your existing FEH SDS Autobuilder script into a clean
# backend function the website can call.
#
# It preserves ALL logic exactly as your script defines it:
# - synergy
# - placement
# - required pairs
# - exclusions
# - seed units
# - re-run logic
# - CSV brigade synergy
#
# You DO NOT need to modify your main script.


import os
import importlib.util
import importlib.machinery
from typing import List, Optional, Set, Dict, Any


# IMPORTANT:
# Update this path to wherever your script is inside the repo.
# If you put your script under backend/, change this accordingly.
USER_SCRIPT_PATH = "/mnt/data/FEH SDS Autobuilder.py"


_loaded_mod = None


def _load_user_script():
    """Load the user's FEH SDS Autobuilder.py as a Python module."""
    global _loaded_mod

    if _loaded_mod is not None:
        return _loaded_mod

    if not os.path.exists(USER_SCRIPT_PATH):
        raise FileNotFoundError(
            f"Could not find FEH SDS Autobuilder.py at {USER_SCRIPT_PATH}"
        )

    loader = importlib.machinery.SourceFileLoader(
        "user_feh_autobuilder",
        USER_SCRIPT_PATH
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    _loaded_mod = module
    return module




def generate_fe_teams(payload: Dict[str, Any]):
    """
    Main entry called by backend/main.py

    Expected payload fields:
    - available_units
    - forbidden_pairs
    - required_pairs
    - seed_units
    - must_use_units
    - csv_paths
    - excluded_units_per_team  <-- you confirmed this is sent
    """

    mod = _load_user_script()

    # ---- Extract FEHTeamBuilder class ----
    if not hasattr(mod, "FEHTeamBuilder"):
        raise RuntimeError(
            "Your script does not define FEHTeamBuilder. "
            "Cannot generate teams."
        )

    FEHTeamBuilder = mod.FEHTeamBuilder

    # ---- Extract input fields from payload ----
    available_units = payload.get("available_units", [])
    forbidden_pairs = payload.get("forbidden_pairs", [])
    required_pairs = payload.get("required_pairs", [])
    seed_units = payload.get("seed_units", [[] for _ in range(4)])
    must_use_units = payload.get("must_use_units", [])
    excluded_units_per_team = payload.get("excluded_units_per_team", [[], [], [], []])
    csv_paths = payload.get("csv_paths", [])

    # Convert lists to sets where appropriate
    excluded_units_per_team = [
        set(team_list) for team_list in excluded_units_per_team
    ]

    # ---- Initialize the builder ----
    builder = FEHTeamBuilder(csv_paths=csv_paths, skip_header_rows=3)

    # ---- Call your real team builder method ----
    teams_raw = builder.build_multiple_teams(
        available_units=available_units,
        forbidden_pairs=forbidden_pairs,
        required_pairs=required_pairs,
        required_units_per_team=[[] for _ in range(4)],   # not used now
        excluded_units_per_team=excluded_units_per_team,
        seed_units_per_team=seed_units,
        must_use_units=must_use_units,
    )

    # ---- Normalize format for frontend ----
    # The frontend expects:
    # [
    #   {"team": [...], "captain_skill": null},
    #   ...
    # ]
    results = []
    for team_list in teams_raw:
        results.append({
            "team": list(team_list),
            "captain_skill": None
        })

    return results
