# backend/feh_sds_autobuilder.py
import os
import importlib.util
import importlib.machinery
from typing import Dict, Any, List

# Path to the original script you uploaded (leave as-is)
USER_SCRIPT_PATH = "/mnt/data/FEH SDS Autobuilder.py"

_loaded = None

def _load_user_module():
    global _loaded
    if _loaded is not None:
        return _loaded
    if not os.path.exists(USER_SCRIPT_PATH):
        raise FileNotFoundError(f"Could not find user script at {USER_SCRIPT_PATH}")
    loader = importlib.machinery.SourceFileLoader("user_autobuilder", USER_SCRIPT_PATH)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    _loaded = mod
    return mod

def generate_fe_teams(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Adapter that calls the FEHTeamBuilder from the user's script.
    Expects the payload shape your frontend sends.
    Returns [{'team': [...], 'captain_skill': None}, ...]
    """
    mod = _load_user_module()

    if not hasattr(mod, "FEHTeamBuilder"):
        raise RuntimeError("User script does not define FEHTeamBuilder")

    FEHTeamBuilder = getattr(mod, "FEHTeamBuilder")

    available_units = payload.get("available_units", [])
    forbidden_pairs = payload.get("forbidden_pairs", [])
    required_pairs = payload.get("required_pairs", [])
    seed_units = payload.get("seed_units", [[] for _ in range(4)])
    must_use_units = payload.get("must_use_units", [])
    excluded_units_per_team = payload.get("excluded_units_per_team", [[], [], [], []])
    csv_paths = payload.get("csv_paths", [])

    # normalize excluded -> sets if needed by user code
    excluded_sets = [set(x) for x in excluded_units_per_team]

    # instantiate builder (adjust constructor args if your class uses different names)
    # many user scripts accept csv_paths and skip_header_rows; adapt if necessary
    builder = FEHTeamBuilder(csv_paths=csv_paths) if "csv_paths" in FEHTeamBuilder.__init__.__code__.co_varnames else FEHTeamBuilder()

    # Call the user's builder method (adjust name if different)
    if hasattr(builder, "build_multiple_teams"):
        teams_raw = builder.build_multiple_teams(
            available_units=available_units,
            forbidden_pairs=forbidden_pairs,
            required_pairs=required_pairs,
            seed_units_per_team=seed_units,
            must_use_units=must_use_units,
            excluded_units_per_team=excluded_sets,
            csv_paths=csv_paths
        )
    else:
        # fallback: try common function name on module
        for fn_name in ("generate_fe_teams", "generate_teams", "build_teams"):
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                teams_raw = fn(payload)
                break
        else:
            raise RuntimeError("Could not find a callable to generate teams in user script")

    # normalize output
    out = []
    for t in teams_raw:
        out.append({"team": list(t), "captain_skill": None})
    return out
