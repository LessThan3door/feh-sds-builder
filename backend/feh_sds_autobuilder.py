# backend/feh_sds_autobuilder.py
"""
Adapter that loads the user's script (uploaded at path) and exposes
a stable function `generate_fe_teams(...)` that the FastAPI backend can call.

It attempts to locate a suitable function inside the user's script:
- generate_fe_teams
- build_multiple_teams
- generate_teams
- build_teams
- main (callable)
- run (callable)

If the located function requires different args, the adapter will try to
call it with named keyword arguments matching the JSON schema provided.
"""

import importlib.util
import importlib.machinery
import os
import types
from typing import List, Optional, Dict, Any, Set

# <-- UPDATE THIS PATH if you move the user script into your repo
USER_SCRIPT_PATH = "/mnt/data/FEH SDS Autobuilder.py"

_loaded_module = None

def _load_user_module(path: str) -> types.ModuleType:
    """Dynamically load the user script as a module and cache it."""
    global _loaded_module
    if _loaded_module is not None:
        return _loaded_module
    if not os.path.exists(path):
        raise FileNotFoundError(f"User script not found at: {path}")
    # Use SourceFileLoader to handle spaces in filename
    loader = importlib.machinery.SourceFileLoader("user_autobuilder", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    _loaded_module = module
    return module

def _pick_callable(module):
    """Return the best callable object from the module that looks like a generator."""
    candidates = [
        "generate_fe_teams",
        "generate_teams",
        "build_multiple_teams",
        "build_teams",
        "generate",
        "run",
        "main",
    ]
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    # fallback: any callable at module-level with 'team' or 'generate' in name
    for attr in dir(module):
        if attr.lower().startswith("gen") or "team" in attr.lower():
            fn = getattr(module, attr)
            if callable(fn):
                return fn
    raise RuntimeError("No suitable callable found in user script")

def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map the incoming JSON shape into a set of args suitable for the user's function.
    We pass through the following fields by default:
      - available_units
      - forbidden_pairs
      - required_pairs
      - seed_units
      - must_use_units
      - csv_paths
      - re_run (if present)
      - current_teams, original_teams, removed_units_per_team (if present)
      - banned_units_per_team (if present)
    """
    keys = [
        "available_units", "forbidden_pairs", "required_pairs", "seed_units",
        "must_use_units", "csv_paths", "re_run", "current_teams", "original_teams",
        "removed_units_per_team", "banned_units_per_team"
    ]
    norm = {}
    for k in keys:
        if k in payload:
            norm[k] = payload[k]
    # ensure lists exist for common fields
    norm.setdefault("available_units", payload.get("available_units", []))
    norm.setdefault("forbidden_pairs", payload.get("forbidden_pairs", []))
    norm.setdefault("required_pairs", payload.get("required_pairs", []))
    norm.setdefault("seed_units", payload.get("seed_units", []))
    norm.setdefault("must_use_units", payload.get("must_use_units", []))
    norm.setdefault("csv_paths", payload.get("csv_paths", []))
    return norm

def generate_fe_teams(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Public adapter function that the FastAPI server will call.
    `payload` is the JSON body from the frontend.
    Returns a list of dicts: [{'team': [...], 'captain_skill': ...}, ...]
    """
    module = _load_user_module(USER_SCRIPT_PATH)
    fn = _pick_callable(module)
    args = _normalize_payload(payload)

    # Try calling in a few compatible ways:
    # 1) If function signature is generate_fe_teams(payload_dict) -> expect it.
    try:
        # many user scripts implement a wrapper named generate_fe_teams that takes a dict
        result = fn(args)  # best-effort single-arg call
        # If result looks like teams, normalize and return
        if isinstance(result, list):
            # if the result already matches [{'team':[...], 'captain_skill':...}, ...] return as-is
            if len(result) > 0 and isinstance(result[0], dict) and 'team' in result[0]:
                return result
            # Otherwise, try to coerce list-of-lists into required format
            if all(isinstance(r, (list, tuple)) for r in result):
                return [{'team': list(r), 'captain_skill': None} for r in result]
    except TypeError:
        # function may expect keyword args; fall through
        pass
    except Exception:
        # allow fallback attempts below; don't swallow important errors
        raise

    # 2) Try calling with explicit keyword args if the function accepts them
    try:
        # Build kwargs mapped to common names
        kwargs = {
            "available_units": args.get("available_units"),
            "forbidden_pairs": args.get("forbidden_pairs"),
            "required_pairs": args.get("required_pairs"),
            "seed_units": args.get("seed_units"),
            "must_use_units": args.get("must_use_units"),
            "csv_paths": args.get("csv_paths"),
            "banned_units_per_team": args.get("banned_units_per_team"),
            "removed_units_per_team": args.get("removed_units_per_team"),
            "current_teams": args.get("current_teams"),
            "original_teams": args.get("original_teams"),
            "re_run": args.get("re_run", False),
        }
        # Clean kwargs: remove None
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        result = fn(**kwargs)
        if isinstance(result, list):
            if len(result)>0 and isinstance(result[0], dict) and 'team' in result[0]:
                return result
            if all(isinstance(r, (list, tuple)) for r in result):
                return [{'team': list(r), 'captain_skill': None} for r in result]
    except TypeError:
        pass

    # 3) If function returned nothing or raised; final attempt: look for function names inside module for stepwise calls
    # Try common function combos: build_multiple_teams(builder_args)
    for combo_name in ("build_multiple_teams", "build_teams", "generate_teams", "generate_fe_teams"):
        fn2 = getattr(module, combo_name, None)
        if callable(fn2):
            try:
                # attempt call with kwargs
                kwargs = {
                    "available_units": args.get("available_units"),
                    "forbidden_pairs": args.get("forbidden_pairs"),
                    "required_pairs": args.get("required_pairs"),
                    "seed_units": args.get("seed_units"),
                    "must_use_units": args.get("must_use_units"),
                    "csv_paths": args.get("csv_paths"),
                    "banned_units_per_team": args.get("banned_units_per_team"),
                }
                kwargs = {k:v for k,v in kwargs.items() if v is not None}
                result = fn2(**kwargs)
                if isinstance(result, list):
                    if len(result)>0 and isinstance(result[0], dict) and 'team' in result[0]:
                        return result
                    if all(isinstance(r, (list, tuple)) for r in result):
                        return [{'team': list(r), 'captain_skill': None} for r in result]
            except Exception:
                continue

    # If we reach here, we couldn't call the user's function in a normalized way
    raise RuntimeError("Could not invoke a team-generation function in the user script. "
                       "Please ensure it exports a callable named generate_fe_teams(...) or build_multiple_teams(...).")
