"""
FEH SDS Autobuilder - Full replacement (Option 3)
- Deterministic 3-pass team builder
- Brigade-based co-occurrence synergy with usage-normalization
- Respects seed_units, forbidden_pairs, must_use, and banned_units_per_team
- Exposes FEHTeamBuilder and generate_fe_teams(...) for compatibility with API wrapper
"""

from typing import List, Optional, Dict, Tuple, Set, Any
import csv, os, itertools, functools
from collections import Counter

# -----------------------------
# Utility: CSV brigade statistics
# -----------------------------
@functools.lru_cache(maxsize=4)
def load_brigade_stats_from_csv(tuple_csv_paths: Tuple[str, ...]) -> Tuple[Dict[str,int], Dict[Tuple[str,str],int], int]:
    """
    Read one or more CSV files and compute brigade-level usage U and co-occurrence C.
    Expects CSV format described by the user (first 3 rows headers, then groups of 4 rows = one brigade).
    Columns F,H,J,L,N (0-based indices 5,7,9,11,13) contain units.
    Returns (U, C, max_usage)
    """
    U = Counter()
    C = Counter()

    for path in tuple_csv_paths or []:
        if not path or not os.path.exists(path):
            continue

        with open(path, newline='', encoding='utf-8', errors='replace') as fh:
            rows = list(csv.reader(fh))

        if len(rows) <= 3:
            continue

        data_rows = rows[3:]
        for i in range(0, len(data_rows), 4):
            brigade = data_rows[i:i+4]
            units = []
            for r in brigade:
                for idx in (5, 7, 9, 11, 13):   # F,H,J,L,N
                    if idx < len(r):
                        v = r[idx].strip()
                        if v:
                            units.append(v)

            if not units:
                continue

            uniq = list(dict.fromkeys(units))
            for u in uniq:
                U[u] += 1

            for a, b in itertools.combinations(sorted(uniq), 2):
                C[(a, b)] += 1
                C[(b, a)] += 1

    max_usage = max(U.values()) if U else 1
    return dict(U), dict(C), max_usage


# -----------------------------
# Synergy scoring functions
# -----------------------------
def synergy_S(U, C, u, v):
    cuv = C.get((u, v), 0)
    uu = U.get(u, 0)
    vv = U.get(v, 0)
    if (uu + vv) == 0:
        return 0.0
    return 2.0 * cuv / (uu + vv)


def final_synergy_score(U, C, max_usage, u, v, usage_weight=0.25):
    base = synergy_S(U, C, u, v)
    usage_factor = (U.get(u, 0) + U.get(v, 0)) / (2.0 * max_usage) if max_usage > 0 else 0.0
    return base * (1.0 + usage_weight * usage_factor)


def team_candidate_score(candidate, team_members, U, C, max_usage, usage_weight=0.25):
    if not team_members:
        return U.get(candidate, 0) / float(max_usage or 1)
    total = 0.0
    for m in team_members:
        total += final_synergy_score(U, C, max_usage, candidate, m, usage_weight)
    return total


# -----------------------------
# FEHTeamBuilder: 3-pass deterministic builder
# -----------------------------
class FEHTeamBuilder:
    def __init__(self, all_units=None, csv_file_path=None, team_size=5):
        self.all_units = list(all_units) if all_units else []
        self.team_size = team_size
        self.csv_paths = list(csv_file_path) if csv_file_path else []
        self.quality_score = {u: 1.0 for u in self.all_units}

    def set_quality_scores(self, qmap):
        self.quality_score.update(qmap)

    def calculate_conditional_synergy(self, candidate, team_members):
        """Small fallback synergy when no CSV synergy exists."""
        score = 0.0
        cand_tokens = set(candidate.lower().split())
        for m in team_members:
            if cand_tokens & set(m.lower().split()):
                score += 0.2
        return score

    def build_multiple_teams(
        self,
        available_units,
        num_teams=4,
        team_size=5,
        seed_units_per_team=None,
        forbidden_pairs=None,
        required_pairs=None,
        must_use_units=None,
        unit_quality_weight=1.0,
        usage_weight=0.25,
        synergy_weight=3.0,
        banned_units_per_team=None,
        csv_paths=None,
        debug=False,
    ):
        forbidden_pairs = forbidden_pairs or []
        required_pairs = required_pairs or []
        must_use_units = must_use_units or []
        seed_units_per_team = seed_units_per_team or [[] for _ in range(num_teams)]
        banned_units_per_team = banned_units_per_team or [set() for _ in range(num_teams)]
        csv_paths = csv_paths or self.csv_paths or []

        # Load synergy stats
        U, C, max_usage = load_brigade_stats_from_csv(tuple(csv_paths))

        pool = set(available_units)
        teams = [[] for _ in range(num_teams)]
        locked = [set() for _ in range(num_teams)]

        # ---- PASS 0: Seed units ----
        for t in range(num_teams):
            for u in seed_units_per_team[t]:
                if u in pool:
                    teams[t].append(u)
                    locked[t].add(u)
                    pool.remove(u)

        # Forbidden pair lookup
        forb = set()
        for pair in forbidden_pairs:
            if len(pair) == 2:
                a, b = pair
                forb.add((a, b))
                forb.add((b, a))

        def violates_forbidden(u, members):
            return any((u, m) in forb for m in members)

        # ---- PASS 1: Place must-use units ----
        remaining_must = [u for u in must_use_units if u in pool]
        remaining_must.sort(key=lambda x: U.get(x, 0), reverse=True)

        for mu in remaining_must:
            best_team = None
            best_score = -1e18
            for t in range(num_teams):
                if len(teams[t]) >= team_size:
                    continue
                if mu in banned_units_per_team[t]:
                    continue
                if mu in locked[t]:
                    continue
                if violates_forbidden(mu, teams[t]):
                    continue

                qual = self.quality_score.get(mu, 1.0)
                syn = team_candidate_score(mu, teams[t], U, C, max_usage, usage_weight)
                cond = self.calculate_conditional_synergy(mu, teams[t])
                score = unit_quality_weight * qual + synergy_weight * syn + 0.5 * cond

                if score > best_score:
                    best_score = score
                    best_team = t

            if best_team is not None:
                teams[best_team].append(mu)
                pool.remove(mu)

        # ---- PASS 2: Greedy synergy fill ----
        def det_sorted(it):
            return sorted(it, key=lambda s: (-U.get(s, 0), s.lower()))

        changed = True
        while changed:
            changed = False
            for t in range(num_teams):
                if len(teams[t]) >= team_size:
                    continue

                banned = banned_units_per_team[t]
                occupied = set().union(*teams)

                candidates = [
                    u for u in pool
                    if u not in banned and u not in occupied and not violates_forbidden(u, teams[t])
                ]

                if not candidates:
                    candidates = [
                        u for u in pool
                        if u not in banned and not violates_forbidden(u, teams[t])
                    ]

                if not candidates:
                    continue

                candidates = det_sorted(candidates)

                best_c = None
                best_score = -1e18

                for cand in candidates:
                    qual = self.quality_score.get(cand, 1.0)
                    syn = team_candidate_score(cand, teams[t], U, C, max_usage, usage_weight)
                    cond = self.calculate_conditional_synergy(cand, teams[t])
                    score = unit_quality_weight * qual + synergy_weight * syn + 0.5 * cond

                    if score > best_score or (
                        abs(score - best_score) < 1e-9 and (best_c is None or cand < best_c)
                    ):
                        best_score = score
                        best_c = cand

                if best_c is not None:
                    teams[t].append(best_c)
                    pool.remove(best_c)
                    changed = True

        # ---- PASS 3: Final fill fallback ----
        for t in range(num_teams):
            banned = banned_units_per_team[t]
            while len(teams[t]) < team_size:
                candidates = [
                    u for u in (set(available_units) - set(teams[t]))
                    if u not in banned and not violates_forbidden(u, teams[t])
                ]

                if not candidates:
                    candidates = [
                        u for u in available_units
                        if u not in banned and not violates_forbidden(u, teams[t])
                    ]
                    if not candidates:
                        break

                candidates = det_sorted(candidates)
                teams[t].append(candidates[0])

        # ---- Required pairs enforcement (soft) ----
        for pair in required_pairs:
            if len(pair) != 2:
                continue
            a, b = pair

            loc = {u: idx for idx, team in enumerate(teams) for u in team}
            if a in loc and b in loc and loc[a] != loc[b]:
                ta, tb = loc[a], loc[b]
                for i, u in enumerate(teams[ta]):
                    if u not in locked[ta] and u not in {a, b}:
                        if not violates_forbidden(b, [x for x in teams[ta] if x != u]):
                            teams[ta][i] = b
                            teams[tb] = [x if x != b else u for x in teams[tb]]
                            break

        return teams


# -----------------------------
# Wrapper function used by API
# -----------------------------
def generate_fe_teams(
    available_units,
    forbidden_pairs=None,
    required_pairs=None,
    seed_units=None,
    must_use_units=None,
    csv_paths=None,
    priority_weights=None,
    skip_header_rows=3,
    debug=False,
    banned_units_per_team=None,
):
    builder = FEHTeamBuilder(
        all_units=available_units or [],
        csv_file_path=csv_paths or [],
    )

    teams = builder.build_multiple_teams(
        available_units=available_units or [],
        num_teams=4,
        team_size=5,
        seed_units_per_team=seed_units or [[] for _ in range(4)],
        forbidden_pairs=forbidden_pairs or [],
        required_pairs=required_pairs or [],
        must_use_units=must_use_units or [],
        banned_units_per_team=banned_units_per_team or [set() for _ in range(4)],
        csv_paths=csv_paths or [],
        debug=debug,
    )

    return [{'team': t, 'captain_skill': None} for t in teams]

