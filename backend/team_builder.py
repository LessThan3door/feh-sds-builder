import pandas as pd
import numpy as np
import os
from collections import defaultdict
from itertools import combinations


class FEHTeamBuilder:
    def __init__(self, csv_file_path, priority_weights=None, skip_header_rows=3):
        self.skip_header_rows = skip_header_rows
        self.datasets = []
        self.priority_weights = priority_weights or []

        if isinstance(csv_file_path, list):
            for p in csv_file_path:
                self.datasets.append(pd.read_csv(p, skiprows=skip_header_rows))
        else:
            self.datasets.append(pd.read_csv(csv_file_path, skiprows=skip_header_rows))

        # ===========================
        # LOAD SUPPORT ROLE RESTRICTIONS (>=8)
        # ===========================
        self.unit_support_roles = {}
        self.restricted_support_types = set()

        support_csv = os.path.join(os.path.dirname(__file__), "FEH SDS Unit Roles - Sheet1 (1).csv")

        if os.path.exists(support_csv):
            df_roles = pd.read_csv(support_csv)

            UNIT_COL = df_roles.columns[0]     # Unit Name
            SUPPORT_COL = df_roles.columns[11] # Unit support list
            SCORE_COL = df_roles.columns[19]   # Support score
            TYPE_COL = df_roles.columns[20]    # Role types list

            score_map = {}

            for _, row in df_roles.iterrows():
                if pd.isna(row[SCORE_COL]) or pd.isna(row[TYPE_COL]):
                    continue
                try:
                    score = int(row[SCORE_COL])
                except:
                    continue

                traits = [t.strip() for t in str(row[TYPE_COL]).split(",") if t.strip()]
                score_map.setdefault(score, set()).update(traits)

            for k in (8, 9, 10):
                self.restricted_support_types.update(score_map.get(k, set()))

            for _, row in df_roles.iterrows():
                unit = str(row[UNIT_COL]).strip()
                support_list = []

                if pd.notna(row[SUPPORT_COL]):
                    support_list = [s.strip() for s in str(row[SUPPORT_COL]).split(",") if s.strip()]

                restricted = {s for s in support_list if s in self.restricted_support_types}
                if restricted:
                    self.unit_support_roles[unit] = restricted


        # ===========================
        # CORRELATION DATA
        # ===========================
        self.unit_counts = defaultdict(float)
        self.unit_cooccurrence = defaultdict(lambda: defaultdict(float))

        self._calculate_correlations()

    def _calculate_correlations(self):
        for df, weight in zip(self.datasets, self.priority_weights or [1]):
            for _, row in df.iterrows():
                team_units = [u for u in row.dropna()]
                for u in team_units:
                    self.unit_counts[u] += weight
                for u, v in combinations(team_units, 2):
                    self.unit_cooccurrence[u][v] += weight
                    self.unit_cooccurrence[v][u] += weight

    def synergy(self, u, v):
        return self.unit_cooccurrence[u][v] + self.unit_cooccurrence[v][u]

    # ===========================
    # SUPPORT LOCK RULE
    # ===========================
    def violates_support_lock(self, unit, team):
        if unit not in self.unit_support_roles:
            return False

        unit_traits = self.unit_support_roles[unit]
        for teammate in team:
            if teammate in self.unit_support_roles:
                if unit_traits & self.unit_support_roles[teammate]:
                    return True
        return False


    def build_multiple_teams(self, available_units, seed_units=None,
                             num_teams=4, team_size=5,
                             forbidden_pairs=None, required_pairs=None,
                             must_use=None,
                             unit_weight=0.8):

        seed_units = seed_units or [[] for _ in range(num_teams)]
        forbidden_pairs = forbidden_pairs or []
        required_pairs = required_pairs or []
        must_use = must_use or []

        teams = [[] for _ in range(num_teams)]
        used_units = set()
        excluded_units_per_team = [set() for _ in range(num_teams)]

        # Seed placement
        for i, seeds in enumerate(seed_units):
            for u in seeds:
                if u in available_units and u not in used_units:
                    teams[i].append(u)
                    used_units.add(u)

        def violates_pair_constraints(team, unit):
            for a, b in forbidden_pairs:
                if (unit == a and b in team) or (unit == b and a in team):
                    return True
            return False

        def fulfills_required(unit, selected):
            if not required_pairs:
                return True
            for a, b in required_pairs:
                if unit == a and b not in selected and b in available_units:
                    return False
                if unit == b and a not in selected and a in available_units:
                    return False
            return True

        # ===========================
        # STEP 3: GREEDY BUILD
        # ===========================
        remaining_units = [u for u in available_units if u not in used_units]

        while remaining_units:
            best = None
            best_score = -1

            for unit in remaining_units:
                for t_idx, team in enumerate(teams):
                    if len(team) >= team_size:
                        continue

                    if unit in excluded_units_per_team[t_idx]:
                        continue

                    # SUPPORT LOCK
                    if self.violates_support_lock(unit, team):
                        continue

                    if violates_pair_constraints(team, unit):
                        continue

                    synergy_value = sum(self.synergy(unit, other) for other in team)
                    quality = self.unit_counts.get(unit, 0)
                    score = unit_weight * quality + (1 - unit_weight) * synergy_value

                    if score > best_score:
                        best_score = score
                        best = (unit, t_idx)

            if not best:
                break

            unit, team_idx = best
            teams[team_idx].append(unit)
            used_units.add(unit)
            remaining_units.remove(unit)

        # ===========================
        # STEP 4: MUST USE INSERTION
        # ===========================
        for must in must_use:
            if must in used_units or must not in available_units:
                continue

            inserted = False
            for t_idx in range(num_teams):
                if len(teams[t_idx]) >= team_size:
                    continue

                temp = teams[t_idx] + [must]

                if violates_pair_constraints(teams[t_idx], must):
                    continue

                # SUPPORT LOCK
                if self.violates_support_lock(must, temp):
                    continue

                teams[t_idx].append(must)
                used_units.add(must)
                inserted = True
                break

            if not inserted:
                print("WARNING: Could not insert must-use", must)

        return teams
