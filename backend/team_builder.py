import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
import warnings
import os

warnings.filterwarnings('ignore')

class FEHTeamBuilder:
    
    def __init__(self, csv_file_path, priority_weights=None, skip_header_rows=0):
        """
        Initialize the team builder with dataset(s).
        
        Parameters:
        -----------
        csv_file_path : str or list
            Path to CSV file(s). If list, first is primary dataset.
            Can be empty list for no CSV data.
        priority_weights : list, optional
            Weights for each dataset [1.0, 0.5, 0.3, ...]. First is always 1.0.
        skip_header_rows : int, optional
            Number of header rows to skip at the start of each CSV (default 0)
        """
        # Load datasets
        self.datasets = []
        if isinstance(csv_file_path, list):
            if csv_file_path:  # Not empty
                self.datasets = [pd.read_csv(f, header=None, skiprows=skip_header_rows) for f in csv_file_path]
                self.priority_weights = priority_weights or [1.0] + [0.5] * (len(csv_file_path) - 1)
            else:
                self.priority_weights = []
        else:
            if csv_file_path:  # Not empty string
                self.datasets = [pd.read_csv(csv_file_path, header=None, skiprows=skip_header_rows)]
                self.priority_weights = [1.0]
            else:
                self.priority_weights = []

        # ===========================
        # LOAD SUPPORT ROLE RESTRICTIONS (>=8)
        # ===========================
        self.unit_support_roles = {}
        self.restricted_support_types = set()

        support_csv = os.path.join(os.path.dirname(__file__), "FEH SDS Unit Roles - Sheet1 (1).csv")

        if os.path.exists(support_csv):
            df_roles = pd.read_csv(support_csv)

            # Column positions (based on your sheet layout)
            UNIT_COL = df_roles.columns[0]     # Col A - Unit Name
            SUPPORT_COL = df_roles.columns[11] # Col L - Unit Support List
            SCORE_COL = df_roles.columns[19]   # Col T - Point Value
            TYPE_COL = df_roles.columns[20]    # Col U - Support Types

            # Create point → support-type mapping
            score_map = {}

            for _, row in df_roles.iterrows():
                if pd.isna(row[SCORE_COL]) or pd.isna(row[TYPE_COL]):
                    continue

                try:
                    score = int(row[SCORE_COL])
                except:
                    continue

                traits = [t.strip() for t in str(row[TYPE_COL]).split(",")]

                score_map.setdefault(score, set()).update(traits)

            # Store only high-value supports (>=8)
            for k in (8, 9, 10):
                self.restricted_support_types.update(score_map.get(k, set()))

            # Map each unit → relevant restricted supports
            for _, row in df_roles.iterrows():
                unit = str(row[UNIT_COL]).strip()
                support_list = []

                if pd.notna(row[SUPPORT_COL]):
                    support_list = [s.strip() for s in str(row[SUPPORT_COL]).split(",")]

                restricted = set()
                for s in support_list:
                    if s in self.restricted_support_types:
                        restricted.add(s)

                if restricted:
                    self.unit_support_roles[unit] = restricted

        # Calculate correlation matrices
        self.unit_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.unit_counts = defaultdict(int)
        self.conditional_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        if self.datasets:
            self._calculate_correlations()
        
    def _calculate_correlations(self):
        """Calculate unit co-occurrence statistics across all datasets."""
        # Track how often each unit appears as captain (historical data)
        # and compute co-occurrence / usage statistics.
        self.captain_usage = defaultdict(int)

        # iterate datasets with explicit index -> weight (keeps exact scoping of `weight`)
        for dataset_idx, df in enumerate(self.datasets):
            # ensure we have a corresponding weight (fallback to 1 if missing)
            weight = self.priority_weights[dataset_idx] if dataset_idx < len(self.priority_weights) else 1

            
            # Process brigades (every 4 rows = 1 brigade)
            for i in range(0, len(df), 4):
                brigade = df.iloc[i:i+4]
    
                # Track which units appear in this brigade (for usage counts)
                brigade_units = set()
    
                # Process each team individually for co-occurrences
                for _, row in brigade.iterrows():
                    # --- TRACK HISTORICAL CAPTAIN USAGE ---
                    # Captain unit is at Column F -> row[5] (per your sheet)
                    if len(row) > 5 and pd.notna(row[5]):
                        captain = str(row[5]).strip()
                        if captain:
                            self.captain_usage[captain] += weight

                    # Now parse team unit columns
                    unit_columns = [5, 7, 9, 11, 13]
                    team_units = [str(row[col]).strip() for col in unit_columns 
                         if col < len(row) and pd.notna(row[col]) and str(row[col]).strip()]

                    
                    # Add to brigade set for usage counting
                    brigade_units.update(team_units)
                    
                    # Count co-occurrences WITHIN THIS TEAM (not brigade-wide)
                    for unit1, unit2 in combinations(team_units, 2):
                        self.unit_cooccurrence[unit1][unit2] += weight
                        self.unit_cooccurrence[unit2][unit1] += weight
                
                # Count each unit once per brigade for usage stats
                for unit in brigade_units:
                    self.unit_counts[unit] += weight

    
    def choose_best_captain(self, team):
        """Pick the unit in this team that was most often a historical captain."""
        if not hasattr(self, "captain_usage") or not self.captain_usage:
            return team[0] if team else None

        return max(team, key=lambda unit: self.captain_usage.get(unit, 0))

    def calculate_synergy_score(self, unit1, unit2):
        """Calculate synergy score between two units."""
        if unit1 not in self.unit_counts or unit2 not in self.unit_counts:
            return 0.0
        
        cooccur = self.unit_cooccurrence[unit1].get(unit2, 0)
        total_appearances = self.unit_counts[unit1] + self.unit_counts[unit2]
        if total_appearances == 0:
            return 0.0
        
        synergy = (2 * cooccur) / total_appearances
        return synergy
    
    def calculate_conditional_synergy(self, unit, given_units):
        """Calculate synergy of a unit given that certain units are already on the team."""
        if not given_units:
            return 0.0
        
        total_synergy = 0.0
        for given_unit in given_units:
            base_synergy = self.calculate_synergy_score(unit, given_unit)
            
            unit_frequency = self.unit_counts.get(unit, 1)
            given_frequency = self.unit_counts.get(given_unit, 1)
            
            rarity_weight = 1.0 / min(unit_frequency, given_frequency) if min(unit_frequency, given_frequency) > 0 else 1.0
            rarity_weight = min(rarity_weight, 5.0)
            
            total_synergy += base_synergy * (1 + rarity_weight * 0.2)
        
        return total_synergy / len(given_units)

    def violates_support_lock(self, unit, team):
        if unit not in self.unit_support_roles:
            return False

        unit_traits = self.unit_support_roles[unit]

        for teammate in team:
            if teammate in self.unit_support_roles:
                if unit_traits & self.unit_support_roles[teammate]:
                    return True

        return False
    
    
    def build_multiple_teams(self, available_units, num_teams=4, team_size=5,
                            seed_units_per_team=None, forbidden_pairs=None, 
                            required_pairs=None, must_use_units=None,
                            unit_quality_weight=0.3, excluded_units_per_team=None,
                            debug=False, required_pairs_per_team=None, 
                            required_units_per_team=None, fill_all_slots=True,
                            return_debug_log=False):
        """Build multiple teams by prioritizing highest synergies across all teams."""
        debug_log = []  # Track placement decisions
        teams = [[] for _ in range(num_teams)]
        remaining_units = list(available_units)
        
        seed_units_per_team = seed_units_per_team or [[] for _ in range(num_teams)]
        excluded_units_per_team = excluded_units_per_team or [[] for _ in range(num_teams)]
        required_pairs_per_team = required_pairs_per_team or [None] * num_teams
        required_units_per_team = required_units_per_team or [None] * num_teams
        
        # Create a map of which seed units belong to which team
        seed_to_team = {}
        if seed_units_per_team:
            for team_idx, seed_list in enumerate(seed_units_per_team):
                if seed_list:
                    for unit in seed_list:
                        seed_to_team[unit] = team_idx
        
        # Add required units to the reservation map
        if required_units_per_team:
            for team_idx, required_list in enumerate(required_units_per_team):
                if required_list:
                    for unit in required_list:
                        if unit not in seed_to_team:
                            seed_to_team[unit] = team_idx
        
        # CRITICAL: Also reserve partners of seeded units from required pairs
        if required_pairs:
            for u1, u2 in required_pairs:
                if u1 in seed_to_team and u2 not in seed_to_team:
                    seed_to_team[u2] = seed_to_team[u1]
                elif u2 in seed_to_team and u1 not in seed_to_team:
                    seed_to_team[u1] = seed_to_team[u2]
        
        # Add required units for each team (non-seeds)
        if required_units_per_team:
            for team_idx, required_list in enumerate(required_units_per_team):
                if required_list:
                    for unit in required_list:
                        if unit not in teams[team_idx] and unit in remaining_units:
                            teams[team_idx].append(unit)
                            remaining_units.remove(unit)
                            if debug:
                                print(f"Team {team_idx+1}: Added required unit {unit}")
        
        if debug:
            print("=" * 60)
            print("BUILDING TEAMS WITH GLOBAL SYNERGY OPTIMIZATION")
            print("=" * 60)
            if any(excluded_units_per_team):
                print("\nEXCLUDED UNITS PER TEAM:")
                for team_idx, excluded_list in enumerate(excluded_units_per_team):
                    if excluded_list:
                        print(f"  Team {team_idx+1}: {', '.join(excluded_list)}")
                print()
        
        # Calculate unit quality scores
        max_usage = max(self.unit_counts.values()) if self.unit_counts else 1
        unit_quality = {}
        for unit in available_units:
            usage = self.unit_counts.get(unit, 0)
            unit_quality[unit] = usage / max_usage if max_usage > 0 else 0
        
        if debug and unit_quality_weight > 0:
            print(f"\nUnit Quality Weight: {unit_quality_weight}")
            print("Top 10 units by usage:")
            sorted_quality = sorted(
                [(u, unit_quality[u], self.unit_counts.get(u, 0)) for u in available_units],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for unit, quality, usage in sorted_quality:
                print(f"  {unit}: {quality:.3f} (used {usage:.1f} times)")
            print()
        
        # STEP 1: Add all seed units first
        for team_idx, seed_list in enumerate(seed_units_per_team):
            if seed_list:
                for unit in seed_list:
                    if unit in remaining_units:
                        teams[team_idx].append(unit)
                        remaining_units.remove(unit)
                        if debug:
                            print(f"Team {team_idx+1}: Added seed unit {unit}")
        
        # Note: Captain selection happens naturally in STEP 3 when teams are empty
        # No special anti-synergy logic - just best synergy placement
        
        # STEP 2: Add required pair partners for seeded units
        if required_pairs:
            for team_idx in range(num_teams):
                for u1, u2 in required_pairs:
                    if u1 in teams[team_idx] and u2 in remaining_units:
                        teams[team_idx].append(u2)
                        remaining_units.remove(u2)
                        if debug:
                            print(f"Team {team_idx+1}: Added required partner {u2} for {u1}")
                    elif u2 in teams[team_idx] and u1 in remaining_units:
                        teams[team_idx].append(u1)
                        remaining_units.remove(u1)
                        if debug:
                            print(f"Team {team_idx+1}: Added required partner {u1} for {u2}")
        
        # STEP 2.5: Handle must_use_units - place them after team building if not already placed
        # Store for later - we'll handle these AFTER all other placements
        must_use_to_place = []
        if must_use_units:
            for unit in must_use_units:
                if unit in available_units and unit not in [u for team in teams for u in team]:
                    must_use_to_place.append(unit)
        
        # STEP 3: Fill remaining slots by prioritizing best synergies globally
        while remaining_units and any(len(team) < team_size for team in teams):
            best_placement = None
            best_score = -float('inf')
            
            # Debug: track all scores for first few placements
            if debug and sum(len(t) for t in teams) < 8:
                print(f"\n--- Evaluating placements (total placed: {sum(len(t) for t in teams)}) ---")
            
            for unit in remaining_units:
                required_partner = None
                if required_pairs:
                    all_placed_units = [u for team in teams for u in team]
                    
                    for u1, u2 in required_pairs:
                        if u1 == unit and u2 not in all_placed_units:
                            required_partner = u2
                            break
                        elif u2 == unit and u1 not in all_placed_units:
                            required_partner = u1
                            break
                
                for team_idx in range(num_teams):
                    if len(teams[team_idx]) >= team_size:
                        continue
                    
                    if unit in excluded_units_per_team[team_idx]:
                        continue
                    # Block duplicate ≥8 support types in this team
                    if self.violates_support_lock(unit, teams[team_idx]):
                        continue
    
                    
                    if unit in seed_to_team and seed_to_team[unit] != team_idx:
                        continue
                    
                    is_forbidden = False
                    if forbidden_pairs:
                        for team_unit in teams[team_idx]:
                            for f1, f2 in forbidden_pairs:
                                if (f1 == unit and f2 == team_unit) or (f2 == unit and f1 == team_unit):
                                    is_forbidden = True
                                    break
                            if is_forbidden:
                                break
                    
                    if is_forbidden:
                        continue
                    
                    if required_partner:
                        partner_can_join = required_partner in remaining_units
                        
                        if partner_can_join and len(teams[team_idx]) + 2 <= team_size:
                            partner_forbidden = False
                            if forbidden_pairs:
                                for team_unit in teams[team_idx]:
                                    for f1, f2 in forbidden_pairs:
                                        if (f1 == required_partner and f2 == team_unit) or \
                                           (f2 == required_partner and f1 == team_unit):
                                            partner_forbidden = True
                                            break
                                    if partner_forbidden:
                                        break
                            
                            if partner_forbidden:
                                continue
                        elif not partner_can_join or len(teams[team_idx]) + 2 > team_size:
                            continue
                    
                    if teams[team_idx]:
                        score = self.calculate_conditional_synergy(unit, teams[team_idx])
                    else:
                        score = self.unit_counts.get(unit, 0)
                    
                    # Debug output for first few units
                    if debug and sum(len(t) for t in teams) < 8:
                        print(f"  {unit} -> Team {team_idx+1}: score={score:.4f}, team={teams[team_idx]}")
                    
                    # When scores are very close, prefer teams with fewer units (to spread units)
                    tiebreaker = -len(teams[team_idx]) * 1e-10
                    
                    # Compare with tiebreaker but store the actual score
                    if score + tiebreaker > best_score:
                        best_score = score
                        best_placement = (team_idx, unit, required_partner)
            
            if best_placement:
                team_idx, unit, partner = best_placement
                teams[team_idx].append(unit)
                remaining_units.remove(unit)
                
                if debug:
                    print(f"Team {team_idx+1}: Added {unit} (synergy: {best_score:.4f})")
                
                if partner and partner in remaining_units:
                    teams[team_idx].append(partner)
                    remaining_units.remove(partner)
                    if debug:
                        print(f"Team {team_idx+1}: Added required partner {partner}")
            else:
                if debug:
                    print(f"No valid placement found for remaining units: {remaining_units}")
                break
        
        # STEP 4: FORCE must_use_units by replacing low-value units
        if must_use_to_place:
            if debug:
                print("\n--- Forcing must-use units by replacing low-value units ---")
            
            for must_use_unit in must_use_to_place:
                if must_use_unit not in remaining_units:
                    continue  # Already placed somehow
                
                # Find best team to place this unit (by swapping out lowest value unit)
                best_swap = None
                best_swap_score = float('inf')  # Lower is better for the unit being removed
                
                for team_idx in range(num_teams):
                    # Skip if must-use unit is excluded from this team
                    if must_use_unit in excluded_units_per_team[team_idx]:
                        continue
                    
                    # Try each non-seed unit as swap candidate
                    seed_units_for_team = seed_units_per_team[team_idx] if seed_units_per_team else []
                    
                    for candidate_idx, candidate_unit in enumerate(teams[team_idx]):
                        # Don't swap out seed units or other must-use units
                        if candidate_unit in (seed_units_for_team or []):
                            continue
                        if candidate_unit in (must_use_units or []):
                            continue
                        
                        # Calculate temp team without candidate
                        temp_team = [u for u in teams[team_idx] if u != candidate_unit]
                        
                        # Check if must_use_unit would be forbidden
                        is_forbidden = False
                        if forbidden_pairs:
                            for team_unit in temp_team:
                                for f1, f2 in forbidden_pairs:
                                    if (f1 == must_use_unit and f2 == team_unit) or \
                                       (f2 == must_use_unit and f1 == team_unit):
                                        is_forbidden = True
                                        break
                                if is_forbidden:
                                    break
                        
                        if is_forbidden:
                            continue
                        # Enforce support-type lock in must-use logic
                        if self.violates_support_lock(must_use_unit, temp_team):
                            continue

                        # Calculate swap score: lower is better (unit we want to remove)
                        # Factors: low synergy + low usage
                        candidate_synergy = self.calculate_conditional_synergy(candidate_unit, temp_team) if temp_team else 0
                        candidate_quality = unit_quality.get(candidate_unit, 0)
                        
                        # Score for removal: prefer low synergy and low quality
                        swap_score = candidate_synergy + (candidate_quality * unit_quality_weight)
                        
                        # Bonus: how well does must_use_unit fit?
                        must_use_synergy = self.calculate_conditional_synergy(must_use_unit, temp_team) if temp_team else 0
                        must_use_quality = unit_quality.get(must_use_unit, 0)
                        must_use_fit = must_use_synergy + (must_use_quality * unit_quality_weight)
                        
                        # Final score: low removal score + high fit score = good swap
                        final_score = swap_score - must_use_fit
                        
                        if final_score < best_swap_score:
                            best_swap_score = final_score
                            best_swap = (team_idx, candidate_idx, candidate_unit, swap_score)
                
                # Execute the swap
                if best_swap:
                    team_idx, candidate_idx, swap_out_unit, swap_out_score = best_swap
                    teams[team_idx][candidate_idx] = must_use_unit
                    remaining_units.remove(must_use_unit)
                    remaining_units.append(swap_out_unit)
                    
                    if debug:
                        print(f"Team {team_idx+1}: Swapped out '{swap_out_unit}' (score: {swap_out_score:.4f}) for MUST-USE '{must_use_unit}'")
                else:
                    if debug:
                        print(f"WARNING: Could not place must-use unit '{must_use_unit}'")

                # --- Ensure chosen captain is placed at index 0 for each team ---
        for ti, team in enumerate(teams):
            if not team:
                continue
            try:
                captain_unit = self.choose_best_captain(team)
            except Exception:
                captain_unit = team[0]  # fallback safe behavior
            if captain_unit and captain_unit in team:
                # preserve relative order of other members:
                team.remove(captain_unit)
                team.insert(0, captain_unit)
            # otherwise leave team unchanged
                        
        results = []

        for team in teams:
            captain_skill = self.suggest_captain_skill(team)
            results.append({
                "team": team,
                "captain_skill": captain_skill
            })

        return results

    
    def suggest_captain_skill(self, team, datasets_with_skills=None):
        """Pick captain skill based ONLY on historical usage of that captain."""
        try:
            datasets = datasets_with_skills if datasets_with_skills else self.datasets

            if not datasets or not team:
                return "No dataset/tea,"

            captain_unit = team[0].strip().lower()
            if not captain_unit:
                return "not captain unit"

            skill_counts = defaultdict(int)

            for df in datasets:
                for _, row in df.iterrows():
    
                    # Captain unit = column F (index 5)
                    if len(row) <= 5 or pd.isna(row[5]):
                        continue

                    historical_captain = str(row[5]).strip().lower()
                    if historical_captain != captain_unit:
                        continue

                    # Captain skill = column D (index 3)
                    if len(row) <= 3 or pd.isna(row[3]):
                        continue

                    skill = str(row[3]).strip()
                    if not skill:
                        continue

                    skill_counts[skill] += 1

            return max(skill_counts, key=skill_counts.get) if skill_counts else "Erosion"

        except Exception as e:
            return f"ERROR: {type(e).__name__} - {str(e)}"
