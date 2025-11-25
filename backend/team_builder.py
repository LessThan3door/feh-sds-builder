import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
import warnings
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
        
        # Calculate correlation matrices
        self.unit_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.unit_counts = defaultdict(int)
        self.conditional_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        if self.datasets:
            self._calculate_correlations()
        
    def _calculate_correlations(self):
        """Calculate unit co-occurrence statistics across all datasets."""
        for dataset_idx, df in enumerate(self.datasets):
            weight = self.priority_weights[dataset_idx]
            
            # Process each player's teams (every 4 rows = 1 brigade)
            for i in range(0, len(df), 4):
                player_teams = df.iloc[i:i+4]
                
                for _, row in player_teams.iterrows():
                    unit_columns = [5, 7, 9, 11, 13]
                    units = [str(row[col]).strip() for col in unit_columns 
                            if col < len(row) and pd.notna(row[col]) and str(row[col]).strip()]
                    
                    # Count individual units
                    for unit in units:
                        self.unit_counts[unit] += weight
                    
                    # Count pairwise co-occurrences
                    for unit1, unit2 in combinations(units, 2):
                        self.unit_cooccurrence[unit1][unit2] += weight
                        self.unit_cooccurrence[unit2][unit1] += weight
    
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
    
    def build_multiple_teams(self, available_units, num_teams=4, team_size=5,
                            seed_units_per_team=None, forbidden_pairs=None, 
                            required_pairs=None, must_use_units=None,
                            unit_quality_weight=0.7, excluded_units_per_team=None,
                            debug=False, required_pairs_per_team=None, 
                            required_units_per_team=None, fill_all_slots=True):
        """Build multiple teams by prioritizing highest synergies across all teams."""
        teams = [[] for _ in range(num_teams)]
        remaining_units = list(available_units)
        
        seed_units_per_team = seed_units_per_team or [None] * num_teams
        required_pairs_per_team = required_pairs_per_team or [None] * num_teams
        required_units_per_team = required_units_per_team or [None] * num_teams
        excluded_units_per_team = excluded_units_per_team or [[] for _ in range(num_teams)]
        
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
                # If u1 is seeded, reserve u2 for the same team
                if u1 in seed_to_team and u2 not in seed_to_team:
                    seed_to_team[u2] = seed_to_team[u1]
                # If u2 is seeded, reserve u1 for the same team
                elif u2 in seed_to_team and u1 not in seed_to_team:
                    seed_to_team[u1] = seed_to_team[u2]
        
        # STEP 1.5: Add required units for each team (non-seeds)
        if required_units_per_team:
            for team_idx, required_list in enumerate(required_units_per_team):
                if required_list:
                    for unit in required_list:
                        # Skip if already added as seed
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
        
        # Calculate unit quality scores (normalized usage frequency)
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
        
        # STEP 2: Add required pair partners for seeded units
        if required_pairs:
            for team_idx in range(num_teams):
                for u1, u2 in required_pairs:
                    # If one is in team, add the other
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
        
        # STEP 2.5: FORCE PLACEMENT of must_use_units (HIGHEST PRIORITY)
        if must_use_units:
            if debug:
                print("\n--- Placing must-use units (highest priority) ---")
            
            for unit in must_use_units:
                if unit not in remaining_units:
                    continue  # Already placed as seed or required partner
                
                # Check if unit has a required partner
                required_partner = None
                all_placed_units = [u for team in teams for u in team]
                
                if required_pairs:
                    for u1, u2 in required_pairs:
                        if u1 == unit and u2 not in all_placed_units:
                            required_partner = u2
                            break
                        elif u2 == unit and u1 not in all_placed_units:
                            required_partner = u1
                            break
                
                # Find best team for this must-use unit
                best_team_idx = None
                best_score = -float('inf')
                
                for team_idx in range(num_teams):
                    if len(teams[team_idx]) >= team_size:
                        continue
                    
                    # Check if unit is excluded from this team
                    if unit in excluded_units_per_team[team_idx]:
                        if debug:
                            print(f"  Skipping Team {team_idx+1} - {unit} is excluded")
                        continue
                    
                    # Check if reserved for different team
                    if unit in seed_to_team and seed_to_team[unit] != team_idx:
                        continue
                    
                    # Check forbidden pairs
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
                    
                    # If has required partner, check if partner can join
                    if required_partner:
                        if required_partner not in remaining_units or len(teams[team_idx]) + 2 > team_size:
                            continue
                        
                        # Check if partner is forbidden
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
                    
                    # Calculate synergy
                    if teams[team_idx]:
                        score = self.calculate_conditional_synergy(unit, teams[team_idx])
                    else:
                        score = self.unit_counts.get(unit, 0)
                    
                    if score > best_score:
                        best_score = score
                        best_team_idx = team_idx
                
                # Place the must-use unit
                if best_team_idx is not None:
                    teams[best_team_idx].append(unit)
                    remaining_units.remove(unit)
                    if debug:
                        print(f"Team {best_team_idx+1}: MUST-USE unit {unit} (synergy: {best_score:.4f})")
                    
                    # Add required partner if applicable
                    if required_partner and required_partner in remaining_units:
                        teams[best_team_idx].append(required_partner)
                        remaining_units.remove(required_partner)
                        if debug:
                            print(f"Team {best_team_idx+1}: Added required partner {required_partner}")
                else:
                    if debug:
                        print(f"ERROR: Could not place must-use unit {unit} - no valid team found!")
        
        if debug and must_use_units:
            print("--- Continuing with regular placements ---\n")
        
        # STEP 3: Fill remaining slots by prioritizing best synergies globally
        while remaining_units and any(len(team) < team_size for team in teams):
            best_placement = None
            best_score = -float('inf')
            
            # For each remaining unit, find its best team placement
            for unit in remaining_units:
                # Check if unit has a required partner
                required_partner = None
                if required_pairs:
                    # Check if partner is already placed on any team
                    all_placed_units = [u for team in teams for u in team]
                    
                    for u1, u2 in required_pairs:
                        if u1 == unit and u2 not in all_placed_units:
                            required_partner = u2
                            break
                        elif u2 == unit and u1 not in all_placed_units:
                            required_partner = u1
                            break
                
                # Try placing unit on each team
                for team_idx in range(num_teams):
                    if len(teams[team_idx]) >= team_size:
                        continue
                    
                    # Check if unit is excluded from this team
                    if unit in excluded_units_per_team[team_idx]:
                        continue
                    
                    # Check if unit is reserved for a different team
                    if unit in seed_to_team and seed_to_team[unit] != team_idx:
                        continue
                    
                    # Check forbidden pairs
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
                    
                    # If unit has required partner, check if partner can also join
                    if required_partner:
                        partner_can_join = required_partner in remaining_units
                        
                        if partner_can_join and len(teams[team_idx]) + 2 <= team_size:
                            # Check if partner is forbidden
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
                            # Can't place this unit if partner can't join
                            continue
                    
                    # Calculate synergy score for this placement
                    if teams[team_idx]:
                        score = self.calculate_conditional_synergy(unit, teams[team_idx])
                    else:
                        score = self.unit_counts.get(unit, 0)
                    
                    # Track best placement
                    if score > best_score:
                        best_score = score
                        best_placement = (team_idx, unit, required_partner)
            
            # Apply best placement
            if best_placement:
                team_idx, unit, partner = best_placement
                teams[team_idx].append(unit)
                remaining_units.remove(unit)
                
                if debug:
                    print(f"Team {team_idx+1}: Added {unit} (synergy: {best_score:.4f})")
                
                # Add required partner if applicable
                if partner and partner in remaining_units:
                    teams[team_idx].append(partner)
                    remaining_units.remove(partner)
                    if debug:
                        print(f"Team {team_idx+1}: Added required partner {partner}")
            else:
                # No valid placement found
                if debug:
                    print(f"No valid placement found for remaining units: {remaining_units}")
                break
        
        return teams
    
    def get_top_synergies(self, unit, top_n=10):
        """Get top synergistic units for a given unit."""
        if unit not in self.unit_counts:
            return []
        
        synergies = []
        for other_unit in self.unit_counts.keys():
            if other_unit != unit:
                score = self.calculate_synergy_score(unit, other_unit)
                synergies.append((other_unit, score))
        
        synergies.sort(key=lambda x: x[1], reverse=True)
        return synergies[:top_n]
        
    def suggest_captain_skill(self, team, datasets_with_skills=None):
        """Suggest captain skill based on team composition."""
        if not self.datasets:
            return None
            
        if not datasets_with_skills:
            datasets_with_skills = self.datasets
        
        skill_counts = defaultdict(int)
        captain_unit = team[0] if team else None
        
        for df in datasets_with_skills:
            for i in range(0, len(df), 4):
                player_teams = df.iloc[i:i+4]
                
                for _, row in player_teams.iterrows():
                    units = [str(u).strip() for u in row[2:8] if pd.notna(u) and str(u).strip()]
                    captain_skill = str(row[3]).strip() if len(row) > 3 and pd.notna(row[3]) else "Erosion"
                    
                    if captain_unit and captain_unit in units:
                        if units[0] == captain_unit:
                            skill_counts[captain_skill] += 2
                        else:
                            skill_counts[captain_skill] += 1
                    
                    overlap = len(set(team) & set(units))
                    if overlap >= 3:
                        skill_counts[captain_skill] += overlap * 0.5
        
        if skill_counts:
            return max(skill_counts.items(), key=lambda x: x[1])[0]
        else:
            return "Erosion"
