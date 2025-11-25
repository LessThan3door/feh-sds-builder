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
                            unit_quality_weight=0.8, excluded_units_per_team=None,
                            debug=False):
        """Build multiple teams by prioritizing highest synergies across all teams."""
        teams = [[] for _ in range(num_teams)]
        remaining_units = list(available_units)
        
        seed_units_per_team = seed_units_per_team or [[] for _ in range(num_teams)]
        excluded_units_per_team = excluded_units_per_team or [[] for _ in range(num_teams)]
        
        # Create a map of which seed units belong to which team
        seed_to_team = {}
        if seed_units_per_team:
            for team_idx, seed_list in enumerate(seed_units_per_team):
                if seed_list:
                    for unit in seed_list:
                        seed_to_team[unit] = team_idx
        
        # Calculate unit quality scores
        max_usage = max(self.unit_counts.values()) if self.unit_counts else 1
        unit_quality = {}
        for unit in available_units:
            usage = self.unit_counts.get(unit, 0)
            unit_quality[unit] = usage / max_usage if max_usage > 0 else 0
        
        # STEP 1: Add all seed units first
        for team_idx, seed_list in enumerate(seed_units_per_team):
            if seed_list:
                for unit in seed_list:
                    if unit in remaining_units:
                        teams[team_idx].append(unit)
                        remaining_units.remove(unit)
        
        # STEP 2: Add required pair partners for seeded units
        if required_pairs:
            for team_idx in range(num_teams):
                for u1, u2 in required_pairs:
                    if u1 in teams[team_idx] and u2 in remaining_units:
                        teams[team_idx].append(u2)
                        remaining_units.remove(u2)
                    elif u2 in teams[team_idx] and u1 in remaining_units:
                        teams[team_idx].append(u1)
                        remaining_units.remove(u1)
        
        # STEP 3: Place must_use_units
        if must_use_units:
            for unit in must_use_units:
                if unit not in remaining_units:
                    continue
                
                best_team_idx = None
                best_score = -float('inf')
                
                for team_idx in range(num_teams):
                    if len(teams[team_idx]) >= team_size:
                        continue
                    
                    if unit in excluded_units_per_team[team_idx]:
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
                    
                    if teams[team_idx]:
                        score = self.calculate_conditional_synergy(unit, teams[team_idx])
                    else:
                        score = self.unit_counts.get(unit, 0)
                    
                    if score > best_score:
                        best_score = score
                        best_team_idx = team_idx
                
                if best_team_idx is not None:
                    teams[best_team_idx].append(unit)
                    remaining_units.remove(unit)
        
        # STEP 4: Fill remaining slots
        while remaining_units and any(len(team) < team_size for team in teams):
            best_placement = None
            best_score = -float('inf')
            
            for unit in remaining_units:
                for team_idx in range(num_teams):
                    if len(teams[team_idx]) >= team_size:
                        continue
                    
                    if unit in excluded_units_per_team[team_idx]:
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
                    
                    if teams[team_idx]:
                        score = self.calculate_conditional_synergy(unit, teams[team_idx])
                    else:
                        score = self.unit_counts.get(unit, 0)
                    
                    if score > best_score:
                        best_score = score
                        best_placement = (team_idx, unit)
            
            if best_placement:
                team_idx, unit = best_placement
                teams[team_idx].append(unit)
                remaining_units.remove(unit)
            else:
                break
        
        return teams
    
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
