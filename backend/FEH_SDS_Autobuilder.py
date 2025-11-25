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
        priority_weights : list, optional
            Weights for each dataset [1.0, 0.5, 0.3, ...]. First is always 1.0.
        skip_header_rows : int, optional
            Number of header rows to skip at the start of each CSV (default 0)
        """
        # Load datasets
        if isinstance(csv_file_path, list):
            self.datasets = [pd.read_csv(f, header=None, skiprows=skip_header_rows) for f in csv_file_path]
            self.priority_weights = priority_weights or [1.0] + [0.5] * (len(csv_file_path) - 1)
        else:
            self.datasets = [pd.read_csv(csv_file_path, header=None, skiprows=skip_header_rows)]
            self.priority_weights = [1.0]
        
        # Calculate correlation matrices
        self.unit_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.unit_counts = defaultdict(int)
        self.conditional_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        self._calculate_correlations()
        
    def _calculate_correlations(self):
        """Calculate unit co-occurrence statistics across all datasets."""
        for dataset_idx, df in enumerate(self.datasets):
            weight = self.priority_weights[dataset_idx]
            
            # Process each player's teams (every 4 rows = 1 brigade)
            for i in range(0, len(df), 4):
                player_teams = df.iloc[i:i+4]
                
                for _, row in player_teams.iterrows():
                    # CSV structure: Region (0), Player (1), blank (2), Captain (3), blank (4), 
                    # Unit1 (5), blank (6), Unit2 (7), blank (8), Unit3 (9), blank (10), 
                    # Unit4 (11), blank (12), Unit5 (13)
                    # Extract units from columns 5, 7, 9, 11, 13 (every other column starting at 5)
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
        
        # Co-occurrence count
        cooccur = self.unit_cooccurrence[unit1].get(unit2, 0)
        
        # Normalize by individual frequencies (Jaccard-like similarity)
        total_appearances = self.unit_counts[unit1] + self.unit_counts[unit2]
        if total_appearances == 0:
            return 0.0
        
        # Synergy score: how often they appear together relative to individual appearances
        synergy = (2 * cooccur) / total_appearances
        
        return synergy
    
    def calculate_conditional_synergy(self, unit, given_units):
        """
        Calculate synergy of a unit given that certain units are already on the team.
        Higher weight for less common units with strong preferences.
        """
        if not given_units:
            return 0.0
        
        total_synergy = 0.0
        for given_unit in given_units:
            base_synergy = self.calculate_synergy_score(unit, given_unit)
            
            # Weight by rarity: rarer units with strong pairing get higher weight
            unit_frequency = self.unit_counts.get(unit, 1)
            given_frequency = self.unit_counts.get(given_unit, 1)
            
            # Inverse frequency weighting: rarer = higher weight
            rarity_weight = 1.0 / min(unit_frequency, given_frequency) if min(unit_frequency, given_frequency) > 0 else 1.0
            
            # Cap the rarity weight to prevent extreme values
            rarity_weight = min(rarity_weight, 5.0)
            
            total_synergy += base_synergy * (1 + rarity_weight * 0.2)
        
        return total_synergy / len(given_units)
    
    def check_antisynergy(self, unit1, unit2, threshold=0.05):
        """
        Check if two units have anti-synergy (rarely appear together).
        
        Parameters:
        -----------
        threshold : float
            If synergy score is below this, consider it anti-synergy
        """
        return self.calculate_synergy_score(unit1, unit2) < threshold
    
    def build_multiple_teams(self, available_units, num_teams=4, team_size=5,
                            seed_units_per_team=None, forbidden_pairs=None, 
                            required_pairs=None, required_pairs_per_team=None,
                            required_units_per_team=None, must_use_units=None,
                            role_constraints=None, unit_quality_weight=0.3, 
                            fill_all_slots=True, debug=False, 
                            excluded_units_per_team=None):
        """
        Build multiple teams by prioritizing highest synergies across all teams.
        
        Parameters:
        -----------
        available_units : list
            List of all available unit names
        num_teams : int
            Number of teams to build
        team_size : int
            Units per team
        seed_units_per_team : list of lists, optional
            Seed units for each team [[team1_seeds], [team2_seeds], ...]
        forbidden_pairs : list of tuples, optional
            Global forbidden pairs
        required_pairs : list of tuples, optional
            Required pairs that must be on same team (global, any team)
        required_pairs_per_team : list of lists of tuples, optional
            Required pairs for each specific team [[(u1,u2)], [(u3,u4)], ...]
        required_units_per_team : list of lists, optional
            Required units for each specific team [[team1_units], [team2_units], ...]
            These units MUST be on the specified team
        must_use_units : list, optional
            Units that MUST be placed on some team (but any team is fine)
        role_constraints : dict, optional
            Role assignments {'unit_name': 'role', ...}
        unit_quality_weight : float, optional
            Weight for unit quality bonus (0.0 = ignore usage, 1.0 = equal to synergy)
            Default 0.3 means quality is 30% as important as synergy
        fill_all_slots : bool, optional
            If True, ensures all teams are filled to team_size (default True)
            If False, teams may be smaller if units don't have good synergy
        debug : bool, optional
            Print debug information
        excluded_units_per_team : list of lists, optional
            Units that are banned from each team [[team1_excluded], [team2_excluded], ...]
            These units cannot be placed on their corresponding teams
        
        Returns:
        --------
        list : List of teams, each team is a list of units
        """
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
        """
        Suggest captain skill based on team composition.
        
        Parameters:
        -----------
        team : list
            List of units on the team
        datasets_with_skills : list of DataFrames, optional
            Datasets that include captain skill column (column 1)
        
        Returns:
        --------
        str : Suggested captain skill
        """
        if not datasets_with_skills:
            datasets_with_skills = self.datasets
        
        skill_counts = defaultdict(int)
        captain_unit = team[0] if team else None
        
        # Look for teams with similar composition or same captain
        for df in datasets_with_skills:
            for i in range(0, len(df), 4):
                player_teams = df.iloc[i:i+4]
                
                for _, row in player_teams.iterrows():
                    units = [str(u).strip() for u in row[2:8] if pd.notna(u) and str(u).strip()]
                    captain_skill = str(row[3]).strip() if len(row) > 3 and pd.notna(row[3]) else "Erosion"
                    
                    # Check if captain matches or team has significant overlap
                    if captain_unit and captain_unit in units:
                        if units[0] == captain_unit:  # Captain position
                            skill_counts[captain_skill] += 2  # Higher weight for exact captain match
                        else:
                            skill_counts[captain_skill] += 1
                    
                    # Check team overlap
                    overlap = len(set(team) & set(units))
                    if overlap >= 3:
                        skill_counts[captain_skill] += overlap * 0.5
        
        if skill_counts:
            return max(skill_counts.items(), key=lambda x: x[1])[0]
        else:
            return "Erosion"  # Default


def interactive_team_builder(builder, available_units, num_teams=4, team_size=5,
                            seed_units=None, forbidden_pairs=None, required_pairs=None,
                            must_use_units=None, unit_quality_weight=0.8, debug=True):
    """
    Interactive wrapper for building teams with ability to swap units and rerun after removing units.
    
    Returns the teams and tracks which units have been removed from which teams.
    """
    # Track removed units across all iterations
    excluded_units_per_team = [[] for _ in range(num_teams)]
    iteration = 1
    teams = None
    
    while True:
        # Only rebuild if this is first iteration or after a removal
        if teams is None:
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}")
            print(f"{'='*60}")
            
            # Build teams with current exclusions
            teams = builder.build_multiple_teams(
                available_units=available_units,
                num_teams=num_teams,
                team_size=team_size,
                seed_units_per_team=seed_units,
                forbidden_pairs=forbidden_pairs,
                required_pairs=required_pairs,
                must_use_units=must_use_units,
                unit_quality_weight=unit_quality_weight,
                debug=debug,
                excluded_units_per_team=excluded_units_per_team
            )
        
        # Display results
        print("\n" + "=" * 60)
        print("CURRENT TEAMS")
        print("=" * 60)
        for i, team in enumerate(teams, 1):
            captain_skill = builder.suggest_captain_skill(team)
            print(f"\nTeam {i} - Captain Skill: {captain_skill}")
            for j, unit in enumerate(team, 1):
                print(f"  {j}. {unit}")
        
        # Ask user what they want to do
        print("\n" + "=" * 60)
        print("OPTIONS:")
        print("  1. Swap two units between teams")
        print("  2. Remove and/or add unit(s) then regenerate")
        print("  3. Accept final teams and exit")
        print("=" * 60)
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '3':
            print("\nFinal teams accepted!")
            return teams, excluded_units_per_team
        
        elif choice == '1':
            # SWAP UNITS
            print("\n--- SWAP UNITS ---")
            
            # Get first team and unit
            while True:
                try:
                    team1_num = int(input(f"\nFirst team number (1-{num_teams}): "))
                    if 1 <= team1_num <= num_teams:
                        break
                    print(f"Please enter a number between 1 and {num_teams}")
                except ValueError:
                    print("Please enter a valid number")
            
            team1_idx = team1_num - 1
            print(f"\nTeam {team1_num} units:")
            for j, unit in enumerate(teams[team1_idx], 1):
                print(f"  {j}. {unit}")
            
            while True:
                try:
                    unit1_num = int(input(f"\nFirst unit to swap (1-{len(teams[team1_idx])}): "))
                    if 1 <= unit1_num <= len(teams[team1_idx]):
                        break
                    print(f"Please enter a number between 1 and {len(teams[team1_idx])}")
                except ValueError:
                    print("Please enter a valid number")
            
            unit1_idx = unit1_num - 1
            unit1 = teams[team1_idx][unit1_idx]
            
            # Get second team and unit
            while True:
                try:
                    team2_num = int(input(f"\nSecond team number (1-{num_teams}): "))
                    if 1 <= team2_num <= num_teams:
                        break
                    print(f"Please enter a number between 1 and {num_teams}")
                except ValueError:
                    print("Please enter a valid number")
            
            team2_idx = team2_num - 1
            print(f"\nTeam {team2_num} units:")
            for j, unit in enumerate(teams[team2_idx], 1):
                print(f"  {j}. {unit}")
            
            while True:
                try:
                    unit2_num = int(input(f"\nSecond unit to swap (1-{len(teams[team2_idx])}): "))
                    if 1 <= unit2_num <= len(teams[team2_idx]):
                        break
                    print(f"Please enter a number between 1 and {len(teams[team2_idx])}")
                except ValueError:
                    print("Please enter a valid number")
            
            unit2_idx = unit2_num - 1
            unit2 = teams[team2_idx][unit2_idx]
            
            # Perform the swap
            teams[team1_idx][unit1_idx] = unit2
            teams[team2_idx][unit2_idx] = unit1
            
            print(f"\n✓ Swapped '{unit1}' (Team {team1_num}) with '{unit2}' (Team {team2_num})")
            
            # Update seed units to reflect the swap
            if seed_units is None:
                seed_units = [[] for _ in range(num_teams)]
            
            # Update both teams' seeds to current state
            seed_units[team1_idx] = list(teams[team1_idx])
            seed_units[team2_idx] = list(teams[team2_idx])
            
            # Don't set teams to None - keep showing current state
            continue
        
        elif choice == '2':
            # REMOVE AND REGENERATE
            print("\n--- REMOVE UNITS AND REGENERATE ---")
            
            units_to_remove = []  # List of (team_idx, unit_name) tuples
            
            # Allow multiple removals
            while True:
                # Get team number to remove from
                while True:
                    try:
                        team_num_input = input(f"\nWhich team to remove from? (1-{num_teams}, or 'done' to finish removing): ").strip()
                        if team_num_input.lower() == 'done':
                            team_num = None
                            break
                        team_num = int(team_num_input)
                        if 1 <= team_num <= num_teams:
                            break
                        print(f"Please enter a number between 1 and {num_teams}")
                    except ValueError:
                        print("Please enter a valid number or 'done'")
                
                if team_num is None:
                    break
                
                team_idx = team_num - 1
                
                # Display team units (accounting for already-marked removals)
                current_team = [u for u in teams[team_idx] if (team_idx, u) not in units_to_remove]
                
                if not current_team:
                    print(f"\nTeam {team_num} has no units left to remove!")
                    continue
                
                print(f"\nTeam {team_num} units:")
                for j, unit in enumerate(current_team, 1):
                    print(f"  {j}. {unit}")
                
                # Get unit to remove
                while True:
                    try:
                        unit_num_input = input(f"\nWhich unit to remove? (1-{len(current_team)}, or 'back' to choose different team): ").strip()
                        if unit_num_input.lower() == 'back':
                            unit_num = None
                            break
                        unit_num = int(unit_num_input)
                        if 1 <= unit_num <= len(current_team):
                            break
                        print(f"Please enter a number between 1 and {len(current_team)}")
                    except ValueError:
                        print("Please enter a valid number or 'back'")
                
                if unit_num is None:
                    continue
                
                removed_unit = current_team[unit_num - 1]
                units_to_remove.append((team_idx, removed_unit))
                
                print(f"✓ Marked '{removed_unit}' for removal from Team {team_num}")
                
                # Ask if they want to remove more
                more = input("\nRemove another unit? (yes/no): ").strip().lower()
                if more not in ['yes', 'y']:
                    break
            
            # Check if any units were marked for removal
            if not units_to_remove:
                print("\nNo units marked for removal. Returning to menu.")
                continue
            
            # Process all removals
            print("\n--- PROCESSING REMOVALS ---")
            for team_idx, removed_unit in units_to_remove:
                # Add removed unit to exclusion list for this team
                excluded_units_per_team[team_idx].append(removed_unit)
                
                # Update seed units: remaining units in the team become new seeds
                if seed_units is None:
                    seed_units = [[] for _ in range(num_teams)]
                
                # Calculate remaining units after this removal
                removed_from_team = [u for (t_idx, u) in units_to_remove if t_idx == team_idx]
                seed_units[team_idx] = [u for u in teams[team_idx] if u not in removed_from_team]
                
                print(f"✓ Removed '{removed_unit}' from Team {team_idx + 1}")
                print(f"  → Excluded from Team {team_idx + 1} in regeneration")
            
            # Summary
            print("\n--- REMOVAL SUMMARY ---")
            for team_idx in range(num_teams):
                removed_from_team = [u for (t_idx, u) in units_to_remove if t_idx == team_idx]
                if removed_from_team:
                    print(f"Team {team_idx + 1}:")
                    print(f"  Removed: {', '.join(removed_from_team)}")
                    print(f"  Seeds (kept): {', '.join(seed_units[team_idx]) if seed_units[team_idx] else 'None'}")
            
            # Ask if user wants to manually add units before regenerating
            print("\n" + "=" * 60)
            add_units = input("\nWould you like to manually add units before regenerating? (yes/no): ").strip().lower()
            
            if add_units in ['yes', 'y']:
                print("\n--- MANUALLY ADD UNITS ---")
                
                # Get list of all units not currently on any team (accounting for removals)
                current_placed_units = []
                for team_idx in range(num_teams):
                    removed_from_team = [u for (t_idx, u) in units_to_remove if t_idx == team_idx]
                    current_placed_units.extend([u for u in teams[team_idx] if u not in removed_from_team])
                
                available_to_add = [u for u in available_units if u not in current_placed_units]
                
                if not available_to_add:
                    print("\nNo units available to add - all units are already placed on teams!")
                else:
                    units_to_add = []  # List of (team_idx, unit_name) tuples
                    
                    # Allow multiple additions
                    while True:
                        print(f"\n{len(available_to_add)} units available to add")
                        
                        # Get team number to add to
                        while True:
                            try:
                                team_num_input = input(f"\nWhich team to add to? (1-{num_teams}, or 'done' to finish adding): ").strip()
                                if team_num_input.lower() == 'done':
                                    team_num = None
                                    break
                                team_num = int(team_num_input)
                                if 1 <= team_num <= num_teams:
                                    break
                                print(f"Please enter a number between 1 and {num_teams}")
                            except ValueError:
                                print("Please enter a valid number or 'done'")
                        
                        if team_num is None:
                            break
                        
                        team_idx = team_num - 1
                        
                        # Calculate current team size after removals and planned additions
                        removed_from_team = [u for (t_idx, u) in units_to_remove if t_idx == team_idx]
                        planned_additions_to_team = sum(1 for (t_idx, u) in units_to_add if t_idx == team_idx)
                        current_team_size = len(teams[team_idx]) - len(removed_from_team) + planned_additions_to_team
                        
                        if current_team_size >= team_size:
                            print(f"\nTeam {team_num} will be at maximum size ({team_size}) after removals!")
                            continue
                        
                        # Show current team state
                        remaining_units = [u for u in teams[team_idx] if u not in removed_from_team]
                        print(f"\nTeam {team_num} after removals ({len(remaining_units)} + {planned_additions_to_team} planned):")
                        for j, unit in enumerate(remaining_units, 1):
                            print(f"  {j}. {unit}")
                        if planned_additions_to_team > 0:
                            print("  Planned additions:")
                            for (t_idx, u) in units_to_add:
                                if t_idx == team_idx:
                                    print(f"    + {u}")
                        
                        # Show available units (excluding already planned additions)
                        planned_unit_names = [u for (t_idx, u) in units_to_add]
                        available_now = [u for u in available_to_add if u not in planned_unit_names]
                        
                        if not available_now:
                            print("\nNo more units available to add!")
                            break
                        
                        print(f"\nAvailable units ({len(available_now)}):")
                        for j, unit in enumerate(available_now, 1):
                            print(f"  {j}. {unit}")
                        
                        # Get unit to add
                        while True:
                            try:
                                unit_input = input(f"\nWhich unit to add? (1-{len(available_now)}, or 'back' to choose different team): ").strip()
                                if unit_input.lower() == 'back':
                                    unit_num = None
                                    break
                                unit_num = int(unit_input)
                                if 1 <= unit_num <= len(available_now):
                                    break
                                print(f"Please enter a number between 1 and {len(available_now)}")
                            except ValueError:
                                print("Please enter a valid number or 'back'")
                        
                        if unit_num is None:
                            continue
                        
                        unit_to_add = available_now[unit_num - 1]
                        units_to_add.append((team_idx, unit_to_add))
                        
                        print(f"✓ Marked '{unit_to_add}' to be added to Team {team_num}")
                        
                        # Ask if they want to add more
                        more = input("\nAdd another unit? (yes/no): ").strip().lower()
                        if more not in ['yes', 'y']:
                            break
                    
                    # Process manual additions by adding them to seed units
                    if units_to_add:
                        print("\n--- PROCESSING MANUAL ADDITIONS ---")
                        for team_idx, unit_to_add in units_to_add:
                            if seed_units[team_idx] is None:
                                seed_units[team_idx] = []
                            seed_units[team_idx].append(unit_to_add)
                            print(f"✓ Will add '{unit_to_add}' to Team {team_idx + 1} as seed")
                        
                        print("\n--- UPDATED SEEDS FOR REGENERATION ---")
                        for team_idx in range(num_teams):
                            if seed_units[team_idx]:
                                print(f"Team {team_idx + 1} seeds: {', '.join(seed_units[team_idx])}")
            
            iteration += 1
            teams = None  # Trigger rebuild on next loop
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


# Making a Brigade from Barracks
if __name__ == "__main__":
    # Initialize with single or multiple datasets
    builder = FEHTeamBuilder(
        csv_file_path=['dataset1.csv'],
        priority_weights=[1.0],
        skip_header_rows=3
    )
    
    # Available units
    available_units = [
        "Sakura Legendary", "Thorr Summer Duo", "Alfador Mythic", "Heimdallr Mythic",
        "Chrom Feroxi", "Celica Summer Harmonic", "Vali Mythic", "Camilla Young",
        "Freyja Halloween Duo", "Byleth (M) Brave", "Lyn Mythic", "Micaiah Emblem",
        "Ryoma Attuned", "Ingrid Teatime Harmonic", "Veronica Teatime", "Marth (Masked) Feroxi",
        "Marth Anniversary Duo", "Heidrun New Year Duo", "Loki Mythic", "Sigurd Emblem",
        "Plumeria Spring Harmonic", "Black Knight Legendary", "Rhea Valentine's Duo", "Tharja Attuned",
        "Eikthyrnir Brave", "Hapi Summer Duo", "Alm Attuned", "Elm Mythic",
        "Hinoka Rearmed", "Hraesvelgr Mythic", "Laeradr Mythic", "Lyn Emblem",
        "Sonia", "Chrom Brave", "Ymir", "Alear (M) Fallen",
        "Baldr Brave", "Corrin (F) Brave", "Dedue Fallen Rearmed", "Gromell",
        "Noire Rearmed"
    ]
    
    # Defining constraints
    forbidden_pairs = []
    required_pairs = []
    seed_units = [[], [], [], []]  # Team 1, 2, 3, 4
    must_use_units = []
    
    # Run interactive team builder
    final_teams, exclusions = interactive_team_builder(
        builder=builder,
        available_units=available_units,
        num_teams=4,
        team_size=5,
        seed_units=seed_units,
        forbidden_pairs=forbidden_pairs,
        required_pairs=required_pairs,
        must_use_units=must_use_units,
        unit_quality_weight=0.8,
        debug=True
    )
    
    print("\n" + "=" * 60)
    print("SESSION COMPLETE")
    print("=" * 60)
    print("\nFinal exclusions per team:")
    for i, excluded in enumerate(exclusions, 1):
        if excluded:
            print(f"  Team {i}: {', '.join(excluded)}")
        else:
            print(f"  Team {i}: None")
