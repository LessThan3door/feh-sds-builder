# Original file reference: /mnt/data/FEH SDS Autobuilder.py

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
    
    def build_team(self, available_units, team_size=5, seed_units=None, 
                   forbidden_pairs=None, required_pairs=None, role_constraints=None,
                   team_number=None, debug=False, all_seed_units=None):
        """
        Build a single team based on synergies.
        
        Parameters:
        -----------
        available_units : list
            List of available unit names
        team_size : int
            Number of units per team (default 5)
        seed_units : list, optional
            Units that must be on this team
        forbidden_pairs : list of tuples, optional
            Pairs of units that cannot be on the same team [(unit1, unit2), ...]
        required_pairs : list of tuples, optional
            Pairs of units that must be together [(unit1, unit2), ...]
        role_constraints : dict, optional
            Role requirements {'unit_name': 'role', ...}
        team_number : int, optional
            Team number for debugging (1-indexed)
        debug : bool, optional
            Print debug information
        all_seed_units : dict, optional
            Map of team_number -> list of seed units (for reservation)
        
        Returns:
        --------
        list : Selected team of units
        """
        team = []
        remaining = list(available_units)
        
        if debug and team_number is not None:
            print(f"\n=== Building Team {team_number} ===")
        
        # STEP 1: Add seed units first (MANDATORY)
        if seed_units:
            for unit in seed_units:
                if unit in remaining and len(team) < team_size:
                    team.append(unit)
                    remaining.remove(unit)
                    if debug and team_number is not None:
                        print(f"Added seed unit: {unit}")
        
        if debug and team_number is not None:
            print(f"After seeds: {team}")
            print(f"Required pairs to check: {required_pairs}")
        
        # STEP 2: Add required pairs
        if required_pairs:
            for unit1, unit2 in required_pairs:
                if len(team) >= team_size:
                    break
                    
                # Check if this pair is relevant to this team
                u1_in_team = unit1 in team
                u2_in_team = unit2 in team
                
                # If one is already in team (from seeds), add the other
                if u1_in_team and not u2_in_team and unit2 in remaining:
                    if len(team) < team_size:
                        team.append(unit2)
                        remaining.remove(unit2)
                        if debug and team_number is not None:
                            print(f"Added required pair partner: {unit2} (partner of {unit1})")
                elif u2_in_team and not u1_in_team and unit1 in remaining:
                    if len(team) < team_size:
                        team.append(unit1)
                        remaining.remove(unit1)
                        if debug and team_number is not None:
                            print(f"Added required pair partner: {unit1} (partner of {unit2})")
                # If neither is in team, check if they should be added together
                elif not u1_in_team and not u2_in_team:
                    if unit1 in remaining and unit2 in remaining and len(team) + 2 <= team_size:
                        # Check synergy with current team
                        if team:
                            synergy1 = self.calculate_conditional_synergy(unit1, team)
                            synergy2 = self.calculate_conditional_synergy(unit2, team)
                            avg_synergy = (synergy1 + synergy2) / 2
                            
                            if debug and team_number is not None:
                                print(f"Checking pair ({unit1}, {unit2}): synergies = {synergy1:.4f}, {synergy2:.4f}, avg = {avg_synergy:.4f}")
                            
                            # BOTH units must have reasonable synergy (not just average)
                            # This prevents pairing units with 0 synergy
                            min_synergy = min(synergy1, synergy2)
                            if min_synergy > 0.005 and avg_synergy > 0.008:
                                team.append(unit1)
                                remaining.remove(unit1)
                                team.append(unit2)
                                remaining.remove(unit2)
                                if debug and team_number is not None:
                                    print(f"Added both from pair: {unit1}, {unit2}")
        
        if debug and team_number is not None:
            print(f"After required pairs: {team}")
        
        # STEP 3: Fill remaining slots based on synergy
        while len(team) < team_size and remaining:
            best_unit = None
            best_score = -float('inf')
            
            for unit in remaining:
                # CRITICAL: Skip units that are seeded for OTHER teams
                if all_seed_units and team_number is not None:
                    is_reserved_elsewhere = False
                    for other_team_num, other_seeds in all_seed_units.items():
                        if other_team_num != team_number and other_seeds and unit in other_seeds:
                            is_reserved_elsewhere = True
                            if debug:
                                print(f"Skipping {unit} - reserved for Team {other_team_num}")
                            break
                    
                    if is_reserved_elsewhere:
                        continue
                
                # CRITICAL: Check if this unit has a required pair partner
                # If so, only allow it if the partner can also join
                if required_pairs:
                    has_required_partner = False
                    partner_unit = None
                    
                    for u1, u2 in required_pairs:
                        if u1 == unit:
                            has_required_partner = True
                            partner_unit = u2
                            break
                        elif u2 == unit:
                            has_required_partner = True
                            partner_unit = u1
                            break
                    
                    if has_required_partner and partner_unit not in team:
                        # Check if partner is available and not forbidden
                        partner_available = partner_unit in remaining
                        partner_forbidden = False
                        
                        if partner_available and forbidden_pairs:
                            for current_team_unit in team:
                                for f1, f2 in forbidden_pairs:
                                    if (f1 == partner_unit and f2 == current_team_unit) or \
                                       (f2 == partner_unit and f1 == current_team_unit):
                                        partner_forbidden = True
                                        break
                                if partner_forbidden:
                                    break
                        
                        # Skip this unit if partner can't join
                        if not partner_available or partner_forbidden:
                            if debug and team_number is not None:
                                reason = "not available" if not partner_available else "forbidden"
                                print(f"Skipping {unit} - required partner {partner_unit} is {reason}")
                            continue
                
                # Check forbidden pairs
                if forbidden_pairs:
                    is_forbidden = False
                    for u1, u2 in forbidden_pairs:
                        if (unit == u1 and u2 in team) or (unit == u2 and u1 in team):
                            is_forbidden = True
                            break
                    if is_forbidden:
                        continue
                
                # Calculate synergy with current team
                if team:
                    score = self.calculate_conditional_synergy(unit, team)
                else:
                    # First unit: prefer more common units as captains
                    score = self.unit_counts.get(unit, 0)
                
                if score > best_score:
                    best_score = score
                    best_unit = unit
            
            if best_unit:
                team.append(best_unit)
                remaining.remove(best_unit)
                if debug and team_number is not None:
                    print(f"Added by synergy ({best_score:.4f}): {best_unit}")
            else:
                # No valid unit found, break
                break
        
        return team
    
    def build_multiple_teams(self, available_units, num_teams=4, team_size=5,
                            seed_units_per_team=None, forbidden_pairs=None, 
                            required_pairs=None, required_pairs_per_team=None,
                            required_units_per_team=None, must_use_units=None,
                            role_constraints=None, unit_quality_weight=0.3, 
                            fill_all_slots=True, debug=False):
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
        
        Returns:
        --------
        list : List of teams, each team is a list of units
        """
        teams = [[] for _ in range(num_teams)]
        remaining_units = list(available_units)
        
        seed_units_per_team = seed_units_per_team or [None] * num_teams
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
        
        # STEP 1.5: If no seeds provided, intelligently select captains
        # Select units that are popular but have low synergy with each other (anti-synergy)
        if not any(seed_units_per_team):
            if debug:
                print("\nNo seeds provided - selecting captains based on anti-synergy...")
            
            # Find top units by quality
            top_units = sorted(
                [(u, unit_quality[u]) for u in available_units],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Select first captain (highest quality)
            if top_units and remaining_units:
                first_captain = top_units[0][0]
                teams[0].append(first_captain)
                remaining_units.remove(first_captain)
                if debug:
                    print(f"Team 1: Selected captain {first_captain} (quality: {top_units[0][1]:.3f})")
                
                # Select remaining captains based on low synergy with existing captains
                existing_captains = [first_captain]
                
                for team_idx in range(1, num_teams):
                    best_captain = None
                    best_score = -float('inf')
                    
                    # Consider top 50% of units by quality
                    quality_threshold = 0.25  # Must be at least 25% as popular as top unit
                    candidates = [u for u, q in top_units if q >= quality_threshold and u in remaining_units]
                    
                    if not candidates:
                        candidates = remaining_units  # Fallback to all remaining
                    
                    for unit in candidates:
                        # Calculate average synergy with existing captains
                        avg_synergy = sum(self.calculate_synergy_score(unit, cap) 
                                        for cap in existing_captains) / len(existing_captains)
                        
                        # Score: prefer high quality and LOW synergy with other captains
                        # Negative synergy in score means we want low synergy
                        quality_score = unit_quality[unit]
                        score = quality_score - (avg_synergy * 2.0)  # Penalize high synergy
                        
                        if score > best_score:
                            best_score = score
                            best_captain = unit
                    
                    if best_captain:
                        teams[team_idx].append(best_captain)
                        remaining_units.remove(best_captain)
                        existing_captains.append(best_captain)
                        
                        if debug:
                            avg_syn = sum(self.calculate_synergy_score(best_captain, cap) 
                                        for cap in existing_captains[:-1]) / (len(existing_captains) - 1)
                            print(f"Team {team_idx+1}: Selected captain {best_captain} "
                                  f"(quality: {unit_quality[best_captain]:.3f}, "
                                  f"avg synergy with other captains: {avg_syn:.4f})")
        
        if debug and not any(seed_units_per_team):
            print()
        
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
        
        # Final check: Ensure all must_use_units were placed
        # If not, FORCE placement by swapping with lowest synergy units
        if must_use_units:
            all_placed_units = [u for team in teams for u in team]
            missing_must_use = [u for u in must_use_units if u not in all_placed_units]
            
            if missing_must_use:
                if debug:
                    print("\n--- FORCING placement of remaining must-use units ---")
                
                for unit in missing_must_use:
                    # Check if unit has a required partner
                    required_partner = None
                    if required_pairs:
                        for u1, u2 in required_pairs:
                            if u1 == unit and u2 not in all_placed_units:
                                required_partner = u2
                                break
                            elif u2 == unit and u1 not in all_placed_units:
                                required_partner = u1
                                break
                    
                    # Find the best swap candidate across all teams
                    best_swap = None
                    best_swap_score = float('inf')  # Lower is better (we want low synergy to swap out)
                    
                    for team_idx in range(num_teams):
                        # Skip if team is too small
                        if len(teams[team_idx]) <= 1:  # Keep at least seed
                            continue
                        
                        # Check if unit is reserved for different team
                        if unit in seed_to_team and seed_to_team[unit] != team_idx:
                            continue
                        
                        # Check if there's room for partner if needed
                        slots_needed = 1 if not required_partner else 2
                        if len(teams[team_idx]) < slots_needed:
                            continue
                        
                        # Try each non-seed unit as swap candidate
                        for candidate_unit in teams[team_idx][1:]:  # Skip seed (index 0)
                            # Don't swap out must-use units or seed units
                            if candidate_unit in must_use_units:
                                continue
                            if seed_units_per_team and seed_units_per_team[team_idx] and candidate_unit in seed_units_per_team[team_idx]:
                                continue
                            if required_units_per_team and required_units_per_team[team_idx] and candidate_unit in required_units_per_team[team_idx]:
                                continue
                            
                            # Check if swap would violate forbidden pairs
                            temp_team = [u for u in teams[team_idx] if u != candidate_unit]
                            
                            swap_forbidden = False
                            if forbidden_pairs:
                                for team_unit in temp_team:
                                    for f1, f2 in forbidden_pairs:
                                        if (f1 == unit and f2 == team_unit) or (f2 == unit and f1 == team_unit):
                                            swap_forbidden = True
                                            break
                                    if swap_forbidden:
                                        break
                            
                            if swap_forbidden:
                                continue
                            
                            # Check if partner would be forbidden
                            if required_partner:
                                partner_forbidden = False
                                if forbidden_pairs:
                                    for team_unit in temp_team:
                                        for f1, f2 in forbidden_pairs:
                                            if (f1 == required_partner and f2 == team_unit) or \
                                               (f2 == required_partner and f1 == team_unit):
                                                partner_forbidden = True
                                                break
                                        if partner_forbidden:
                                            break
                                
                                if partner_forbidden:
                                    continue
                            
                            # Calculate swap score
                            # Lower score = better candidate for removal
                            # Prefer swapping units that:
                            # 1. Have low synergy with their current team
                            # 2. Have low synergy with the incoming unit (similar role)
                            
                            candidate_synergy = self.calculate_conditional_synergy(candidate_unit, temp_team)
                            unit_candidate_similarity = self.calculate_synergy_score(unit, candidate_unit)
                            
                            # Score: prioritize low team synergy, with bonus for low similarity
                            # (negative similarity because lower is better)
                            swap_score = candidate_synergy - (unit_candidate_similarity * 0.5)
                            
                            # Apply quality bonus: prefer swapping out lower-quality units
                            candidate_quality = unit_quality.get(candidate_unit, 0)
                            unit_quality_score = unit_quality.get(unit, 0)
                            quality_diff = unit_quality_score - candidate_quality
                            
                            # Add quality difference to swap score (positive = better to swap)
                            swap_score = swap_score - (quality_diff * unit_quality_weight)
                            
                            if swap_score < best_swap_score:
                                best_swap_score = swap_score
                                best_swap = (team_idx, candidate_unit, candidate_synergy, unit_candidate_similarity)
                    
                    # Execute the swap
                    if best_swap:
                        team_idx, swap_out_unit, swap_out_synergy, similarity = best_swap
                        teams[team_idx].remove(swap_out_unit)
                        teams[team_idx].append(unit)
                        
                        if debug:
                            print(f"Team {team_idx+1}: FORCED swap - removed {swap_out_unit} (synergy: {swap_out_synergy:.4f}) "
                                  f"for MUST-USE {unit} (similarity: {similarity:.4f})")
                        
                        # Add required partner if applicable
                        if required_partner:
                            # Find another unit to swap out
                            if len(teams[team_idx]) >= team_size:
                                # Need to make room for partner
                                lowest_synergy_unit = None
                                lowest_synergy = float('inf')
                                
                                for team_unit in teams[team_idx][1:]:  # Skip seed
                                    if team_unit == unit:  # Don't swap the unit we just added
                                        continue
                                    if team_unit in must_use_units:
                                        continue
                                    
                                    temp_team = [u for u in teams[team_idx] if u != team_unit]
                                    synergy = self.calculate_conditional_synergy(team_unit, temp_team)
                                    
                                    if synergy < lowest_synergy:
                                        lowest_synergy = synergy
                                        lowest_synergy_unit = team_unit
                                
                                if lowest_synergy_unit:
                                    teams[team_idx].remove(lowest_synergy_unit)
                                    if debug:
                                        print(f"Team {team_idx+1}: Removed {lowest_synergy_unit} (synergy: {lowest_synergy:.4f}) to make room")
                            
                            if required_partner in remaining_units:
                                teams[team_idx].append(required_partner)
                                remaining_units.remove(required_partner)
                            else:
                                teams[team_idx].append(required_partner)
                            
                            if debug:
                                print(f"Team {team_idx+1}: Added required partner {required_partner}")
                    else:
                        print(f"\nCRITICAL ERROR: Could not find any valid swap for must-use unit: {unit}")
        
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


# Making a Brigade from Barracks
if __name__ == "__main__":
    # Initialize with single or multiple datasets
    # For multiple datasets with priority: first is most important
    # IMPORTANT: Set skip_header_rows if your CSV has headers!
    builder = FEHTeamBuilder(
        csv_file_path=['dataset1.csv'],
        priority_weights=[1.0],  # Primary dataset gets full weight
        skip_header_rows=3  # Skip first 3 rows if they're headers
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
    forbidden_pairs = [
        
    ]
    
    # Global required pairs - will try to put these together on ANY team
    # BUT only if they have synergy with the team
    required_pairs = [
       
    ]
    
    # Seed units for each team - currently set as captains
    seed_units = [
        [],  # Team 1
        [],  # Team 2
        [],  # Team 3
        [],  # Team 4
    ]
    
    
    # Optional: Units that must be used somewhere (any team)
    must_use_units = [
        
    ]
    
    # Build teams with debug output
    teams = builder.build_multiple_teams(
        available_units=available_units,
        num_teams=4,
        team_size=5,
        seed_units_per_team=seed_units,
        forbidden_pairs=forbidden_pairs,
        required_pairs=required_pairs,
        must_use_units=must_use_units,  # Add this parameter
        unit_quality_weight=0.8,
        debug=True  # Enable debug output
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("BUILT TEAMS")
    print("=" * 60)
    for i, team in enumerate(teams, 1):
        captain_skill = builder.suggest_captain_skill(team)
        print(f"\nTeam {i} - Captain Skill: {captain_skill}")
        for j, unit in enumerate(team, 1):
            print(f"  {j}. {unit}")
    

# Wrapper generate_fe_teams
from typing import List, Optional, Set, Dict, Any

def _normalize_banned(banned_units_per_team, num_teams=4):
    if banned_units_per_team is None:
        return [set() for _ in range(num_teams)]
    out = []
    for i in range(num_teams):
        v = banned_units_per_team[i] if i < len(banned_units_per_team) else set()
        if isinstance(v, (list, tuple)):
            out.append(set(v))
        elif isinstance(v, set):
            out.append(v)
        else:
            out.append(set())
    return out

def generate_fe_teams(
    available_units: List[str],
    forbidden_pairs: Optional[List[List[str]]] = None,
    required_pairs: Optional[List[List[str]]] = None,
    seed_units: Optional[List[List[str]]] = None,
    must_use_units: Optional[List[str]] = None,
    csv_paths: Optional[List[str]] = None,
    priority_weights: Optional[List[float]] = None,
    skip_header_rows: int = 3,
    debug: bool = False,
    banned_units_per_team: Optional[List[Set[str]]] = None
) -> List[Dict[str, Any]]:
    forbidden_pairs = forbidden_pairs or []
    required_pairs = required_pairs or []
    seed_units = seed_units or [[], [], [], []]
    must_use_units = must_use_units or []
    banned_units_per_team = _normalize_banned(banned_units_per_team, num_teams=4)

    # Attempt to create builder from original code
    builder = None
    try:
        builder = FEHTeamBuilder(csv_file_path=csv_paths or [], priority_weights=priority_weights or [1.0], skip_header_rows=skip_header_rows)
    except Exception:
        builder = None

    if builder is not None and hasattr(builder, 'build_multiple_teams'):
        try:
            teams = builder.build_multiple_teams(
                available_units=available_units,
                num_teams=4,
                team_size=5,
                seed_units_per_team=seed_units,
                forbidden_pairs=forbidden_pairs,
                required_pairs=required_pairs,
                must_use_units=must_use_units,
                unit_quality_weight=0.8,
                debug=debug,
                banned_units_per_team=banned_units_per_team
            )
        except TypeError:
            teams = builder.build_multiple_teams(
                available_units=available_units,
                num_teams=4,
                team_size=5,
                seed_units_per_team=seed_units,
                forbidden_pairs=forbidden_pairs,
                required_pairs=required_pairs,
                must_use_units=must_use_units,
                unit_quality_weight=0.8,
                debug=debug
            )
    else:
        if 'build_multiple_teams' in globals():
            teams = build_multiple_teams(
                available_units=available_units,
                num_teams=4,
                team_size=5,
                seed_units_per_team=seed_units,
                forbidden_pairs=forbidden_pairs,
                required_pairs=required_pairs,
                must_use_units=must_use_units,
                debug=debug,
                banned_units_per_team=banned_units_per_team
            )
        else:
            raise RuntimeError('No FEHTeamBuilder or build_multiple_teams found in autobuilder.')

    out = []
    for t in teams:
        try:
            captain = builder.suggest_captain_skill(t) if builder is not None and hasattr(builder, 'suggest_captain_skill') else None
        except Exception:
            captain = None
        out.append({'team': list(t), 'captain_skill': captain})
    return out
