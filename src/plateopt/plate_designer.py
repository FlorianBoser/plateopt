"Module to generate reaction templates from computed scores"

from typing import List, Tuple, Dict

import pandas as pd

from plateopt.mapping_utils import ReactionScoreAnalyzer


class HTEPlateDesigner:
    "Class for obtaining plate designs"

    def __init__(
        self,
        score_analyzer: ReactionScoreAnalyzer,
        plate_sizes: List[int] = [96, 48, 24],
    ):
        self.analyzer = score_analyzer
        self.plate_sizes = plate_sizes
        # Store both 2-factor and 3-factor decompositions
        self.factor_decompositions = {
            size: self.get_factor_decompositions(size) for size in plate_sizes
        }

    def get_factor_decompositions(self, n: int) -> List[Tuple[int, ...]]:
        """
        Get both 2-factor and 3-factor decompositions for number n.

        Args:
            n (int): Plate size

        Returns:
            Tuple of both two and three factor decomposition templates

        Example:
        96 -> [(1,96), (2,48), (3,32), (4,24), (6,16), (8,12),
               (2,2,24), (2,3,16), (2,4,12), (2,6,8), (3,4,8), ...] # excluding cases with 1s
        """
        decompositions = []

        # Get 2-factor decompositions
        for i in range(1, int(n**0.5) + 1):
            # loop from 1 to square root for two factor pairs
            if n % i == 0:
                if i <= n // i:
                    decompositions.append((n // i, i))

        # Get 3-factor decompositions (excluding 1s)
        for i in range(2, int(n**0.5) + 1):
            if n % 2 == 0:
                remainder = n // i
                for j in range(2, int(remainder**0.5) + 1):
                    if remainder % j == 0:
                        k = remainder // j
                        # Only append if numbers are in ascending order
                        if i <= j <= k:
                            decompositions.append((k, j, i))

        return sorted(decompositions)

    def get_valid_decompositions(self) -> List[Dict]:
        """Find valid factor decompositions based on component availability"""
        decompositions = []

        # Get available components counts
        component_counts = {
            comp_type: len(self.analyzer.get_available_components(comp_type))
            for comp_type in self.analyzer.condition_types
        }

        # Constrain to 1 temperature per plate
        if 'temperature' in component_counts:
            component_counts['temperature'] = 1

        # For each plate size
        for plate_size in self.plate_sizes:
            # For each possible decomposition
            for decomp in self.factor_decompositions[plate_size]:
                # For 2-factor decompositions
                if len(decomp) == 2:
                    self._add_two_factor_decompositions(
                        decomp, component_counts, plate_size, decompositions
                    )
                # For 3-factor decompositions
                elif len(decomp) == 3:
                    self._add_three_factor_decompositions(
                        decomp, component_counts, plate_size, decompositions
                    )

        return decompositions

    def _add_two_factor_decompositions(
        self, decomp, component_counts, plate_size, decompositions
    ):
        """Helper to add valid 2-factor decompositions"""
        rows, cols = decomp

        # loops over every possible component combination
        for comp1 in component_counts:
            for comp2 in component_counts:
                # checks that components are not equal
                if comp1 != comp2:
                    count1 = component_counts[comp1]
                    count2 = component_counts[comp2]

                    # reverse order will be checked by double for loop
                    if count1 >= rows and count2 >= cols:
                        decompositions.append(
                            {
                                "components": [comp1, comp2],
                                "dimensions": [rows, cols],
                                "plate_size": plate_size,
                            }
                        )

    def _add_three_factor_decompositions(
        self, decomp, component_counts, plate_size, decompositions
    ):
        """
        Helper to add valid 3-factor decompositions.

        Loops over every component, checks the length, and
        adds the component combination if valid.
        """
        dim1, dim2, dim3 = decomp

        # Try all possible component combinations
        for comp1 in component_counts:
            for comp2 in component_counts:
                for comp3 in component_counts:
                    # checks the set of components are all different
                    if len({comp1, comp2, comp3}) == 3:
                        count1 = component_counts[comp1]
                        count2 = component_counts[comp2]
                        count3 = component_counts[comp3]

                        # Check if we have enough unique components, for loop checks all orderings already
                        if count1 >= dim1 and count2 >= dim2 and count3 >= dim3:
                            decompositions.append(
                                {
                                    "components": [comp1, comp2, comp3],
                                    "dimensions": [dim1, dim2, dim3],
                                    "plate_size": plate_size,
                                }
                            )

                        # Could add other permutations of dimensions if needed

    def get_plate_components(self, decomposition: Dict, metric: str = "mean"):
        """
        Generate HTE plate layout based on filling in decompositions with the
        top scored components.

        Args:
            decomposition: Dictionary containing:
                - components: List of component types
                - dimensions: List of corresponding dimension sizes
            metric: Metric used for aggregating average component scores

        Returns:
        """
        # components and dimensions of main decomposition
        components = decomposition["components"]
        dimensions = decomposition["dimensions"]

        # Get components for main decomposition
        plate_components = {}
        for comp_type, dim_size in zip(components, dimensions):
            top_comps = [
                comp
                for comp, _ in self.analyzer.get_top_components(
                    comp_type, n=dim_size, metric=metric
                )
            ]
            plate_components[comp_type] = top_comps

        # Fill in remaining components with their top performer
        unused_components = set(self.analyzer.condition_types) - set(components)
        for comp_type in unused_components:
            # Get single best component
            top_comp = [
                comp
                for comp, _ in self.analyzer.get_top_components(
                    comp_type, n=1, metric=metric
                )
            ]
            plate_components[comp_type] = top_comp

        return plate_components

    def score_plate_design(self, decomposition, metric):
        """
        Evaluate a plate design by getting total score of included reactions.

        Args:
            plate_components (Dict): Dictionary containing components for each condition.

        Returns:
            float: Total score of all reactions in the plate design
        """
        # Create mask for filtering reactions
        plate_components = self.get_plate_components(decomposition, metric=metric)
        mask = pd.Series(True, index=self.analyzer._merged_data.index)

        # Filter for each condition type
        for condition, components in plate_components.items():
            mask &= self.analyzer._merged_data[condition].isin(components)

        # Get total score
        total_score = self.analyzer._merged_data[mask]["score"].sum()

        return total_score

    def find_best_plate_designs(
        self, metric: str = "mean", n_designs: int = 5
    ) -> List[Dict]:
        """
        Get the top n plate designs based on total reaction scores.

        Args:
            metric: Metric used for component scoring ('mean', 'median', etc.)
            n_designs: Number of top designs to return

        Returns:
            List of dictionaries containing:
                - decomposition: Original decomposition dict
                - plate_components: Dict of components selected for each condition
                - total_score: Sum of scores for reactions in design
                - plate_size: Size of the plate design
        """
        decompositions = self.get_valid_decompositions()
        scored_designs = []

        # Score each decomposition
        for decomp in decompositions:
            plate_components = self.get_plate_components(decomp, metric=metric)

            # create design info
            design = {
                "decomposition": decomp,
                "plate_components": plate_components,
                "total_score": self.score_plate_design(decomp, metric=metric),
                "stock_solns": self.calculate_stock_solutions(plate_components),
            }
            scored_designs.append(design)

        # Sort by total score
        return sorted(scored_designs, key=lambda x: x["total_score"], reverse=True)[
            :n_designs
        ]

    def get_plate_design_rxn_ids(self, decomposition, metric="mean"):
        """
        Get the rxn_ids of the reactions included in a plate design.

        Args:
            decomposition (Dict): Dictionary containing the plate components.
            metric (str): The metric to use for evaluating the plate design.

        Returns:
            List[str]: List of rxn_id values for the reactions in the plate design.
        """
        plate_components = self.get_plate_components(decomposition, metric=metric)
        mask = pd.Series(True, index=self.analyzer._merged_data.index)

        # iteratively apply mask to dataframe
        for condition, components in plate_components.items():
            current_mask = self.analyzer._merged_data[condition].isin(components)
            mask &= current_mask

        # Combinations may be missing due to removal of values in emulator.
        final_combinations = self.analyzer._merged_data[mask][
            list(plate_components.keys())
        ]
        all_possible = pd.MultiIndex.from_product(
            [plate_components[col] for col in final_combinations.columns]
        )
        missing = set(all_possible) - set(map(tuple, final_combinations.values))

        if missing:
            print(f"Missing combinations: {missing}")

        rxn_ids = self.analyzer._merged_data[mask]["rxn_id"].tolist()
        return rxn_ids

    def calculate_stock_solutions(self, plate_components) -> int:
        """
        Calculate number of stock solutions needed for a plate design.
        Formula: (2 + sum of non-solvent components) * (number of solvents)
        Neglects continuous variables such as temperature

        Args:
            plate_components: Dictionary mapping condition types to their components
                e.g., {'ligand': ['L1', 'L2'], 'solvent': ['S1'], 'catalyst': ['C1']}

        Returns:
            int: Number of stock solutions needed
        """
        # Count number of solvents
        n_solvents = len(plate_components.get("solvent", []))

        # Sum all non-solvent components (plus 2 for substrates)
        n_other_components = 2  # Start with 2 for substrates
        for comp_type, components in plate_components.items():
            if comp_type not in ["solvent", "temperature"]:
                n_other_components += len(components)

        # Calculate total stock solutions
        n_stocks = n_other_components * n_solvents

        return n_stocks


def calculate_stock_solutions_from_df(
        df: pd.DataFrame,
        excluded_cols: List[str] = ['solvent', 'temperature']
) -> int:
    """
    Calculate number of stock solutions needed from a DataFrame of
    reaction conditions. Each column contains a component.

    (2 + number of unique non-solvent components) * (number of unique solvents)

    Args:
        df: DataFrame where columns are reaction components and rows are combinations
        excluded_cols: Columns to exclude from component count (default: solvent and temperature)

    Returns:
        int: Number of stock solutions needed
    """
    # Get number of unique solvents if solvent column exists
    n_solvents = len(df['solvent'].unique()) if 'solvent' in df.columns else 1

    # Get number of other unique components (excluding specified columns)
    n_other_components = 2  # Start with 2 for substrates
    for col in df.columns:
        if col not in excluded_cols:
            n_other_components += len(df[col].unique())

    # Calculate total stock solutions
    n_stocks = n_other_components * n_solvents

    return n_stocks
