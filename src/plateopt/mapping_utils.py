"Module to obtain sorted components from scored reaction conditions"

from typing import Dict, List, Tuple, Set

import pandas as pd


class ReactionScoreAnalyzer:
    "Class for sorting reaction components"

    def __init__(self, conditions_df: pd.DataFrame, scores_df: pd.DataFrame):
        """
        Initialize with reaction conditions and their scores.

        Args:
            conditions_df: DataFrame with columns [rxn_id, ...condition_columns]
            scores_df: DataFrame with columns [rxn_id, score]
        """
        self.conditions = conditions_df
        self.scores = scores_df
        self._merged_data = pd.merge(conditions_df, scores_df, on="rxn_id")

        # Get condition columns e.g. ligand, solvent, etc. (all columns except rxn_id and score)
        self.condition_types = [
            col for col in self._merged_data.columns if col not in ["rxn_id", "score"]
        ]

    def get_component_scores(self, component_type: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for scores grouped by a reaction component.
        E.g. Aggregated scores of all ligands.

        Args:
            component_type: Any column name from the conditions DataFrame

        Returns:
            Dictionary with component values as keys and statistics as values
        """
        if component_type not in self.condition_types:
            raise ValueError(
                f"Invalid component type: {component_type}. "
                f"Available types: {self.condition_types}"
            )

        stats = {}
        # group data by specific parameter and take all scores
        grouped = self._merged_data.groupby(component_type)["score"]

        for component, scores in grouped: # looks through the column entries in component_type
            percentile_75 = scores.quantile(0.75)
            percentile_90 = scores.quantile(0.90)

            stats[component] = {
                "mean": scores.mean(),
                "std": scores.std(),
                "median": scores.median(),
                "count": len(scores),
                "max": scores.max(),
                "min": scores.min(),
                "ucb": scores.mean() + scores.std(),
                "ucb_2": scores.mean() + 2 * scores.std(),
                # percentile based metrics
                "percentile_75": percentile_75,
                "percentile_90": percentile_90,
                "sum_above_75": scores[scores > percentile_75].sum(),
                "sum_above_90": scores[scores > percentile_90].sum(),
            }

        return stats

    def get_available_components(self, component_type: str) -> Set[str]:
        """Get all unique values for a given component type."""
        if component_type not in self.condition_types:
            raise ValueError(
                f"Invalid component type: {component_type}. "
                f"Available types: {self.condition_types}"
            )
        return set(self._merged_data[component_type].unique())

    def get_top_components(
        self, component_type: str, n: int = 5, metric: str = "mean", min_count: int = 1
    ) -> List[Tuple[str, float]]:
        """
        Get the top n components based on their scores.

        Args:
            component_type: Any column name from the conditions DataFrame
            n: Number of top components to return
            metric: Metric to sort by ('mean', 'median', 'max')
            min_count: Minimum number of observations required

        Returns:
            List of (component, score) tuples
        """
        stats = self.get_component_scores(component_type)
        filtered_stats = {
            comp: stat for comp, stat in stats.items() if stat["count"] >= min_count
        }

        sorted_components = sorted(
            filtered_stats.items(), key=lambda x: x[1][metric], reverse=True
        )
        return [(comp, stats[metric]) for comp, stats in sorted_components[:n]]

    def get_sorted_components(
        self, component_type: str, metric: str = "mean", reverse: bool = True
    ) -> List[str]:
        """
        Get components sorted by their score metrics, returns only component names.

        Args:
            component_type: Any column name from the conditions DataFrame
            metric: Metric to sort by ('mean', 'median', 'max')
            reverse: If True, sort in descending order (highest to lowest)

        Returns:
            List of component names sorted by the specified metric
        """
        stats = self.get_component_scores(component_type)
        sorted_components = sorted(
            stats.items(), key=lambda x: x[1][metric], reverse=True
        )
        return [comp for comp, _ in sorted_components]
