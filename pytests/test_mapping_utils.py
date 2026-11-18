import pytest
import pandas as pd
import numpy as np
from plateopt.mapping_utils import ReactionScoreAnalyzer


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    # Create conditions DataFrame
    conditions_data = {
        "rxn_id": [1, 2, 3, 4, 5, 6],
        "temperature": ["50C", "80C", "50C", "80C", "50C", "80C"],
        "catalyst": ["Pd", "Pd", "Ni", "Ni", "Pd", "Ni"],
    }
    conditions_df = pd.DataFrame(conditions_data)

    # Create scores DataFrame
    scores_data = {
        "rxn_id": [1, 2, 3, 4, 5, 6],
        "score": [0.75, 0.82, 0.70, 0.85, 0.72, 0.88],
    }
    scores_df = pd.DataFrame(scores_data)
    print(conditions_df, scores_df)

    return conditions_df, scores_df


def test_reaction_score_analyzer_init(sample_data):
    """Test initialization of ReactionScoreAnalyzer"""
    conditions_df, scores_df = sample_data
    analyzer = ReactionScoreAnalyzer(conditions_df, scores_df)

    # Check if condition types are correctly identified
    assert set(analyzer.condition_types) == {"temperature", "catalyst"}

    # Check if data is properly merged
    assert len(analyzer._merged_data) == 6
    assert set(analyzer._merged_data.columns) == {
        "rxn_id",
        "temperature",
        "catalyst",
        "score",
    }


def test_get_component_scores(sample_data):
    """Test get_component_scores method"""
    conditions_df, scores_df = sample_data
    analyzer = ReactionScoreAnalyzer(conditions_df, scores_df)

    # Test temperature scores
    temp_scores = analyzer.get_component_scores("temperature")

    # Check structure, updated to include new metrics
    expected_metrics = {
        "mean",
        "std",
        "median",
        "count",
        "max",
        "min",
        "ucb",
        "ucb_2",
        "percentile_75",
        "percentile_90",
        "sum_above_75",
        "sum_above_90",
    }

    # Check structure
    assert "50C" in temp_scores
    assert "80C" in temp_scores
    assert set(temp_scores["50C"].keys()) == expected_metrics

    # Check basic calculations for 50C
    assert temp_scores["50C"]["count"] == 3
    assert np.isclose(temp_scores["50C"]["mean"], (0.75 + 0.70 + 0.72) / 3)
    assert np.isclose(temp_scores["50C"]["max"], 0.75)
    assert np.isclose(temp_scores["50C"]["min"], 0.70)

    # Test new metrics for 50C
    scores_50C = [0.75, 0.70, 0.72]
    percentile_75 = np.percentile(scores_50C, 75)
    assert np.isclose(temp_scores["50C"]["percentile_75"], percentile_75)

    # Test basic calculations for 50C
    scores_50C = np.array([0.75, 0.70, 0.72])
    mean_50C = np.mean(scores_50C)
    std_50C = np.std(scores_50C, ddof=1)

    assert np.isclose(
        temp_scores["50C"]["ucb"], mean_50C + std_50C
    )
    # Test catalyst scores
    cat_scores = analyzer.get_component_scores("catalyst")
    assert set(cat_scores.keys()) == {"Pd", "Ni"}


def test_get_available_components(sample_data):
    """Test get_available_components method"""
    conditions_df, scores_df = sample_data
    analyzer = ReactionScoreAnalyzer(conditions_df, scores_df)

    # Test temperature components
    temp_components = analyzer.get_available_components("temperature")
    assert temp_components == {"50C", "80C"}

    # Test catalyst components
    cat_components = analyzer.get_available_components("catalyst")
    assert cat_components == {"Pd", "Ni"}

    print(temp_components, cat_components)

    # Test invalid component type
    with pytest.raises(ValueError):
        analyzer.get_available_components("invalid_type")


def test_get_top_components(sample_data):
    """Test get_top_components method"""
    conditions_df, scores_df = sample_data
    analyzer = ReactionScoreAnalyzer(conditions_df, scores_df)

    # Test top temperatures by mean score
    top_temps = analyzer.get_top_components("temperature", n=2, metric="mean")
    assert len(top_temps) == 2
    # 80C should be first as it has higher scores
    assert top_temps[0][0] == "80C"

    # Test with minimum count requirement
    top_temps_min_count = analyzer.get_top_components(
        "temperature", n=2, metric="mean", min_count=4
    )
    assert len(top_temps_min_count) == 0  # No temperature has 4 or more samples

    # Test with different metrics
    top_temps_max = analyzer.get_top_components("temperature", n=1, metric="max")
    assert top_temps_max[0][0] == "80C"  # 80C has highest max score

    print(top_temps, top_temps_max)


def test_invalid_inputs(sample_data):
    """Test error handling"""
    conditions_df, scores_df = sample_data
    analyzer = ReactionScoreAnalyzer(conditions_df, scores_df)

    # Test invalid component type
    with pytest.raises(ValueError):
        analyzer.get_component_scores("invalid_type")

    # Test invalid metric in get_top_components
    with pytest.raises(KeyError):
        analyzer.get_top_components("temperature", metric="invalid_metric")


def test_empty_data():
    """Test handling of empty DataFrames"""
    empty_conditions = pd.DataFrame(columns=["rxn_id", "temperature", "catalyst"])
    empty_scores = pd.DataFrame(columns=["rxn_id", "score"])

    analyzer = ReactionScoreAnalyzer(empty_conditions, empty_scores)

    # Test that methods handle empty data gracefully
    temp_scores = analyzer.get_component_scores("temperature")
    assert len(temp_scores) == 0

    available_temps = analyzer.get_available_components("temperature")
    assert len(available_temps) == 0

    top_temps = analyzer.get_top_components("temperature")
    assert len(top_temps) == 0


def test_component_ranking_consistency(sample_data):
    """Test that different ranking metrics are calculated correctly"""
    conditions_df, scores_df = sample_data
    analyzer = ReactionScoreAnalyzer(conditions_df, scores_df)

    temp_scores = analyzer.get_component_scores('temperature')

    # Test that UCB is correctly higher than mean
    for temp in temp_scores:
        assert temp_scores[temp]['ucb'] > temp_scores[temp]['mean']

    # Test percentile relationships
    for temp in temp_scores:
        assert temp_scores[temp]['percentile_90'] >= temp_scores[temp]['percentile_75']


def test_get_top_components_with_new_metrics(sample_data):
    """Test get_top_components with different scoring metrics"""
    conditions_df, scores_df = sample_data
    analyzer = ReactionScoreAnalyzer(conditions_df, scores_df)

    # Test different ranking metrics
    metrics = ['mean', 'ucb', 'percentile_75', 'sum_above_75']
    for metric in metrics:
        top_temps = analyzer.get_top_components('temperature', n=2, metric=metric)
        assert len(top_temps) == 2
        # Verify ordering
        assert top_temps[0][1] >= top_temps[1][1]

    # Test invalid metric
    with pytest.raises(KeyError):
        analyzer.get_top_components('temperature', metric='invalid_metric')