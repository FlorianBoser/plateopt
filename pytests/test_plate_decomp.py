import pytest
import pandas as pd
from unittest.mock import MagicMock, call

from plateopt.plate_designer import HTEPlateDesigner

@pytest.fixture
def mock_designer_and_analyzer():
    """
    Creates a mock ReactionScoreAnalyzer and an HTEPlateDesigner instance for testing.
    This fixture simulates an environment with a known set of components and scores.
    """
    # 1. Create a mock ReactionScoreAnalyzer
    mock_analyzer = MagicMock()

    # 2. Configure the mock's properties and methods
    mock_analyzer.condition_types = ['ligand', 'base', 'solvent', 'temperature']

    # Mock the number of available components for each type
    component_map = {
        'ligand': {'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8'},  # 8 ligands
        'base': {'B1', 'B2', 'B3', 'B4'},  # 4 bases
        'solvent': {'S1', 'S2'},  # 2 solvents
        'temperature': {'T1', 'T2', 'T3'} # 3 temperatures
    }
    mock_analyzer.get_available_components.side_effect = lambda comp_type: component_map[comp_type]

    # Mock the "top" components. When asked for the top N, it returns the first N.
    def mock_get_top_components(comp_type, n, metric):
        prefix = comp_type[0].upper()
        return [(f"{prefix}{i}", 1.0) for i in range(1, n + 1)]
    mock_analyzer.get_top_components.side_effect = mock_get_top_components

    # Mock the internal merged data for scoring tests
    mock_analyzer._merged_data = pd.DataFrame({
        'rxn_id': [1, 2, 3, 4, 5, 6],
        'ligand': ['L1', 'L2', 'L1', 'L2', 'L3', 'L8'],
        'base': ['B1', 'B1', 'B2', 'B2', 'B1', 'B4'],
        'solvent': ['S1', 'S1', 'S1', 'S1', 'S2', 'S2'],
        'score': [10, 8, 9, 7, 5, 1] # High scores for L1/L2 + B1/B2 in S1
    })

    # 3. Create the HTEPlateDesigner instance with the mock analyzer
    designer = HTEPlateDesigner(mock_analyzer, plate_sizes=[24])

    return designer, mock_analyzer

### The Tests ###

def test_get_factor_decompositions(mock_designer_and_analyzer):
    """Tests the mathematical factorization of a plate size."""
    designer, _ = mock_designer_and_analyzer
    
    # Test for n=24
    decomps = designer.get_factor_decompositions(24)
    expected = {
        (24, 1), (12, 2), (8, 3), (6, 4), # 2-factors
        (6, 2, 2), (4, 3, 2)             # 3-factors
    }
    # Use set for order-independent comparison
    assert set(decomps) == expected

def test_get_valid_decompositions(mock_designer_and_analyzer):
    """Tests if the class correctly identifies chemically feasible decompositions."""
    designer, _ = mock_designer_and_analyzer
    
    # Based on the fixture: 8 ligands, 4 bases, 2 solvents, 1 temperature
    valid_decomps = designer.get_valid_decompositions()

    expected_decomp_1 = {
        "components": ["ligand", "base"],
        "dimensions": [6, 4],
        "plate_size": 24,
    }
    assert expected_decomp_1 in valid_decomps

    expected_decomp_2 = {
        "components": ["ligand", "base"],
        "dimensions": [8, 3],
        "plate_size": 24,
    }
    assert expected_decomp_2 in valid_decomps

    expected_decomp_3 = {
        "components": ["ligand", "base", "solvent"],
        "dimensions": [4, 3, 2],
        "plate_size": 24,
    }
    assert expected_decomp_3 in valid_decomps

def test_get_plate_components(mock_designer_and_analyzer):
    """Tests if a valid decomposition is correctly filled with top components."""
    designer, _ = mock_designer_and_analyzer

    # Define a template to fill
    decomposition = {
        "components": ["ligand", "base"],
        "dimensions": [3, 2], # Request top 3 ligands and top 2 bases
        "plate_size": 24,
    }

    plate_components = designer.get_plate_components(decomposition, metric="mean")

    # Check the "varied" components
    assert plate_components["ligand"] == ["L1", "L2", "L3"]
    assert plate_components["base"] == ["B1", "B2"]

    # Check the "static" (unused) components, which should be filled with the single best
    assert plate_components["solvent"] == ["S1"]
    assert plate_components["temperature"] == ["T1"]

def test_find_best_plate_designs_integration(mock_designer_and_analyzer):
    """An integration test to ensure the highest-scoring plate is found."""
    designer, _ = mock_designer_and_analyzer
    designer.plate_sizes = [4] # Override to a very small size for a predictable test
    
    # With a 4-well plate, a 2x2 decomposition is possible.
    # Our mock data gives high scores to L1/L2 + B1/B2.
    # So, a 2x2 ligand/base plate should be the winner.
    designer.factor_decompositions[4] = designer.get_factor_decompositions(4) # Recalculate for 4
    
    # Redefine component counts for this smaller test case
    designer.analyzer.get_available_components.side_effect = lambda comp_type: {
        'ligand': {'L1', 'L2'}, 'base': {'B1', 'B2'}, 'solvent': {'S1'}
    }[comp_type]
    designer.analyzer.condition_types = ['ligand', 'base', 'solvent']


    best_designs = designer.find_best_plate_designs(metric="mean", n_designs=1)
    
    assert len(best_designs) == 1
    best_design = best_designs[0]

    # The winning design should be a 2x2 ligand vs. base plate
    assert best_design["decomposition"]["components"] == ["ligand", "base"]
    assert best_design["decomposition"]["dimensions"] == [2, 2]
    
    # The score should be the sum of the 4 relevant reactions in our mock data
    # L1/B1 (10), L2/B1 (8), L1/B2 (9), L2/B2 (7) -> Total = 34
    assert best_design["total_score"] == 34