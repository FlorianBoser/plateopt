"Module for testing plate designer"

import pytest

import pandas as pd
import numpy as np

from plateopt.plate_designer import (
    HTEPlateDesigner,
    calculate_stock_solutions_from_df,
)
from plateopt.mapping_utils import ReactionScoreAnalyzer


@pytest.fixture
def sample_data():
    """Create sample data with known scores to test component selection"""
    # Create basic reaction components
    ligands = [f"L{i}" for i in range(1, 13)]  # 12 ligands
    solvents = [f"S{i}" for i in range(1, 9)]  # 8 solvents
    catalysts = [f"C{i}" for i in range(1, 5)]  # 4 catalysts
    bases = [f"B{i}" for i in range(1, 4)]  # 3 bases

    # Create combinations with controlled scores
    all_combinations = []
    rxn_id = 1

    for l in ligands:
        for s in solvents:
            for c in catalysts:
                for b in bases:
                    # Assign scores in a predictable way:
                    # Higher index = better score
                    score = (
                        float(l[1:]) / 12  # L1=0.08, L12=1.0
                        + float(s[1:]) / 8  # S1=0.125, S8=1.0
                        + float(c[1:]) / 4  # C1=0.25, C4=1.0
                        + float(b[1:]) / 3  # B1=0.33, B3=1.0
                    )

                    all_combinations.append(
                        {
                            "rxn_id": rxn_id,
                            "ligand": l,
                            "solvent": s,
                            "catalyst": c,
                            "base": b,
                            "score": score,
                        }
                    )
                    rxn_id += 1

    # Create DataFrames
    conditions_df = pd.DataFrame(
        [{k: v for k, v in d.items() if k != "score"} for d in all_combinations]
    )
    scores_df = pd.DataFrame(
        [
            {k: v for k, v in d.items() if k in ["rxn_id", "score"]}
            for d in all_combinations
        ]
    )

    return conditions_df, scores_df


@pytest.fixture
def sample_data_for_scoring():
    """Create sample data where certain combinations have predictably high scores"""
    # Create components
    ligands = [f"L{i}" for i in range(1, 5)]  # 4 ligands
    solvents = [f"S{i}" for i in range(1, 4)]  # 3 solvents
    catalysts = [f"C{i}" for i in range(1, 3)]  # 2 catalysts

    all_combinations = []
    rxn_id = 1

    # Create all combinations with controlled scores
    for l in ligands:
        for s in solvents:
            for c in catalysts:
                # Create predictable scoring pattern:
                # L1+S1+C1 combinations get score 1.0
                # L2+S2+C1 combinations get score 0.8
                # Others get score 0.1
                if l == "L1" and s == "S1" and c == "C1":
                    score = 1.0
                elif l == "L2" and s == "S2" and c == "C1":
                    score = 0.8
                else:
                    score = 0.1

                all_combinations.append(
                    {
                        "rxn_id": rxn_id,
                        "ligand": l,
                        "solvent": s,
                        "catalyst": c,
                        "score": score,
                    }
                )
                rxn_id += 1

    conditions_df = pd.DataFrame(
        [{k: v for k, v in d.items() if k != "score"} for d in all_combinations]
    )
    scores_df = pd.DataFrame(
        [
            {k: v for k, v in d.items() if k in ["rxn_id", "score"]}
            for d in all_combinations
        ]
    )

    return conditions_df, scores_df


@pytest.fixture
def plate_designer(sample_data):
    """Create HTEPlateDesigner instance with sample data"""
    conditions_df, scores_df = sample_data
    analyzer = ReactionScoreAnalyzer(conditions_df, scores_df)
    return HTEPlateDesigner(analyzer, plate_sizes=[24])


@pytest.fixture
def plate_designer_scoring(sample_data_for_scoring):
    """Create HTEPlateDesigner instance with sample data"""
    conditions_df, scores_df = sample_data_for_scoring
    analyzer = ReactionScoreAnalyzer(conditions_df, scores_df)
    return HTEPlateDesigner(analyzer, plate_sizes=[24])


@pytest.fixture
def sample_data_rxn_ids():
    """Create sample merged data and plate components"""
    # Create merged data with some missing combinations
    merged_data = pd.DataFrame(
        {
            "rxn_id": [1, 2, 3, 4, 5],
            "ligand": ["L1", "L1", "L2", "L2", "L1"],
            "solvent": ["S1", "S2", "S1", "S2", "S1"],
            "catalyst": ["C1", "C1", "C1", "C2", "C2"],
        }
    )

    plate_components = {
        "ligand": ["L1", "L2"],
        "solvent": ["S1", "S2"],
        "catalyst": ["C1"],
    }

    return merged_data, plate_components


def test_get_factor_decompositions_24(plate_designer):
    """Test decompositions for n=24"""
    designer = plate_designer
    decomps = designer.get_factor_decompositions(24)

    # Expected 2-factor decompositions (larger number first)
    two_factors = [(24, 1), (12, 2), (8, 3), (6, 4)]

    # Expected 3-factor decompositions (k,j,i where k >= j >= i)
    three_factors = [(6, 2, 2), (4, 3, 2)]

    # All expected decompositions
    expected = sorted(two_factors + three_factors)

    assert decomps == expected


def test_get_factor_decompositions_48(plate_designer):
    """Test decompositions for n=48"""
    designer = plate_designer
    decomps = designer.get_factor_decompositions(48)

    # Expected 2-factor decompositions
    two_factors = [(48, 1), (24, 2), (16, 3), (12, 4), (8, 6)]

    # Expected 3-factor decompositions
    three_factors = [(12, 2, 2), (8, 3, 2), (6, 4, 2), (4, 4, 3)]

    expected = sorted(two_factors + three_factors)

    assert decomps == expected


def test_get_factor_decompositions_96(plate_designer):
    """Test decompositions for n=96"""
    designer = plate_designer
    decomps = designer.get_factor_decompositions(96)

    # Expected 2-factor decompositions
    two_factors = [(96, 1), (48, 2), (32, 3), (24, 4), (16, 6), (12, 8)]

    # Expected 3-factor decompositions
    three_factors = [
        (24, 2, 2),
        (16, 3, 2),
        (12, 4, 2),
        (8, 6, 2),
        (8, 4, 3),
        (6, 4, 4),
    ]

    expected = sorted(two_factors + three_factors)

    assert decomps == expected


def test_get_factor_decompositions_properties(plate_designer):
    """Test general properties of decompositions"""
    designer = plate_designer

    for n in [24, 48, 96]:
        decomps = designer.get_factor_decompositions(n)

        # Test that all decompositions multiply to n
        for decomp in decomps:
            product = 1
            for factor in decomp:
                product *= factor
            assert product == n

        # Test ordering properties
        for decomp in decomps:
            if len(decomp) == 2:
                # First number should be larger in 2-factor decomps
                assert decomp[0] >= decomp[1]
            else:
                # Numbers should be in descending order in 3-factor decomps
                assert decomp[0] >= decomp[1] >= decomp[2]


def test_component_counts(plate_designer):
    """Test that component counts are correct"""
    component_counts = {
        comp_type: len(plate_designer.analyzer.get_available_components(comp_type))
        for comp_type in plate_designer.analyzer.condition_types
    }

    # Verify we have enough components for plate designs
    assert component_counts == {
        "ligand": 12,  # Enough for largest dimension
        "solvent": 8,  # Enough for 8-dimension
        "catalyst": 4,  # Enough for smaller dimensions
        "base": 3,  # Enough for smaller dimensions
    }


def test_two_factor_decompositions(plate_designer):
    """Test valid 2-factor decompositions"""

    # get all two factor decompositions
    decomps = [
        d
        for d in plate_designer.get_valid_decompositions()
        if len(d["components"]) == 2
    ]

    for decomp in decomps:
        # Get components and their counts
        comp1, comp2 = decomp["components"]
        dim1, dim2 = decomp["dimensions"]

        # Verify components are different
        assert comp1 != comp2

        # Verify we have enough unique values for each component
        count1 = len(plate_designer.analyzer.get_available_components(comp1))
        count2 = len(plate_designer.analyzer.get_available_components(comp2))
        assert count1 >= dim1 and count2 >= dim2


def test_three_factor_decompositions(plate_designer):
    """Test valid 3-factor decompositions"""
    decomps = [
        d
        for d in plate_designer.get_valid_decompositions()
        if len(d["components"]) == 3
    ]

    for decomp in decomps:
        # Get components and their counts
        comp1, comp2, comp3 = decomp["components"]
        dim1, dim2, dim3 = decomp["dimensions"]

        # Verify all components are different
        assert len({comp1, comp2, comp3}) == 3

        # Verify we have enough unique values
        count1 = len(plate_designer.analyzer.get_available_components(comp1))
        count2 = len(plate_designer.analyzer.get_available_components(comp2))
        count3 = len(plate_designer.analyzer.get_available_components(comp3))
        assert count1 >= dim1 and count2 >= dim2 and count3 >= dim3


def test_invalid_combinations_excluded(plate_designer):
    """Test that invalid combinations are excluded"""
    decomps = plate_designer.get_valid_decompositions()

    for decomp in decomps:
        # No component should be used twice
        assert len(set(decomp["components"])) == len(decomp["components"])

        # No dimension should exceed available components
        for comp, dim in zip(decomp["components"], decomp["dimensions"]):
            count = len(plate_designer.analyzer.get_available_components(comp))
            assert count >= dim


def test_get_plate_components(plate_designer):
    """Test plate component selection with known scores"""
    # Test with a 2-factor decomposition
    decomp_2f = {
        "components": ["ligand", "solvent"],
        "dimensions": [3, 2],
        "plate_size": 6,  # Small example for clarity
    }

    plate_comps = plate_designer.get_plate_components(decomp_2f)

    # Test main decomposition components
    assert len(plate_comps["ligand"]) == 3
    assert len(plate_comps["solvent"]) == 2

    # Verify we got top scoring components
    assert plate_comps["ligand"] == ["L12", "L11", "L10"]  # Top 3 ligands
    assert plate_comps["solvent"] == ["S8", "S7"]  # Top 2 solvents

    # Test unused components
    assert len(plate_comps["catalyst"]) == 1
    assert len(plate_comps["base"]) == 1
    assert plate_comps["catalyst"] == ["C4"]  # Best catalyst
    assert plate_comps["base"] == ["B3"]  # Best base


def test_score_plate_design(plate_designer_scoring):
    """Test plate design scoring with known scores"""
    decomps = plate_designer_scoring.get_valid_decompositions()
    plate_comps = plate_designer_scoring.get_plate_components(decomps[0])

    # should only be one valid decomp here
    score = plate_designer_scoring.score_plate_design(decomps[0], metric="mean")

    expected = 1.0 + 0.8 + 22 * 0.1

    # Calculate expected score based on our known scoring pattern

    # We know our best components will be selected:
    # L1, L2 for ligands and S1, S2 for solvents with C1 (best catalyst)
    # This should give us reactions:
    # L1+S1+C1 (1.0) + L1+S2+C1 (0.1) + L2+S1+C1 (0.1) + L2+S2+C1 (0.8) = 2.0
    assert np.isclose(score, expected)


@pytest.fixture
def sample_data_best_designs():
    """
    Create dataset where specific designs will have predictably high scores.
    Using a 24-well plate example with clear "best" configurations.
    """
    # Create components with controlled availability
    ligands = [f"L{i}" for i in range(1, 7)]  # 6 ligands
    solvents = [f"S{i}" for i in range(1, 5)]  # 4 solvents
    catalysts = [f"C{i}" for i in range(1, 3)]  # 2 catalysts

    all_combinations = []
    rxn_id = 1

    for l in ligands:
        for s in solvents:
            for c in catalysts:
                # Design scoring pattern:
                # Best design should be (6 ligands × 4 solvents) with C1:
                if c == "C1":
                    if int(l[1]) <= 6 and int(s[1]) <= 4:
                        score = 1.0  # High score for 6×4 design
                    else:
                        score = 0.2
                elif c == "C2":
                    if int(l[1]) <= 8 and int(s[1]) <= 3:
                        score = 0.8  # Medium score for 8×3 design
                    else:
                        score = 0.1

                all_combinations.append(
                    {
                        "rxn_id": rxn_id,
                        "ligand": l,
                        "solvent": s,
                        "catalyst": c,
                        "score": score,
                    }
                )
                rxn_id += 1

    conditions_df = pd.DataFrame(
        [{k: v for k, v in d.items() if k != "score"} for d in all_combinations]
    )
    scores_df = pd.DataFrame(
        [
            {k: v for k, v in d.items() if k in ["rxn_id", "score"]}
            for d in all_combinations
        ]
    )

    return conditions_df, scores_df


@pytest.fixture
def plate_designer_best_designs(sample_data_best_designs):
    """Create HTEPlateDesigner instance with sample data"""
    conditions_df, scores_df = sample_data_best_designs
    analyzer = ReactionScoreAnalyzer(conditions_df, scores_df)
    return HTEPlateDesigner(analyzer, plate_sizes=[24])


def test_find_best_plate_designs(plate_designer_best_designs):
    """Test finding best plate designs with known optimal configurations"""
    best_designs = plate_designer_best_designs.find_best_plate_designs(n_designs=3)

    # Test 1: Basic structure
    assert len(best_designs) == 3  # Should return 3 designs
    for design in best_designs:
        assert set(design.keys()) == {
            "decomposition",
            "total_score",
            "plate_components",
            "stock_solns",
        }

    # Test 2: Scores should be in descending order
    scores = [d["total_score"] for d in best_designs]
    assert scores == sorted(scores, reverse=True)

    # Test 3: Best design should be 6×4 with catalyst C1
    best_design = best_designs[0]
    best_components = best_design["plate_components"]
    assert len(best_components["ligand"]) == 6
    assert len(best_components["solvent"]) == 4
    assert best_components["catalyst"] == ["C1"]

    # Print designs for inspection
    print("\nBest plate designs:")
    for i, design in enumerate(best_designs, 1):
        print(f"\nDesign {i}:")
        print(f"Components: {design['plate_components']}")
        print(f"Score: {design['total_score']}")


def test_find_best_plate_designs_different_n(plate_designer_best_designs):
    """Test requesting different numbers of designs"""
    # Test getting just one design
    one_design = plate_designer_best_designs.find_best_plate_designs(n_designs=1)
    assert len(one_design) == 1


def test_calculate_stock_solutions_basic(plate_designer):
    """Test basic stock solution calculations with different component combinations"""
    # Create test cases with known expected results
    test_cases = [
        # Case 1: 2 ligands, 2 solvents, 1 catalyst, 1 base
        (
            {
                "ligand": ["L1", "L2"],
                "solvent": ["S1", "S2"],
                "catalyst": ["C1"],
                "base": ["B1"],
            },
            12,
        ),  # (2 + 2 + 1 + 1)*2 = 6*2 = 12
        # Case 2: 3 ligands, 1 solvent, 2 catalysts, 1 base
        (
            {
                "ligand": ["L1", "L2", "L3"],
                "solvent": ["S1"],
                "catalyst": ["C1", "C2"],
                "base": ["B1"],
            },
            8,
        ),  # (2 + 3 + 2 + 1)*1 = 8*1 = 8
        # Case 3: 2 ligands, 3 solvents, 1 catalyst
        (
            {"ligand": ["L1", "L2"], "solvent": ["S1", "S2", "S3"], "catalyst": ["C1"]},
            15,
        ),  # (2 + 2 + 1)*3 = 5*3 = 15
    ]
    for plate_components, expected in test_cases:
        result = plate_designer.calculate_stock_solutions(plate_components)
        assert result == expected, f"Expected {expected} stock solutions, got {result}"


def test_basic_stock_calculation():
    """Test basic stock solution calculation with standard columns"""
    # Create simple test DataFrame
    df = pd.DataFrame(
        {
            "ligand": ["L1", "L1", "L2", "L2"],
            "solvent": ["S1", "S2", "S1", "S2"],
            "catalyst": ["C1", "C1", "C2", "C2"],
        }
    )

    # Should be (2 substrates + 2 ligands + 2 catalysts)*2 solvents
    n_stocks = calculate_stock_solutions_from_df(df)
    assert n_stocks == 6*2 == 12


def test_duplicate_components():
    """Test that duplicates are handled correctly"""
    df = pd.DataFrame(
        {
            "ligand": ["L1", "L1", "L1", "L1"],  # Only 1 unique ligand
            "solvent": ["S1", "S2", "S1", "S2"],  # 2 unique solvents
            "catalyst": ["C1", "C1", "C1", "C1"],  # Only 1 unique catalyst
        }
    )

    # Should be (2 substrates + 1 ligand + 1 catalyst)*2 solvents
    n_stocks = calculate_stock_solutions_from_df(df)
    assert n_stocks == 4*2 == 8


def test_with_temperature():
    """Test that temperature column is correctly excluded"""
    df = pd.DataFrame(
        {
            "ligand": ["L1", "L1", "L2", "L2"],
            "solvent": ["S1", "S2", "S1", "S2"],
            "catalyst": ["C1", "C1", "C2", "C2"],
            "temperature": ["80C", "100C", "80C", "100C"],
        }
    )

    # Temperature should be excluded from count
    n_stocks = calculate_stock_solutions_from_df(df)
    assert n_stocks == 6*2 == 12


def test_find_reaction_ids(sample_data_rxn_ids):
    """Test finding reaction IDs for plate design"""
    merged_data, plate_components = sample_data_rxn_ids

    # Create initial mask
    mask = pd.Series(True, index=merged_data.index)

    # Apply filters
    for condition, components in plate_components.items():
        current_data = merged_data[condition].isin(components)
        mask &= current_data

    # Get filtered combinations
    final_combinations = merged_data[mask][list(plate_components.keys())]

    # Check missing combinations
    all_possible = pd.MultiIndex.from_product(
        [plate_components[col] for col in final_combinations.columns]
    )
    missing = set(all_possible) - set(map(tuple, final_combinations.values))

    # Get reaction IDs
    rxn_ids = merged_data[mask]["rxn_id"].tolist()

    # Tests
    assert len(rxn_ids) == 3  # Should find 3 reactions matching criteria
    assert set(rxn_ids) == {1, 2, 3}  # Specific rxn_ids we expect
    assert missing == {("L2", "S2", "C1")}  # Missing combination we expect


def test_empty_result(sample_data_rxn_ids):
    """Test when no reactions match criteria"""
    merged_data, plate_components = sample_data_rxn_ids

    # Modify plate_components to ensure no matches
    plate_components["catalyst"] = ["C3"]  # Non-existent catalyst

    mask = pd.Series(True, index=merged_data.index)

    for condition, components in plate_components.items():
        current_data = merged_data[condition].isin(components)
        mask &= current_data

    rxn_ids = merged_data[mask]["rxn_id"].tolist()

    assert len(rxn_ids) == 0  # Should find no matching reactions


def test_all_combinations_present(sample_data_rxn_ids):
    """Test when all combinations are present"""
    merged_data, _ = sample_data_rxn_ids

    # Create plate_components where all combinations exist
    plate_components = {"ligand": ["L1"], "solvent": ["S1"], "catalyst": ["C1"]}

    mask = pd.Series(True, index=merged_data.index)

    for condition, components in plate_components.items():
        current_data = merged_data[condition].isin(components)
        mask &= current_data

    final_combinations = merged_data[mask][list(plate_components.keys())]
    all_possible = pd.MultiIndex.from_product(
        [plate_components[col] for col in final_combinations.columns]
    )
    missing = set(all_possible) - set(map(tuple, final_combinations.values))

    assert len(missing) == 0  # Should find no missing combinations


def test_partial_matches():
    """Test with partially matching conditions"""
    # Create data where some conditions match but not all
    merged_data = pd.DataFrame(
        {
            "rxn_id": [1, 2, 3],
            "ligand": ["L1", "L1", "L2"],
            "solvent": ["S1", "S2", "S3"],
            "catalyst": ["C1", "C2", "C3"],
        }
    )

    plate_components = {"ligand": ["L1"], "solvent": ["S1", "S2"], "catalyst": ["C1"]}

    mask = pd.Series(True, index=merged_data.index)

    for condition, components in plate_components.items():
        current_data = merged_data[condition].isin(components)
        mask &= current_data

    rxn_ids = merged_data[mask]["rxn_id"].tolist()

    assert rxn_ids == [1]  # Only rxn_id 1 should match all conditions
