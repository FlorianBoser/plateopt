"Test plate layout and plotting in visualisation.py"

from itertools import product

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from plateopt.visualisation import plate_layout, plot_plate


@pytest.fixture
def conditions_df():
    "Full factorial chemical space: 8 precursors x 3 ligands x 2 solvents x 2 bases x 2 temperatures"
    combos = list(product(
        [f"P{i}" for i in range(1, 9)], ["L1", "L2", "L3"], ["S1", "S2"], ["B1", "B2"], ["T1", "T2"]
    ))
    df = pd.DataFrame(combos, columns=["precursor", "ligand", "solvent", "base", "temperature"])
    df.insert(0, "rxn_id", range(len(df)))
    return df


@pytest.fixture
def design_3_factor():
    "48-well design with the structure returned by HTEPlateDesigner.find_best_plate_designs"
    return {
        "decomposition": {"components": ["precursor", "ligand", "solvent"], "dimensions": [8, 3, 2], "plate_size": 48},
        "plate_components": {
            "precursor": [f"P{i}" for i in range(1, 9)],
            "ligand": ["L3", "L1", "L2"],
            "solvent": ["S2", "S1"],
            "base": ["B1"],
            "temperature": ["T2"],
        },
    }


def test_layout_matches_48_well_format(design_3_factor, conditions_df):
    layout = plate_layout(design_3_factor, conditions_df)

    assert len(layout) == 48
    assert layout["row_idx"].nunique() == 6 and layout["col_idx"].nunique() == 8
    assert layout.attrs["col_factors"] == ["precursor"]
    assert sorted(layout.attrs["row_factors"]) == ["ligand", "solvent"]
    assert layout.attrs["fixed"] == {"base": "B1", "temperature": "T2"}
    assert layout["well"].iloc[0] == "A1" and layout["well"].iloc[-1] == "F8"
    assert layout["well"].is_unique


def test_layout_keeps_component_order(design_3_factor, conditions_df):
    "Top-ranked components should be placed first (top-left)"
    layout = plate_layout(design_3_factor, conditions_df)
    a1 = layout.set_index("well").loc["A1"]

    assert a1["precursor"] == "P1"
    assert a1["ligand"] == "L3"
    assert a1["solvent"] == "S2"


def test_layout_rxn_ids_match_conditions(design_3_factor, conditions_df):
    layout = plate_layout(design_3_factor, conditions_df)
    lookup = conditions_df.set_index("rxn_id")

    for row in layout.itertuples():
        expected = lookup.loc[row.rxn_id]
        for comp in ["precursor", "ligand", "solvent", "base", "temperature"]:
            assert expected[comp] == getattr(row, comp)
    assert layout["rxn_id"].is_unique


def test_layout_two_factor_design(conditions_df):
    "Plain components dict, 8 x 3 on a 24-well plate -> 3 rows x 8 columns fallback (no 4 x 6 split)"
    components = {
        "precursor": [f"P{i}" for i in range(1, 9)],
        "ligand": ["L1", "L2", "L3"],
        "solvent": ["S1"], "base": ["B1"], "temperature": ["T1"],
    }
    layout = plate_layout(components, conditions_df)

    assert len(layout) == 24
    assert layout.attrs["col_factors"] == ["precursor"]
    assert layout.attrs["row_factors"] == ["ligand"]


def test_layout_explicit_factors(design_3_factor, conditions_df):
    layout = plate_layout(design_3_factor, conditions_df, row_factors=["precursor"])

    assert layout["row_idx"].nunique() == 8
    assert layout["col_idx"].nunique() == 6

    with pytest.raises(ValueError):
        plate_layout(design_3_factor, conditions_df, row_factors=["precursor"], col_factors=["ligand"])


def test_layout_missing_combinations(design_3_factor, conditions_df):
    "Combinations absent from the chemical space get rxn_id <NA>"
    reduced = conditions_df[~((conditions_df["precursor"] == "P1") & (conditions_df["ligand"] == "L3"))]
    layout = plate_layout(design_3_factor, reduced)

    assert len(layout) == 48
    assert layout["rxn_id"].isna().sum() == 2  # P1 x L3 x {S1, S2}


def test_plot_plate_runs(design_3_factor, conditions_df):
    layout = plate_layout(design_3_factor, conditions_df)
    layout["value"] = np.linspace(0, 70, len(layout))
    layout.loc[0, "value"] = np.nan

    ax = plot_plate(layout, value_col="value", title="Test plate", vmax=100)
    assert len(ax.patches) == 49  # plate body + 48 wells
    well_texts = [t for t in ax.texts if t.get_text().isdigit()]
    assert len(well_texts) == 47  # one annotation per non-missing well

    ax_empty = plot_plate(layout)
    assert not [t for t in ax_empty.texts if t.get_text().isdigit()]
