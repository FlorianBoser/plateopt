"Module to lay out and plot HTE plate designs as physical well plates"

from itertools import combinations, product
from math import prod
from string import ascii_uppercase
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyBboxPatch

# standard plate formats: number of wells -> (rows, columns)
PLATE_FORMATS = {6: (2, 3), 12: (3, 4), 24: (4, 6), 48: (6, 8), 96: (8, 12), 384: (16, 24)}

# single-hue sequential ramp (light = low, dark = high)
PLATE_CMAP = LinearSegmentedColormap.from_list(
    "plateopt_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

_SURFACE = "#fcfcfb"
_PLATE_BODY = "#f3f2ee"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_OUTLINE = "#c3c2b7"
_LABEL_SEP = " · "


def plate_layout(
    design: Dict,
    conditions_df: pd.DataFrame,
    row_factors: Optional[List[str]] = None,
    col_factors: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Assign every reaction of a plate design to a well (A1, A2, ...).

    Varied components (more than one value) are split over the plate rows and columns
    so that the grid matches a standard plate format where possible, e.g. an 8 x 6
    design on a 48-well plate becomes 6 rows x 8 columns. Components with a single
    value are fixed for the whole plate.

    Args:
        design (Dict): Either a design returned by
            ``HTEPlateDesigner.find_best_plate_designs`` (uses its ``plate_components``)
            or a plain dictionary mapping each component type to its list of values,
            e.g. ``{'ligand': ['L1', 'L2'], 'base': ['B1'], ...}``. Value order is kept,
            so the top-ranked components appear first (top-left).
        conditions_df (pd.DataFrame): Chemical space conditions with columns
            ``rxn_id`` and one column per component type. Used to look up ``rxn_id``.
        row_factors (List[str], optional): Component types to place along the rows.
        col_factors (List[str], optional): Component types to place along the columns.
            If neither is given, the split is chosen automatically.

    Returns:
        pd.DataFrame: One row per well with columns ``well``, ``row``, ``column``,
        ``row_idx``, ``col_idx``, ``row_label``, ``col_label``, one column per
        component type and ``rxn_id`` (``<NA>`` if the combination is not in
        ``conditions_df``). Fixed components are stored in ``layout.attrs['fixed']``.
    """
    components = design.get("plate_components", design)
    components = {comp: list(values) for comp, values in components.items()}

    varied = {comp: len(values) for comp, values in components.items() if len(values) > 1}
    fixed = {comp: values[0] for comp, values in components.items() if len(values) == 1}

    if row_factors is None and col_factors is None:
        row_factors, col_factors = _split_factors(varied, prod(varied.values()))
    elif row_factors is None:
        row_factors = [comp for comp in varied if comp not in col_factors]
    elif col_factors is None:
        col_factors = [comp for comp in varied if comp not in row_factors]

    if sorted(row_factors + col_factors) != sorted(varied):
        raise ValueError(
            f"row_factors {row_factors} and col_factors {col_factors} must together "
            f"contain each varied component exactly once: {sorted(varied)}"
        )

    row_combos = list(product(*[components[comp] for comp in row_factors]))
    col_combos = list(product(*[components[comp] for comp in col_factors]))

    wells = []
    for row_idx, row_values in enumerate(row_combos):
        for col_idx, col_values in enumerate(col_combos):
            well = {
                "well": f"{_row_letter(row_idx)}{col_idx + 1}",
                "row": _row_letter(row_idx),
                "column": col_idx + 1,
                "row_idx": row_idx,
                "col_idx": col_idx,
                "row_label": _LABEL_SEP.join(map(str, row_values)),
                "col_label": _LABEL_SEP.join(map(str, col_values)),
            }
            well.update(dict(zip(row_factors, row_values)))
            well.update(dict(zip(col_factors, col_values)))
            well.update(fixed)
            wells.append(well)

    layout = pd.DataFrame(wells)

    # look up rxn_id of each well from the chemical space
    component_cols = list(components)
    lookup = conditions_df[["rxn_id"] + component_cols].drop_duplicates(component_cols)
    layout = layout.merge(lookup, on=component_cols, how="left")
    layout["rxn_id"] = layout["rxn_id"].astype("Int64")

    layout.attrs["fixed"] = fixed
    layout.attrs["row_factors"] = list(row_factors)
    layout.attrs["col_factors"] = list(col_factors)
    return layout


def plot_plate(
    layout: pd.DataFrame,
    value_col: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Union[str, LinearSegmentedColormap] = PLATE_CMAP,
    annotate: bool = True,
    fmt: str = "{:.0f}",
    title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
) -> plt.Axes:
    """
    Draw a plate layout as a physical well plate.

    Args:
        layout (pd.DataFrame): Output of ``plate_layout``. Add values to colour the
            wells by as a new column, e.g.
            ``layout['predicted_yield'] = layout['rxn_id'].map(predictions)``.
        value_col (str, optional): Column used to colour the wells. Wells with a
            missing value are drawn as empty rings. If None, all wells are drawn empty.
        ax (plt.Axes, optional): Axes to draw on. A new figure is created if None.
        vmin, vmax (float, optional): Colour scale limits. Default to the data range
            (``vmin`` defaults to 0 for non-negative data). Fix them when comparing plates.
        cmap: Matplotlib colormap. Defaults to a single-hue blue ramp.
        annotate (bool): Write the value inside each well.
        fmt (str): Format string for the annotations.
        title (str, optional): Figure title. Fixed components are shown below it.
        colorbar_label (str, optional): Colour scale label, defaults to ``value_col``.

    Returns:
        plt.Axes: The axes containing the plate.
    """
    n_rows = int(layout["row_idx"].max()) + 1
    n_cols = int(layout["col_idx"].max()) + 1

    if ax is None:
        _, ax = plt.subplots(figsize=(2.5 + 0.62 * n_cols, 1.8 + 0.55 * n_rows))

    values = layout[value_col].astype(float) if value_col is not None else None
    norm = None
    if values is not None and values.notna().any():
        data_min, data_max = values.min(), values.max()
        vmin = (0.0 if data_min >= 0 else data_min) if vmin is None else vmin
        vmax = data_max if vmax is None else vmax
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap

    # plate body
    ax.add_patch(
        FancyBboxPatch(
            (-0.6, -0.6), n_cols + 0.2, n_rows + 0.2,
            boxstyle="round,pad=0,rounding_size=0.35",
            facecolor=_PLATE_BODY, edgecolor=_OUTLINE, linewidth=1.0, zorder=0,
        )
    )

    for i, well in enumerate(layout.itertuples(index=False)):
        x, y = well.col_idx, well.row_idx
        value = values.iloc[i] if values is not None else np.nan

        if norm is None or np.isnan(value):
            # empty / missing: hollow ring, never mistaken for a low value
            ax.add_patch(Circle((x, y), 0.38, facecolor=_SURFACE, edgecolor=_OUTLINE,
                                linewidth=1.0, zorder=1))
            continue

        level = norm(value)
        ax.add_patch(Circle((x, y), 0.4, facecolor=cmap(level), edgecolor=_SURFACE,
                            linewidth=2.0, zorder=1))
        if annotate:
            ax.text(x, y, fmt.format(value), ha="center", va="center", fontsize=8,
                    color="#ffffff" if level >= 0.55 else _INK, zorder=2)

    ax.set_xlim(-0.75, n_cols - 0.25)
    ax.set_ylim(n_rows - 0.25, -0.75)
    ax.set_aspect("equal")

    # rows: well letter + component values
    row_labels = layout.drop_duplicates("row_idx").sort_values("row_idx")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{r.row_label}   {r.row}" for r in row_labels.itertuples()])

    # columns: well number on top, component values below
    col_labels = layout.drop_duplicates("col_idx").sort_values("col_idx")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels["col_label"], rotation=40, ha="right", rotation_mode="anchor")
    top = ax.secondary_xaxis("top")
    top.set_xticks(range(n_cols))
    top.set_xticklabels(col_labels["column"])
    top.tick_params(length=0, colors=_MUTED, labelsize=8)
    top.spines["top"].set_visible(False)

    ax.tick_params(length=0, colors=_INK_SECONDARY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    row_factors = layout.attrs.get("row_factors", [])
    col_factors = layout.attrs.get("col_factors", [])
    if row_factors:
        ax.set_ylabel(_LABEL_SEP.join(row_factors), color=_INK_SECONDARY)
    if col_factors:
        ax.set_xlabel(_LABEL_SEP.join(col_factors), color=_INK_SECONDARY)

    fixed = layout.attrs.get("fixed", {})
    subtitle = ("Fixed: " + _LABEL_SEP.join(f"{k} = {v}" for k, v in fixed.items())) if fixed else ""
    # stack above the plate: well numbers, then fixed components, then title
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 20),
                    textcoords="offset points", fontsize=9, color=_INK_SECONDARY,
                    ha="left", va="bottom")
    if title:
        ax.set_title(title, loc="left", fontsize=12, color=_INK, pad=36 if subtitle else 20)

    if norm is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(colorbar_label or value_col, color=_INK_SECONDARY)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(colors=_INK_SECONDARY, length=0, labelsize=8)

    return ax


def _split_factors(varied: Dict[str, int], n_wells: int) -> Tuple[List[str], List[str]]:
    """
    Choose which varied components go along the rows and which along the columns.

    Prefers a split whose grid matches the standard plate format for ``n_wells``
    (with as few components on the columns as possible). Otherwise puts the
    largest component along the columns and the rest along the rows.
    """
    by_size = sorted(varied, key=lambda comp: varied[comp], reverse=True)
    if not by_size:
        return [], []

    if n_wells in PLATE_FORMATS:
        n_rows, n_cols = PLATE_FORMATS[n_wells]
        for n_col_factors in range(1, len(by_size) + 1):
            for cols in combinations(by_size, n_col_factors):
                rows = [comp for comp in by_size if comp not in cols]
                if prod(varied[c] for c in cols) == n_cols and prod(varied[r] for r in rows) == n_rows:
                    return _outer_first(rows, varied), _outer_first(list(cols), varied)

    return _outer_first(by_size[1:], varied), by_size[:1]


def _outer_first(factors: List[str], varied: Dict[str, int]) -> List[str]:
    "Order factors so the one with the fewest values is the outer (slowest-changing) level."
    return sorted(factors, key=lambda comp: varied[comp])


def _row_letter(idx: int) -> str:
    "Row index -> plate row letter (A, B, ..., Z, AA, AB, ...)."
    letters = ""
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        letters = ascii_uppercase[rem] + letters
    return letters
