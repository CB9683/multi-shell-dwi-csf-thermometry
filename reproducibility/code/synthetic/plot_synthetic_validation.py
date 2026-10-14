#!/usr/bin/env python3
"""Create the paper-style validation figure for the synthetic DWI dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib as mpl
# This module writes files and must also work without a GUI event loop.
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd

from reproducibility.code.synthetic.generate_synthetic_bids import (
    ARRHENIUS_A_K,
    ARRHENIUS_B_MM2_S,
    LABELS,
    SUBJECT,
    TEMPERATURES_C,
)
from reproducibility.code.synthetic.run_synthetic_example import (
    PRIMARY_SHELLS,
    load_synthetic_bids,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BIDS_ROOT = REPOSITORY_ROOT / "reproducibility" / "synthetic_bids"
DEFAULT_ANALYSIS_ROOT = REPOSITORY_ROOT / "outputs/synthetic_mean_domain_20261014/rice_mean_moments" / SUBJECT
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "outputs/synthetic_mean_domain_20261014/figure"

REGION_ORDER = ("LV", "SA", "Other")
COLORS = {
    "LV": "#D55E00",
    "SA": "#0072B2",
    "Other": "#CC79A7",
    "excluded": "#B8BDC5",
    "text": "#202124",
    "grid": "#D9DDE3",
}


@dataclass(frozen=True)
class SyntheticFigureData:
    """Images and summaries required by the four figure panels."""

    b0: np.ndarray
    labels: np.ndarray
    temperature_median: np.ndarray
    mad: np.ndarray
    observed_z: np.ndarray
    null_median: np.ndarray
    metadata: dict
    retained: np.ndarray
    voxel_table: pd.DataFrame
    summary: pd.DataFrame
    cluster_overlap: pd.DataFrame
    target_gradient_c: float
    recovered_gradient_c: float


def _single_match(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one file matching {pattern!r} in {directory}; found {len(matches)}"
        )
    return matches[0]


def load_figure_data(
    bids_root: str | Path = DEFAULT_BIDS_ROOT,
    analysis_root: str | Path = DEFAULT_ANALYSIS_ROOT,
    subject: str = SUBJECT,
) -> SyntheticFigureData:
    """Read sealed corrected data and exact classifier decisions, with no fallback."""
    bids_root=Path(bids_root);analysis_root=Path(analysis_root)
    metadata=json.loads((analysis_root/'analysis_metadata.json').read_text())
    if metadata.get('schema_version')!=2 or metadata.get('classifier')!='voxel-specific shared-b0 median-null':
        raise ValueError('Figure requires the adopted mean-domain/shared-baseline analysis')
    if metadata['subject']!=subject or metadata['shells_used']!=list(PRIMARY_SHELLS):
        raise ValueError('Figure subject/shell mismatch')
    for name,digest in metadata['artifact_sha256'].items():
        if hashlib.sha256((analysis_root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Analysis artifact checksum mismatch: '+name)
    for name,digest in metadata['raw_input_sha256'].items():
        if hashlib.sha256((bids_root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Synthetic input checksum mismatch: '+name)
    inputs,labels=load_synthetic_bids(bids_root,subject=subject)
    csf=inputs['csf_mask']
    with np.load(analysis_root/'candidate_state.npz') as saved:
        state={key:saved[key] for key in saved.files}
    np.testing.assert_array_equal(state['coordinates'],np.argwhere(csf))
    np.testing.assert_array_equal(state['truth_labels'],labels[csf])
    temperature_median=np.full(csf.shape,np.nan);temperature_median[csf]=state['temperature']
    mad=np.full(csf.shape,np.nan);mad[csf]=state['mad']
    observed_z=np.full(csf.shape,np.nan);observed_z[csf]=state['z']
    null=np.where(np.isfinite(state['null_median']),state['null_median'],state['other_challenge_null_median'])
    null_median=np.full(csf.shape,np.nan);null_median[csf]=null
    selected=state['primary_retained']|state['other_challenge_retained']
    eligible=state['primary_eligible']|state['challenge_eligible']
    np.testing.assert_array_equal(selected,eligible&(state['z']<null))
    retained=np.zeros(csf.shape,bool);retained[csf]=selected
    names=np.array([{1:'LV',2:'SA',3:'Other'}[int(label)] for label in labels[csf]])
    voxel_table=pd.DataFrame({'region':names,'temperature_median_c':state['temperature'],
        'mad_c':state['mad'],'observed_z':state['z'],'null_median_z':null,'retained':selected})
    summary=pd.read_csv(analysis_root/'regional_ground_truth_summary.tsv',sep='\t').set_index('region')
    for region in REGION_ORDER:
        mask=names==region;valid=mask&state['valid'];picked=mask&selected
        if int(mask.sum())!=summary.loc[region,'n_total'] or int(valid.sum())!=summary.loc[region,'n_valid'] or int(picked.sum())!=summary.loc[region,'n_retained']:
            raise ValueError('Synthetic state and regional table counts disagree')
        if np.any(picked):
            np.testing.assert_allclose(np.mean(state['temperature'][picked]),summary.loc[region,'temperature_mean_retained_c'],rtol=1e-14,atol=1e-12)
    cluster_overlap=pd.read_csv(analysis_root/'cluster_truth_overlap.tsv',sep='\t')
    recovered=float(summary.loc['LV','temperature_mean_retained_c']-summary.loc['SA','temperature_mean_retained_c'])
    if not np.isclose(recovered,metadata['recovered_lv_minus_sa_c'],rtol=0,atol=1e-12):
        raise ValueError('Synthetic paired temperature metadata disagree')
    return SyntheticFigureData(b0=np.asarray(nib.load(analysis_root/'mean_signal_b0.nii.gz').dataobj),
        labels=labels,temperature_median=temperature_median,mad=mad,observed_z=observed_z,
        null_median=null_median,metadata=metadata,retained=retained,voxel_table=voxel_table,
        summary=summary,cluster_overlap=cluster_overlap,
        target_gradient_c=metadata['target_lv_minus_sa_c'],recovered_gradient_c=recovered)


def _image_plane(volume: np.ndarray, y_index: int) -> np.ndarray:
    return np.rot90(volume[:, y_index, :])


def _label_rgba(
    labels: np.ndarray,
    y_index: int,
    retained: np.ndarray | None = None,
) -> np.ndarray:
    plane = _image_plane(labels, y_index)
    rgba = np.zeros((*plane.shape, 4), dtype=np.float64)
    if retained is not None:
        present = plane > 0
        rgba[present] = to_rgba(COLORS["excluded"], 0.68)
        retained_plane = _image_plane(retained, y_index)
    else:
        retained_plane = np.ones_like(plane, dtype=bool)

    for region, label in LABELS.items():
        region_mask = (plane == label) & retained_plane
        rgba[region_mask] = to_rgba(COLORS[region], 0.82)
    return rgba


def _show_digital_slice(
    ax: plt.Axes,
    data: SyntheticFigureData,
    *,
    classification: bool,
) -> None:
    y_index = data.labels.shape[1] // 2
    base = _image_plane(data.b0, y_index)
    brain_values = base[base > 100]
    ax.imshow(
        base,
        cmap="gray",
        vmin=np.percentile(brain_values, 2),
        vmax=np.percentile(brain_values, 99),
        interpolation="nearest",
    )
    ax.imshow(
        _label_rgba(
            data.labels,
            y_index,
            retained=data.retained if classification else None,
        ),
        interpolation="nearest",
    )
    ax.set_axis_off()


def _plot_ground_truth(
    ax: plt.Axes,
    legend_ax: plt.Axes,
    data: SyntheticFigureData,
) -> None:
    _show_digital_slice(ax, data, classification=False)
    ax.set_title("Known digital compartments", loc="left")
    handles = [
        Patch(
            facecolor=COLORS["LV"],
            edgecolor="none",
            label=f"LV  {TEMPERATURES_C['LV']:.1f} °C",
        ),
        Patch(
            facecolor=COLORS["SA"],
            edgecolor="none",
            label=f"SA  {TEMPERATURES_C['SA']:.1f} °C",
        ),
        Patch(
            facecolor=COLORS["Other"],
            edgecolor="none",
            label="Other  34-52 °C across shells",
        ),
    ]
    legend_ax.set_axis_off()
    legend_ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        borderaxespad=0,
        ncol=1,
    )


def _plot_temperature_mad(ax: plt.Axes, data: SyntheticFigureData) -> None:
    for region in ("Other", "SA", "LV"):
        frame = data.voxel_table[data.voxel_table["region"] == region]
        ax.scatter(
            frame["temperature_median_c"],
            frame["mad_c"],
            s=5 if region != "Other" else 7,
            color=COLORS[region],
            alpha=0.27 if region == "SA" else 0.45,
            linewidths=0,
            rasterized=True,
            label=region,
        )
    ax.set_title("Cross-shell temperature variation", loc="left")
    ax.set_xlabel("Median temperature, $T_{med}$ (°C)")
    ax.set_ylabel("Cross-shell MAD (°C)")
    ax.set_xlim(35.0, 44.5)
    ax.set_ylim(0, 6.5)
    ax.legend(frameon=False, fontsize=8, markerscale=2.0, loc="center left")


def _plot_consistency(
    image_ax: plt.Axes,
    bar_ax: plt.Axes,
    data: SyntheticFigureData,
) -> None:
    _show_digital_slice(image_ax, data, classification=True)
    image_ax.set_title("Shared-baseline consistency", loc="left")
    image_ax.text(
        0.5,
        -0.02,
        "colour: passes criterion   grey: fails",
        transform=image_ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#4C4F55",
    )

    fractions = [100 * data.summary.loc[r, "retained_fraction"] for r in REGION_ORDER]
    bars = bar_ax.bar(
        np.arange(3),
        fractions,
        color=[COLORS[r] for r in REGION_ORDER],
        width=0.62,
    )
    bar_ax.set_ylim(0, 116)
    bar_ax.set_yticks([0,25,50,75,100])
    bar_ax.set_xticks(np.arange(3), ("LV","SA","Other*"))
    bar_ax.set_ylabel("Passes criterion (%)")
    bar_ax.spines[["top", "right"]].set_visible(False)
    bar_ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.6)
    bar_ax.set_axisbelow(True)
    for bar, region in zip(bars, REGION_ORDER, strict=True):
        retained = int(data.summary.loc[region, "n_retained"])
        total = int(data.summary.loc[region, "n_valid"])
        bar_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{retained}/{total}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    clusters = data.cluster_overlap[data.cluster_overlap["cluster"] >= 0]
    pure = bool(np.allclose(clusters["truth_purity"], 1.0))
    qualifier = "all 100% truth-pure" if pure else "truth overlap reported"
    bar_ax.text(
        0.5,
        -0.43,
        f"*Other: separate challenge; excluded from regional means\nHDBSCAN: {len(clusters)} clusters, {qualifier}",
        transform=bar_ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color="#4C4F55",
    )


def _plot_target_recovery(
    temperature_ax: plt.Axes,
    gradient_ax: plt.Axes,
    data: SyntheticFigureData,
) -> None:
    temperature_ax.set_title("Programmed and recovered values", loc="left")
    x = np.arange(2)
    for index, region in enumerate(("LV", "SA")):
        target = float(TEMPERATURES_C[region])
        recovered = float(
            data.summary.loc[region, "temperature_mean_retained_c"]
        )
        temperature_ax.plot(
            index - 0.07,
            target,
            marker="D",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=COLORS[region],
            markeredgewidth=1.5,
            linestyle="none",
        )
        temperature_ax.plot(
            index + 0.07,
            recovered,
            marker="o",
            markersize=7,
            color=COLORS[region],
            linestyle="none",
        )
        temperature_ax.plot(
            [index - 0.07, index + 0.07],
            [target, recovered],
            color=COLORS[region],
            linewidth=1,
            alpha=0.65,
        )
        temperature_ax.text(
            index,
            max(target, recovered) + 0.18,
            f"{target:.2f} / {recovered:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS[region],
        )
    temperature_ax.set_xlim(-0.55, 1.55)
    temperature_ax.set_ylim(35.6, 39.5)
    temperature_ax.set_xticks(x, ("LV", "SA"))
    temperature_ax.set_ylabel("Temperature (°C)")
    temperature_ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.6)
    temperature_ax.set_axisbelow(True)
    temperature_ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="D",
                markerfacecolor="white",
                markeredgecolor="#50545A",
                linestyle="none",
                label="Programmed",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color="#50545A",
                linestyle="none",
                label="Recovered",
            ),
        ],
        frameon=False,
        fontsize=8,
        loc="upper right",
    )

    target = data.target_gradient_c
    recovered = data.recovered_gradient_c
    gradient_ax.plot(
        [target, recovered],
        [0, 0],
        color="#80848A",
        linewidth=1.2,
        zorder=1,
    )
    gradient_ax.plot(
        target,
        0,
        marker="D",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="#202124",
        markeredgewidth=1.5,
        linestyle="none",
        zorder=2,
    )
    gradient_ax.plot(
        recovered,
        0,
        marker="o",
        markersize=7,
        color="#202124",
        linestyle="none",
        zorder=3,
    )
    gradient_ax.text(
        target,
        0.28,
        f"target {target:.2f}",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    gradient_ax.text(
        recovered,
        -0.28,
        f"recovered {recovered:.2f}",
        ha="center",
        va="top",
        fontsize=7.5,
    )
    gradient_ax.set_xlim(2.35, 2.65)
    gradient_ax.set_ylim(-0.7, 0.7)
    gradient_ax.set_yticks([])
    gradient_ax.set_xlabel("LV-SA gradient (°C)")
    gradient_ax.spines[["left", "right", "top"]].set_visible(False)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.018,
        0.978,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        weight="bold",
        color="#111111",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.4, "alpha": 0.9},
        zorder=10,
    )


def build_figure(data: SyntheticFigureData) -> plt.Figure:
    """Build the four-panel synthetic validation figure."""
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
    with mpl.rc_context(style):
        figure = plt.figure(figsize=(7.2, 7.0), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.08))
        grid_a = grid[0, 0].subgridspec(2, 1, height_ratios=(1.0, 0.23))
        ax_a = figure.add_subplot(grid_a[0, 0])
        ax_a_legend = figure.add_subplot(grid_a[1, 0])
        ax_b = figure.add_subplot(grid[0, 1])

        grid_c = grid[1, 0].subgridspec(2, 1, height_ratios=(1.45, 1.0))
        ax_c_image = figure.add_subplot(grid_c[0, 0])
        ax_c_bar = figure.add_subplot(grid_c[1, 0])

        grid_d = grid[1, 1].subgridspec(2, 1, height_ratios=(1.4, 0.8))
        ax_d_temperature = figure.add_subplot(grid_d[0, 0])
        ax_d_gradient = figure.add_subplot(grid_d[1, 0])

        _plot_ground_truth(ax_a, ax_a_legend, data)
        _plot_temperature_mad(ax_b, data)
        _plot_consistency(ax_c_image, ax_c_bar, data)
        _plot_target_recovery(ax_d_temperature, ax_d_gradient, data)

        for axis, label in (
            (ax_a, "a"),
            (ax_b, "b"),
            (ax_c_image, "c"),
            (ax_d_temperature, "d"),
        ):
            _panel_label(axis, label)
        return figure


def write_figure(
    data: SyntheticFigureData,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Write the figure in publication and review formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = build_figure(data)
    pdf_path = output_dir / "synthetic_validation.pdf"
    png_path = output_dir / "synthetic_validation.png"
    with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.04,
            metadata={
                "Title": "Synthetic demonstration of mean-domain shared-baseline CSF thermometry",
                "Author": "Thermo Amyloid study",
                "Subject": "Implementation and reproducibility validation",
            },
        )
    figure.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    values={"schema_version":2,"branch":data.metadata['branch'],
        "classifier":data.metadata['classifier'],"target_difference_c":data.target_gradient_c,
        "recovered_difference_c":data.recovered_gradient_c,
        "regional_values":[{key:None if pd.isna(value) else value for key,value in row.items()} for row in data.summary.reset_index().to_dict(orient='records')],
        "Other_scope":"separate same-rule challenge; not a regional estimate",
        "files_sha256":{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in [pdf_path,png_path]}}
    (output_dir/'figure_values.json').write_text(json.dumps(values,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return {"pdf": pdf_path, "png": png_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=DEFAULT_BIDS_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data = load_figure_data(args.bids_root, args.analysis_root)
    outputs = write_figure(data, args.output_dir)
    for kind, path in outputs.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
