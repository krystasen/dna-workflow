"""Gaussian/IEFPCM 11660态 SPURIOUS30 V2 的 Figure 3a-d 生产绘图与公共绘图函数。"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree


MATERIAL = Path(
    r"D:\data\DNA DoS\computation\GGCC_GAU_0.00_IEFPCM"
    r"\GAU_0.00_IEFPCM_torch_full_11660_spurious30_v2"
)
RESULTS = MATERIAL / "results_c64"
OUTPUT = RESULTS / "figure3_system_11660_gaussian_spurious30_v2"
KBT_EV = 8.617333262145e-5 * 300.0
PAPER_BIN_EV = 0.024
DISPLAY_SIGMA_EV = 0.005
GATEWAY = {
    "energy_index": 153,
    "energy_eV": -5.235,
    "left_contact_projected_dos": 0.018140891566872597,
}
GATEWAY_CANDIDATES = [
    {**GATEWAY, "left_base_pair_dos": 0.17643057176726717},
    {
        "energy_index": 173,
        "energy_eV": -5.135,
        "left_contact_projected_dos": 0.016832834109663963,
        "left_base_pair_dos": 0.07822124104615913,
    },
]
BRIDGE_INDICES = np.arange(6, 10)
VDW_A = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80, "Cl": 1.75}
COVALENT_A = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "P": 1.07}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def load_inputs():
    metadata = json.loads((MATERIAL / "metadata.json").read_text(encoding="utf-8"))
    with np.load(RESULTS / "dos2d_results.npz", allow_pickle=False) as data:
        energies = np.asarray(data["energy_eV"])
        raw = np.asarray(data["dos2d"])
        total = np.asarray(data["total_dos"])
        labels = data["base_pair_display_labels"].astype(str).tolist()
        dtype = str(data["dtype"])
    with np.load(RESULTS / "dos2d_binned_0p024.npz", allow_pickle=False) as data:
        bin_energy = np.asarray(data["energy_eV"])
        bin_width = np.asarray(data["bin_width_eV"])
        binned = np.asarray(data["dos2d"])
    if raw.shape != (1201, 16) or total.shape != (1201,) or binned.shape != (250, 16):
        raise ValueError(f"Unexpected DoS shapes: {raw.shape}, {total.shape}, {binned.shape}.")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(binned)):
        raise ValueError("DoS contains NaN or infinity.")
    if not np.allclose(raw.sum(axis=1), total, rtol=1e-10, atol=1e-10):
        raise ValueError("Base-pair DoS does not conserve total DoS.")
    raw_area = np.trapezoid(raw.sum(axis=1), energies)
    binned_area = np.sum(binned.sum(axis=1) * bin_width)
    if not np.isclose(raw_area, binned_area, rtol=1e-11, atol=1e-11):
        raise ValueError("0.024 eV bins do not conserve total spectral weight.")
    spacing = float(np.median(np.diff(energies)))
    display = gaussian_filter1d(raw, DISPLAY_SIGMA_EV / spacing, axis=0, mode="nearest")
    return metadata, energies, raw, display, bin_energy, bin_width, binned, labels, dtype


def gaussian_geometry(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(
        index for index, line in enumerate(lines) if re.fullmatch(r"\s*-?\d+\s+\d+\s*", line)
    ) + 1
    elements, coordinates = [], []
    for line in lines[start:]:
        fields = line.split()
        if len(fields) < 4:
            break
        elements.append(fields[0].upper())
        coordinates.append([float(value) for value in fields[1:4]])
    if any(element not in VDW_A for element in elements):
        raise ValueError("Unsupported element in Gaussian geometry.")
    return elements, np.asarray(coordinates)


def base_pair_atoms(partition: dict, order: list[str], n_atoms: int):
    groups = {label: [] for label in order}
    atom_to_pair = np.full(n_atoms, -1, dtype=int)
    for nucleotide in partition["nucleotides"]:
        first, last = map(int, nucleotide["atom_range_1based"])
        groups[str(nucleotide["base_pair"])].extend(range(first - 1, last))
    arrays = []
    for column, label in enumerate(order):
        indices = np.asarray(sorted(groups[label]), dtype=int)
        if not indices.size or np.any(atom_to_pair[indices] >= 0):
            raise ValueError(f"Invalid atom grouping for base pair {label}.")
        atom_to_pair[indices] = column
        arrays.append(indices)
    if np.any(atom_to_pair < 0):
        raise ValueError("Base-pair groups do not cover all atoms exactly once.")
    return arrays, atom_to_pair


def infer_bonds(elements: list[str], coordinates: np.ndarray):
    bonds = []
    for first, second in cKDTree(coordinates).query_pairs(2.4):
        distance = float(np.linalg.norm(coordinates[first] - coordinates[second]))
        if distance < 1.25 * (COVALENT_A[elements[first]] + COVALENT_A[elements[second]]):
            bonds.append((first, second))
    return bonds


def pca_coordinates(coordinates: np.ndarray):
    centered = coordinates - coordinates.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rotated = centered @ vh.T
    if rotated[0, 0] > rotated[-1, 0]:
        rotated[:, 0] *= -1
    return rotated


def cumulative_density(
    bin_energy: np.ndarray,
    bin_width: np.ndarray,
    binned: np.ndarray,
    volumes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    low = GATEWAY["energy_eV"] - 3 * KBT_EV
    high = GATEWAY["energy_eV"] + 3 * KBT_EV
    left, right = bin_energy - bin_width / 2, bin_energy + bin_width / 2
    overlap = np.maximum(0.0, np.minimum(right, high) - np.maximum(left, low))
    integrated_states = np.sum(binned * overlap[:, None], axis=0)
    return integrated_states, integrated_states / PAPER_BIN_EV / volumes


def panel_label(ax, label: str, x: float = -0.14, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def style_bridge_ticks(ax) -> None:
    for index, tick in enumerate(ax.get_xticklabels()):
        if index in BRIDGE_INDICES:
            tick.set_color("#C2185B")


def draw_figure(
    energies,
    display,
    labels,
    cumulative,
    rotated,
    bp_atoms,
    bonds,
):
    fig = plt.figure(figsize=(7.20, 6.25), facecolor="white")
    grid = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.045],
        height_ratios=[1.08, 0.92],
        left=0.08,
        right=0.96,
        bottom=0.10,
        top=0.96,
        wspace=0.32,
        hspace=0.48,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    cax = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1:], projection="3d")

    dos_norm = LogNorm(vmin=1e-2, vmax=1e2)
    image = ax_a.imshow(
        display,
        origin="lower",
        aspect="auto",
        extent=[-0.5, len(labels) - 0.5, energies[0], energies[-1]],
        cmap="plasma",
        norm=dos_norm,
        interpolation="bilinear",
        rasterized=True,
    )
    ax_a.set(xlabel="Base pair", ylabel="Energy (eV)")
    ax_a.set_xticks(range(len(labels)), labels, rotation=90)
    style_bridge_ticks(ax_a)
    ax_a.annotate(
        "",
        xy=(-0.45, GATEWAY["energy_eV"]),
        xytext=(-1.25, GATEWAY["energy_eV"]),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.9},
        annotation_clip=False,
    )
    panel_label(ax_a, "a")

    zoom = np.abs(energies - GATEWAY["energy_eV"]) <= 9 * KBT_EV
    relative = (energies[zoom] - GATEWAY["energy_eV"]) / KBT_EV
    ax_b.imshow(
        display[zoom],
        origin="lower",
        aspect="auto",
        extent=[-0.5, len(labels) - 0.5, relative[0], relative[-1]],
        cmap="plasma",
        norm=dos_norm,
        interpolation="bilinear",
        rasterized=True,
    )
    ax_b.axhline(-3, color="white", lw=0.9, ls=(0, (4, 3)))
    ax_b.axhline(3, color="white", lw=0.9, ls=(0, (4, 3)))
    ax_b.set_ylim(-9, 9)
    ax_b.set_yticks([-9, -6, -3, 0, 3, 6, 9])
    ax_b.set(xlabel="Base pair", ylabel=r"$(E-E_{\mathrm{GS(L)}})/k_\mathrm{B}T$")
    ax_b.set_xticks(range(len(labels)), labels, rotation=90)
    style_bridge_ticks(ax_b)
    panel_label(ax_b, "b")
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label(r"DoS (eV$^{-1}$)")

    x = np.arange(len(labels))
    ymax = max(float(cumulative.max()) * 1.13, 1.0)
    ax_c.plot(x, cumulative, color="#202020", lw=1.0, marker="o", ms=2.6)
    for index in BRIDGE_INDICES:
        ax_c.vlines(index, 0, ymax * 0.92, color="#C2185B", lw=0.9, ls=(0, (6, 3)))
    ax_c.text(BRIDGE_INDICES.mean(), ymax * 0.96, "GGCC", color="#C2185B", ha="center", va="top")
    ax_c.set_ylim(0, ymax)
    ax_c.set(xlabel="Base pair", ylabel=r"Cumulative states (eV$^{-1}$ nm$^{-3}$)")
    ax_c.set_xticks(x, labels, rotation=90)
    style_bridge_ticks(ax_c)
    panel_label(ax_c, "c")

    segments = np.asarray([[rotated[first], rotated[second]] for first, second in bonds])
    ax_d.add_collection3d(
        Line3DCollection(segments, colors="#B0B0B0", linewidths=0.25, alpha=0.25)
    )
    ax_d.scatter(
        rotated[:, 0], rotated[:, 1], rotated[:, 2],
        color="#B8B8B8", s=2.2, alpha=0.18, depthshade=False,
    )
    centers = np.asarray([rotated[indices].mean(axis=0) for indices in bp_atoms])
    log_density = np.log10(np.maximum(cumulative, 10**-0.5))
    bp_norm = Normalize(vmin=-0.5, vmax=1.75, clip=True)
    scatter = ax_d.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2],
        c=log_density, cmap="plasma", norm=bp_norm,
        s=64, edgecolors="white", linewidths=0.45, depthshade=False,
    )
    ax_d.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="#606060", lw=0.5, alpha=0.55)
    for index, center in enumerate(centers, 1):
        text_color = "white" if bp_norm(log_density[index - 1]) < 0.58 else "black"
        ax_d.text(*center, str(index), fontsize=4.5, ha="center", va="center", color=text_color)
    ax_d.set_axis_off()
    spans = np.ptp(rotated, axis=0)
    lower, upper = rotated.min(axis=0), rotated.max(axis=0)
    margins = np.maximum(spans * 0.05, 0.5)
    ax_d.set_xlim(lower[0] - margins[0], upper[0] + margins[0])
    ax_d.set_ylim(lower[1] - margins[1], upper[1] + margins[1])
    ax_d.set_zlim(lower[2] - margins[2], upper[2] + margins[2])
    ax_d.set_box_aspect((spans[0], max(spans[1], 1), max(spans[2], 1)), zoom=1.15)
    ax_d.view_init(elev=11, azim=-88)
    ax_d.text2D(-0.04, 1.02, "d", transform=ax_d.transAxes, fontsize=9, fontweight="bold")
    inset = ax_d.inset_axes([0.28, -0.07, 0.48, 0.055])
    cbar_d = fig.colorbar(scatter, cax=inset, orientation="horizontal")
    cbar_d.set_ticks([-0.5, 1.75])
    cbar_d.set_label(r"log$_{10}$(cumulative DoS density)", labelpad=-1)
    return fig, centers, log_density


def calculate():
    metadata, energies, raw, display, bin_energy, bin_width, binned, labels, dtype = load_inputs()
    elements, coordinates = gaussian_geometry(Path(metadata["source_geometry"]))
    partition = json.loads((MATERIAL / "gaussian_partitions.json").read_text(encoding="utf-8"))
    order = [str(label) for label in metadata["base_pair_order"]]
    bp_atoms, atom_to_pair = base_pair_atoms(partition, order, len(elements))
    atom_volumes = np.asarray([4 * np.pi * VDW_A[element] ** 3 / 3 / 1000 for element in elements])
    volumes = np.asarray([atom_volumes[indices].sum() for indices in bp_atoms])
    integrated, cumulative = cumulative_density(bin_energy, bin_width, binned, volumes)
    rotated = pca_coordinates(coordinates)
    bonds = infer_bonds(elements, coordinates)
    return {
        "metadata": metadata,
        "energies": energies,
        "raw": raw,
        "display": display,
        "bin_energy": bin_energy,
        "bin_width": bin_width,
        "binned": binned,
        "labels": labels,
        "dtype": dtype,
        "elements": elements,
        "coordinates": coordinates,
        "rotated": rotated,
        "bp_atoms": bp_atoms,
        "atom_to_pair": atom_to_pair,
        "volumes": volumes,
        "integrated": integrated,
        "cumulative": cumulative,
        "bonds": bonds,
    }


def run() -> None:
    data = calculate()
    fig, centers, log_density = draw_figure(
        data["energies"], data["display"], data["labels"], data["cumulative"],
        data["rotated"], data["bp_atoms"], data["bonds"],
    )
    base = OUTPUT / f"figure3_abcd_gaussian_spurious30_v2_{data['dtype']}"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    write_csv(
        OUTPUT / "gateway_candidates.csv",
        ["energy_index", "energy_eV", "left_base_pair_dos", "left_contact_projected_dos", "selected"],
        ([row["energy_index"], row["energy_eV"], row["left_base_pair_dos"], row["left_contact_projected_dos"], row["energy_index"] == GATEWAY["energy_index"]] for row in GATEWAY_CANDIDATES),
    )
    write_csv(OUTPUT / "panel_a_display_dos.csv", ["energy_eV", *data["labels"]], np.column_stack([data["energies"], data["display"]]))
    zoom = np.flatnonzero(np.abs(data["energies"] - GATEWAY["energy_eV"]) <= 9 * KBT_EV)
    write_csv(
        OUTPUT / "panel_b_gateway_heatmap.csv",
        ["energy_eV", "energy_minus_gateway_kBT", *data["labels"]],
        ([data["energies"][index], (data["energies"][index] - GATEWAY["energy_eV"]) / KBT_EV, *data["display"][index]] for index in zoom),
    )
    write_csv(
        OUTPUT / "panel_c_cumulative_states.csv",
        ["base_pair_1based", "label", "volume_nm3", "integrated_states", "cumulative_states_eV-1_nm-3"],
        ([index + 1, data["labels"][index], data["volumes"][index], data["integrated"][index], data["cumulative"][index]] for index in range(16)),
    )
    write_csv(
        OUTPUT / "panel_d_base_pair_values.csv",
        ["base_pair_1based", "label", "centroid_x_A", "centroid_y_A", "centroid_z_A", "cumulative_states_eV-1_nm-3", "log10_cumulative_density"],
        ([index + 1, data["labels"][index], *centers[index], data["cumulative"][index], log_density[index]] for index in range(16)),
    )

    figure_metadata = {
        "figure_contract": {
            "core_conclusion": "Test the Gaussian gateway and spatial DoS after analytic KK, 30 dangling-bond exclusions and finite-resolution integration.",
            "archetype": "asymmetric mixed-modality figure",
            "backend": "Python",
            "final_size_inches": [7.2, 6.25],
            "reviewer_risk": "The 3prime-O contact projector and phenomenological Gamma parameters remain reproduction assumptions.",
        },
        "model": "GGCC Gaussian/IEFPCM full 11660-state spurious30 v2",
        "numerical_precision": data["dtype"],
        "gateway_definition": "Upper-band left-terminal peaks ranked by 3prime-O contact-projected DoS.",
        "gateway": GATEWAY,
        "gateway_candidates": GATEWAY_CANDIDATES,
        "temperature_K": 300.0,
        "panel_b_window_kBT": [-9.0, 9.0],
        "panel_c_d_window_kBT": [-3.0, 3.0],
        "panel_c_d_source": "0.024 eV bin-averaged DoS with fractional overlap at the integration-window boundaries",
        "panel_c_max": float(data["cumulative"].max()),
        "panel_c_max_bp_1based": int(data["cumulative"].argmax() + 1),
        "panel_d_colour_unit": "base pair; exactly 16 coloured centroid markers",
        "display_gaussian_sigma_eV": DISPLAY_SIGMA_EV,
        "physical_lorentzian_half_width_eV": data["metadata"]["eta_eV"],
        "binned_integral_total_states": float(np.sum(data["binned"].sum(axis=1) * data["bin_width"])),
        "volume_model": "sum of Bondi van der Waals sphere volumes; overlaps not subtracted",
        "bonds_drawn": len(data["bonds"]),
        "model_boundary": data["metadata"]["model_boundary"],
    }
    (OUTPUT / "figure3_metadata.json").write_text(json.dumps(figure_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "FIGURE_CAPTION.md").write_text(
        "# Figure 3 — Gaussian/IEFPCM full 11660-state GGCC DoS, spurious30 v2\n\n"
        f"**a,** Base-pair-resolved two-dimensional DoS. The arrow marks the reconstructed left gateway at {GATEWAY['energy_eV']:.3f} eV. "
        "**b,** Gateway region within ±9kBT; dashed lines delimit the ±3kBT integration interval. "
        "**c,** Cumulative states obtained from the integral-conserving 0.024 eV binned DoS and normalized by base-pair van der Waals volume. "
        "**d,** Base-pair-level structural map of the same density; the 16 coloured markers are the quantitative colour units and grey atoms/bonds provide structural context. "
        "The Gaussian/IEFPCM Hamiltonian contains all 11,660 localized states; the v2 self-energy uses an infinite-domain analytic KK transform and excludes 30 O–P partition dangling-bond states from decoherence probes.\n",
        encoding="utf-8",
    )
    print(json.dumps(figure_metadata, ensure_ascii=False, indent=2))


def self_test() -> None:
    data = calculate()
    assert data["dtype"] == "complex64"
    assert len(data["labels"]) == len(data["bp_atoms"]) == 16
    assert np.isclose(np.trapezoid(data["raw"].sum(axis=1), data["energies"]), np.sum(data["binned"].sum(axis=1) * data["bin_width"]), rtol=1e-11)
    assert 0 < data["cumulative"].max() < 100
    assert GATEWAY["energy_index"] == int(np.argmin(np.abs(data["energies"] - GATEWAY["energy_eV"])))
    print(f"PASS: Figure 3 inputs, 0.024 eV integration and 16-bp colour mapping; panel-c max={data['cumulative'].max():.6f}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "self-test"])
    args = parser.parse_args()
    self_test() if args.command == "self-test" else run()
