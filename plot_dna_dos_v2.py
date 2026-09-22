"""Sequence-independent Figure 3a-d plotting for returned DNA V2 GPU results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import dna_dos_v2 as workflow
import plot_figure3_abcd_gaussian_spurious30_v2 as reference


KB_EV = 8.617333262145e-5


def window_integral(centers, widths, dos, volumes, gateway, temperature, bin_width):
    low, high = gateway - 3 * KB_EV * temperature, gateway + 3 * KB_EV * temperature
    left, right = centers - widths / 2, centers + widths / 2
    if low < left[0] - 1e-10 or high > right[-1] + 1e-10:
        raise ValueError("gateway +/-3 kBT extends outside the computed energy window; expand the calculation window or choose a covered energy.")
    overlap = np.maximum(0.0, np.minimum(right, high) - np.maximum(left, low))
    integrated = np.sum(dos * overlap[:, None], axis=0)
    # Same paper/V2 convention: divide by the reporting bin, not by 6 kBT.
    return integrated, integrated / bin_width / volumes


def choose_gateway(energies, dos, projected, left_bp, temperature, manual=None, bounds=(None, None)):
    margin = 3 * KB_EV * temperature
    low = energies[0] + margin if bounds[0] is None else max(bounds[0], energies[0] + margin)
    high = energies[-1] - margin if bounds[1] is None else min(bounds[1], energies[-1] - margin)
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("The gateway search interval is empty or invalid.")
    region = np.flatnonzero((energies >= low) & (energies <= high))
    if len(region) < 3:
        raise ValueError("Too few energy samples in the gateway search interval.")
    peaks, _ = find_peaks(dos[region, left_bp])
    method = "terminal-bp local maxima ranked by left contact-projected DoS"
    if not len(peaks):
        peaks = np.asarray([int(np.argmax(dos[region, left_bp]))])
        method = "terminal-bp maximum fallback; no local peak in search interval"
    candidates = [{"energy_index": int(region[index]), "energy_eV": float(energies[region[index]]),
                   "terminal_bp_dos": float(dos[region[index], left_bp]),
                   "left_contact_dos": float(projected[region[index], 0])} for index in peaks]
    candidates.sort(key=lambda row: row["left_contact_dos"], reverse=True)
    if manual is None:
        selected = candidates[0] | {"selection": method}
    else:
        if not np.isfinite(manual) or not energies[0] + margin <= manual <= energies[-1] - margin:
            raise ValueError("Manual gateway must have a fully covered +/-3 kBT integration window.")
        selected = {"energy_index": int(np.argmin(abs(energies - manual))), "energy_eV": float(manual),
                    "left_contact_dos": float(np.interp(manual, energies, projected[:, 0])),
                    "selection": "user-specified energy"}
    selected["search_interval_eV"] = [float(low), float(high)]
    return selected, candidates


def calculate(results: Path, gateway=None, temperature=None, display_sigma=0.005, bounds=(None, None)) -> dict:
    workflow.verify_results(results)
    metadata = workflow.read_json(results / "run_manifest.json")
    temperature = metadata["parameters"]["temperature_K"] if temperature is None else temperature
    if not np.isfinite(temperature) or temperature <= 0 or not np.isfinite(display_sigma) or display_sigma < 0:
        raise ValueError("Temperature must be positive and display smoothing nonnegative.")
    with np.load(results / "dos2d_results.npz", allow_pickle=False) as raw:
        energies, dos, contact = raw["energy_eV"], raw["dos2d"], raw["contact_dos"]
    with np.load(results / "dos2d_binned.npz", allow_pickle=False) as bins:
        centers, widths, binned = bins["energy_eV"], bins["bin_width_eV"], bins["dos2d"]
    left_bp = next(item for item in metadata["contacts"] if item["side"] == "left")["base_pair_index"]
    selected, candidates = choose_gateway(energies, dos, contact, left_bp, temperature, gateway, bounds)
    elements, xyz = reference.gaussian_geometry(results / "dna.gjf")
    if len(elements) != metadata["atom_count"]:
        raise ValueError("Returned structure atom count differs from the calculation.")
    partition = workflow.read_json(results / "gaussian_partitions.json")
    bp_atoms, atom_to_pair = reference.base_pair_atoms(partition, metadata["base_pair_order"], len(elements))
    atomic_volume = np.asarray([4 * np.pi * reference.VDW_A[element] ** 3 / 3 / 1000 for element in elements])
    volumes = np.asarray([atomic_volume[indices].sum() for indices in bp_atoms])
    integrated, cumulative = window_integral(centers, widths, binned, volumes, selected["energy_eV"],
                                             temperature, metadata["parameters"]["dos_bin_width_eV"])
    display = gaussian_filter1d(dos, display_sigma / np.median(np.diff(energies)), axis=0, mode="nearest") if display_sigma else dos.copy()
    rotated = reference.pca_coordinates(xyz)
    bp_centers = np.asarray([rotated[indices].mean(axis=0) for indices in bp_atoms])
    if bp_centers[0, 0] > bp_centers[-1, 0]:
        rotated[:, 0] *= -1
        bp_centers[:, 0] *= -1
    return {"metadata": metadata, "energies": energies, "raw": dos, "contact": contact,
            "display": display, "temperature": temperature, "display_sigma": display_sigma,
            "gateway": selected, "candidates": candidates, "bin_centers": centers, "bin_widths": widths,
            "binned": binned, "labels": metadata["base_pair_display_labels"], "volumes": volumes,
            "integrated": integrated, "cumulative": cumulative, "elements": elements, "rotated": rotated,
            "bp_centers": bp_centers, "bp_atoms": bp_atoms, "atom_to_pair": atom_to_pair,
            }


def draw_figure(data: dict):
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    labels, energy = data["labels"], data["energies"]
    n = len(labels)
    ticks = [f"{index + 1} {label}" for index, label in enumerate(labels)]
    figure = plt.figure(figsize=(7.2, 8.0), facecolor="white")
    grid = figure.add_gridspec(3, 3, width_ratios=[1, 1, 0.045], height_ratios=[1.2, 0.7, 0.85],
                              left=0.10, right=0.885, bottom=0.16, top=0.945, hspace=0.65, wspace=0.32)
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]),
            figure.add_subplot(grid[1, :2]), figure.add_subplot(grid[2, :2])]
    a, b, c, d = axes
    norm = LogNorm(vmin=1e-2, vmax=1e2)
    image = a.imshow(data["display"], origin="lower", aspect="auto", extent=[0.5, n + 0.5, energy[0], energy[-1]],
                     cmap="plasma", norm=norm, interpolation="bilinear", rasterized=True)
    a.set(xlabel="Base-pair position" if len(labels[0]) == 2 else "Nucleotide position", ylabel="Energy (eV)")
    a.set_xticks(np.arange(1, n + 1), ticks, rotation=90, fontsize=5.2)
    a.axhline(data["gateway"]["energy_eV"], color="white", lw=0.6, ls=(0, (3, 3)))
    relative = (energy - data["gateway"]["energy_eV"]) / (KB_EV * data["temperature"])
    b.imshow(data["display"], origin="lower", aspect="auto", extent=[0.5, n + 0.5, relative[0], relative[-1]],
             cmap="plasma", norm=norm, interpolation="bilinear", rasterized=True)
    b.set(ylim=(-9, 9), yticks=[-9, -6, -3, 0, 3, 6, 9], xlabel="Position", ylabel=r"$(E-E_{\mathrm{gateway}})/k_\mathrm{B}T$")
    b.set_xticks(np.arange(1, n + 1), ticks, rotation=90, fontsize=5.2)
    for limit in (-3, 3):
        b.axhline(limit, color="white", ls=(0, (4, 3)), lw=0.7)
    figure.colorbar(image, cax=figure.add_subplot(grid[0, 2]), extend="both").set_label(r"DoS (eV$^{-1}$)")
    c.plot(np.arange(1, n + 1), data["cumulative"], "o-", color="#202020", lw=0.95, ms=2.8)
    c.set(xlim=(0.5, n + 0.5), ylim=(0, max(float(data["cumulative"].max()) * 1.16, 1)),
          ylabel="Cumulative states\n" + r"(eV$^{-1}$ nm$^{-3}$)")
    c.set_xticks(np.arange(1, n + 1), ticks, rotation=90, fontsize=6)
    c.grid(axis="y", lw=0.4, color="#eaecf0")
    c.set_axisbelow(True)
    xyz, centers = data["rotated"], data["bp_centers"]
    log_density = np.log10(np.maximum(data["cumulative"], 10**-0.5))
    depth_order = np.argsort(xyz[:, 2])
    span = max(float(np.ptp(xyz[:, 0])), 1.0)
    points_per_A = 7.2 * 72 * 0.76 / span
    sizes = (np.asarray([reference.VDW_A[element] for element in data["elements"]]) * points_per_A * 0.65) ** 2
    scatter = d.scatter(xyz[depth_order, 0], xyz[depth_order, 1], s=sizes[depth_order],
                         c=log_density[data["atom_to_pair"]][depth_order], cmap="plasma",
                         norm=Normalize(-0.5, 1.75, clip=True), edgecolors="none", linewidths=0,
                         alpha=0.70, zorder=2)
    d.set_xlim(xyz[:, 0].min() - span * 0.04, xyz[:, 0].max() + span * 0.04)
    d.set_ylim(xyz[:, 1].min() - 2.5, xyz[:, 1].max() + 2.5)
    d.set_aspect("equal", adjustable="box")
    # Evenly spaced labels with leaders preserve readability at bent/frayed ends.
    label_positions = np.linspace(centers[:, 0].min(), centers[:, 0].max(), n)
    d.set_xticks(label_positions, ticks, rotation=90, fontsize=5.8)
    for center, label_position in zip(centers[:, 0], label_positions):
        d.plot([center, label_position], [0, -0.055], transform=d.get_xaxis_transform(),
               color="#b7bbc2", lw=0.35, clip_on=False)
    d.spines["bottom"].set_position(("axes", -0.055))
    d.set_yticks([])
    for spine in d.spines.values():
        spine.set_visible(False)
    d.tick_params(axis="x", length=2, color="#9a9a9a", labelrotation=90, labelsize=5.8)
    d.set_xlabel("Base-pair position / sequence" if len(labels[0]) == 2 else "Nucleotide position / sequence", labelpad=8)
    # The colourbar reports the underlying bp value, not opacity-composited pixels.
    colour_scale = matplotlib.cm.ScalarMappable(norm=scatter.norm, cmap=scatter.cmap)
    colorbar = figure.colorbar(colour_scale, cax=figure.add_axes([0.37, 0.043, 0.24, 0.012]),
                              orientation="horizontal", extend="both", ticks=[-0.5, 1.75])
    colorbar.set_label(r"log$_{10}$(cumulative DoS density)", fontsize=7, labelpad=2)
    colorbar.ax.tick_params(labelsize=7, pad=2)
    for label, axis in zip("abcd", axes):
        if label == "d":
            figure.text(0.05, grid[2, :2].get_position(figure).y1 + 0.015, label, fontsize=9, fontweight="bold")
        else:
            axis.text(-0.12 if label in "ab" else -0.065, 1.015 if label in "ab" else 1.055, label, transform=axis.transAxes,
                      fontsize=9, fontweight="bold", va="bottom")
    figure.text(0.10, 0.983, f"{n} positions  |  {data['metadata']['state_count']:,} states  |  "
                 f"gateway {data['gateway']['energy_eV']:.3f} eV  |  {data['temperature']:g} K", fontsize=8, va="top")
    return figure


def run_plot(results: Path, output: Path, gateway=None, temperature=None, display_sigma=0.005, bounds=(None, None)) -> dict:
    data = calculate(results, gateway, temperature, display_sigma, bounds)
    output.mkdir(parents=True, exist_ok=False)
    figure = draw_figure(data)
    for extension in ("png", "svg", "pdf", "tiff"):
        options = {"pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}
        figure.savefig(output / f"figure3_abcd.{extension}", dpi=600 if extension == "tiff" else 300, **options)
    plt.close(figure)
    columns = [f"bp_{index + 1:03d}" for index in range(len(data["labels"]))]
    workflow.write_csv(output / "panel_a_display.csv", ["energy_eV", *columns], np.column_stack([data["energies"], data["display"]]))
    relative = (data["energies"] - data["gateway"]["energy_eV"]) / (KB_EV * data["temperature"])
    zoom = np.abs(relative) <= 9
    workflow.write_csv(output / "panel_b_display.csv", ["energy_eV", "relative_kBT", *columns],
                       np.column_stack([data["energies"][zoom], relative[zoom], data["display"][zoom]]))
    workflow.write_csv(output / "panel_c_d_values.csv", ["position_1based", "bases", "volume_nm3", "integrated_states", "cumulative_states_eV-1_nm-3"],
                       ([i + 1, label, data["volumes"][i], data["integrated"][i], data["cumulative"][i]] for i, label in enumerate(data["labels"])))
    fields = ["energy_index", "energy_eV", "terminal_bp_dos", "left_contact_dos"]
    workflow.write_csv(output / "gateway_candidates.csv", fields, ([row[key] for key in fields] for row in data["candidates"]))
    workflow.write_csv(output / "contact_dos.csv", ["energy_eV", "left_contact_dos_eV-1", "right_contact_dos_eV-1"],
                       np.column_stack([data["energies"], data["contact"]]))
    report = {"run_id": data["metadata"]["run_id"], "state_count": data["metadata"]["state_count"],
              "base_pair_count": len(data["labels"]), "gateway": data["gateway"], "temperature_K": data["temperature"],
              "display_gaussian_sigma_eV": display_sigma, "quantification_uses_display_smoothing": False,
              "panel_c_max": float(data["cumulative"].max()), "panel_d_colour_unit": "base pair (nucleotide for ssDNA)",
              "panel_d_view": "orthographic PCA projection; all atoms inherit their position value",
              "panel_d_style": "translucent atomic dots; alpha=0.70; no bonds, outlines, jitter or spatial smoothing",
              "panel_d_log10_colour_limits": [-0.5, 1.75],
              "gateway_window_eV": [data["gateway"]["energy_eV"] - 3 * KB_EV * data["temperature"],
                                    data["gateway"]["energy_eV"] + 3 * KB_EV * data["temperature"]],
              "volume_convention": "sum of Bondi atomic sphere volumes; not the molecular union volume",
              "panel_c_convention": "integrated states / reporting energy bin width / bp volume",
              "single_conformation": True, "preview": str((output / "figure3_abcd.png").resolve())}
    workflow.save_json(output / "figure_metadata.json", report)
    (output / "FIGURE_CAPTION.md").write_text(
        "# Figure 3-style DNA DoS\n\n"
        f"One conformation, {len(data['labels'])} positions, {data['metadata']['state_count']} full-space states. "
        f"Gateway = {data['gateway']['energy_eV']:.5f} eV; T = {data['temperature']:g} K.\n\n"
        "a, Position-resolved DoS. b, Gateway-centred view; dashed lines mark +/-3 kBT. "
        "c, Integral over +/-3 kBT, divided by the reporting energy bin width and the sum of Bondi atomic sphere volumes. "
        "d, The same values on the structure; all atoms of each bp share one quantitative colour.\n\n"
        f"Display-only energy smoothing: sigma={display_sigma:g} eV; bilinear image interpolation. "
        "Neither affects c/d quantification. Structure shown by an orthographic PCA projection, "
        "with translucent atomic dots (alpha=0.70), no bonds or outlines, and no spatial smoothing or jitter. "
        "Opacity overlap changes the displayed shade, not the assigned bp value. "
        "The log10 colour range is fixed at [-0.5, 1.75]; out-of-range values are saturated. "
        "No ensemble statistics or inference of conductance from DoS alone.\n", encoding="utf-8")
    print(f"Figure 3a-d exported: {output}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gateway", type=float)
    parser.add_argument("--temperature", type=float)
    args = parser.parse_args()
    run_plot(args.results, args.output, args.gateway, args.temperature)
