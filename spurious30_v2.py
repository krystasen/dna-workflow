"""Program-independent SPURIOUS30 V2 mathematics for localized DNA Hamiltonians."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ENERGY_GRID_EV = (-6.0, 0.0, 1201)
ETA_EV = 0.012
DOS_BIN_WIDTH_EV = 0.024
GAMMA_B_EV = 0.01
LAMBDA_EV = 0.05
SPURIOUS_ENERGY_WINDOW_EV = (-6.05, -5.65)
COVALENT_RADII_A = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "P": 1.07}


def partition_indices(entry: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [np.arange(int(first) - 1, int(last), dtype=np.int64) for first, last in entry["ao_ranges"]]
    )


def load_geometry(path: Path) -> list[tuple[str, np.ndarray]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    orca = next(
        (i for i, line in enumerate(lines) if re.match(r"\s*\*\s*xyz\s+-?\d+\s+\d+", line, re.I)),
        None,
    )
    if orca is not None:
        start, stop_at_star = orca + 1, True
    else:
        charge = next(
            (i for i, line in enumerate(lines) if re.fullmatch(r"\s*-?\d+\s+\d+\s*", line)),
            None,
        )
        if charge is None:
            raise ValueError(f"Cannot locate ORCA or Gaussian geometry in {path}.")
        start, stop_at_star = charge + 1, False

    atoms = []
    for line in lines[start:]:
        if stop_at_star and line.strip().startswith("*"):
            break
        fields = line.split()
        if len(fields) < 4:
            if atoms:
                break
            continue
        element = fields[0].upper()
        if element not in COVALENT_RADII_A:
            raise ValueError(f"Unsupported element {element} in {path}.")
        atoms.append((element, np.asarray(fields[1:4], dtype=float)))
    if not atoms:
        raise ValueError(f"No atoms read from {path}.")
    return atoms


def identify_spurious_states(
    entries: list[dict[str, Any]],
    atoms: list[tuple[str, np.ndarray]],
    onsite: np.ndarray,
    hamiltonian: np.ndarray,
    energy_window: tuple[float, float] = SPURIOUS_ENERGY_WINDOW_EV,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for entry in entries:
        match = re.fullmatch(r"(s\d+)_(\d+)_(A|C|G|T)", entry["name"])
        if not match:
            raise ValueError(f"Cannot identify strand position from {entry['name']}.")
        grouped.setdefault(match.group(1), []).append((int(match.group(2)), entry))

    records = []
    for strand, positioned in sorted(grouped.items()):
        positioned.sort(key=lambda item: item[0])
        for (_, previous), (_, current) in zip(positioned, positioned[1:]):
            previous_first, previous_last = map(int, previous["atom_range_1based"])
            current_first, current_last = map(int, current["atom_range_1based"])
            cross_bonds = []
            for atom_a in range(previous_first, previous_last + 1):
                element_a, coordinate_a = atoms[atom_a - 1]
                for atom_b in range(current_first, current_last + 1):
                    element_b, coordinate_b = atoms[atom_b - 1]
                    distance = float(np.linalg.norm(coordinate_a - coordinate_b))
                    if distance < 1.25 * (
                        COVALENT_RADII_A[element_a] + COVALENT_RADII_A[element_b]
                    ):
                        cross_bonds.append((atom_a, element_a, atom_b, element_b, distance))
            if len(cross_bonds) != 1 or {cross_bonds[0][1], cross_bonds[0][3]} != {"O", "P"}:
                raise ValueError(
                    f"Expected one O-P partition bond at {previous['name']}->{current['name']}; "
                    f"found {cross_bonds}."
                )

            previous_indices = partition_indices(previous)
            current_indices = partition_indices(current)
            coupling_norms = np.linalg.norm(
                np.asarray(hamiltonian[np.ix_(current_indices, previous_indices)]), axis=1
            )
            current_onsite = onsite[current_indices]
            candidates = np.flatnonzero(
                (current_onsite >= energy_window[0]) & (current_onsite <= energy_window[1])
            )
            if not candidates.size:
                raise ValueError(f"No dangling-bond candidate in {current['name']}.")
            ranked = candidates[np.argsort(coupling_norms[candidates])[::-1]]
            candidate = int(ranked[0])
            runner_up = float(coupling_norms[ranked[1]]) if ranked.size > 1 else 0.0
            configured = current.get("exclude_probe_local", [])
            if configured and configured != [candidate + 1]:
                raise ValueError(
                    f"Configured exclusion for {current['name']} is {configured}; "
                    f"identified {[candidate + 1]}."
                )
            current["exclude_probe_local"] = [candidate + 1]
            atom_a, element_a, atom_b, element_b, distance = cross_bonds[0]
            records.append(
                {
                    "boundary": f"{previous['name']}->{current['name']}",
                    "nucleotide": current["name"],
                    "local_index_1based": candidate + 1,
                    "global_index_0based": int(current_indices[candidate]),
                    "global_index_1based": int(current_indices[candidate] + 1),
                    "onsite_eV": float(current_onsite[candidate]),
                    "coupling_norm_to_previous_eV": float(coupling_norms[candidate]),
                    "runner_up_coupling_norm_eV": runner_up,
                    "coupling_margin_ratio": float(coupling_norms[candidate] / runner_up)
                    if runner_up
                    else None,
                    "boundary_atoms_1based": [atom_a, atom_b],
                    "boundary_elements": f"{element_a}-{element_b}",
                    "boundary_distance_A": distance,
                    "identification": "maximum previous-block coupling in the configured energy window",
                }
            )
    expected = sum(len(items) - 1 for items in grouped.values())
    if len(records) != expected or len({row["global_index_0based"] for row in records}) != expected:
        raise ValueError(f"Expected {expected} unique dangling-bond states; found {len(records)}.")
    return records


def partition_probe_mask(
    entries: list[dict[str, Any]], size: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    mask = np.zeros(size, dtype=bool)
    excluded = []
    for entry in entries:
        indices = partition_indices(entry)
        if entry.get("probe", True):
            mask[indices] = True
        local = np.asarray(entry.get("exclude_probe_local", []), dtype=np.int64)
        if local.size:
            if np.any(local < 1) or np.any(local > indices.size) or np.unique(local).size != local.size:
                raise ValueError(f"Invalid exclude_probe_local in {entry['name']}: {local.tolist()}.")
            global_indices = indices[local - 1]
            mask[global_indices] = False
            excluded.extend(
                {
                    "nucleotide": entry["name"],
                    "local_index_1based": int(local_index),
                    "global_index_0based": int(global_index),
                }
                for local_index, global_index in zip(local, global_indices)
            )
    return mask, excluded


def analytic_sigma_b(
    energies: np.ndarray,
    onsite: np.ndarray,
    probe_mask: np.ndarray,
    gamma_b_eV: float = GAMMA_B_EV,
    lambda_eV: float = LAMBDA_EV,
) -> np.ndarray:
    from scipy.special import expi

    output = np.empty((energies.size, onsite.size), dtype=np.complex128)
    for start in range(0, onsite.size, 256):
        stop = min(start + 256, onsite.size)
        delta = (energies[:, None] - onsite[None, start:stop]) / lambda_eV
        gamma = gamma_b_eV * np.exp(-np.abs(delta))
        gamma[:, ~probe_mask[start:stop]] = 0.0
        shape = np.zeros_like(delta)
        middle = (np.abs(delta) > 1e-12) & (np.abs(delta) <= 50)
        shape[middle] = (
            -np.exp(-delta[middle]) * expi(delta[middle])
            + np.exp(delta[middle]) * expi(-delta[middle])
        )
        far = np.abs(delta) > 50
        shape[far] = -2 / delta[far] - 4 / delta[far] ** 3 - 48 / delta[far] ** 5
        real = -0.5 * gamma_b_eV * shape / np.pi
        real[:, ~probe_mask[start:stop]] = 0.0
        output[:, start:stop] = real - 0.5j * gamma
    return output


def bin_average_dos(
    energies: np.ndarray, values: np.ndarray, bin_width: float = DOS_BIN_WIDTH_EV
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if energies.ndim != 1 or values.shape[0] != energies.size or np.any(np.diff(energies) <= 0):
        raise ValueError("DoS binning requires a strictly increasing energy grid.")
    if bin_width <= 0 or np.max(np.diff(energies)) > bin_width / 2:
        raise ValueError("Energy spacing must not exceed half the requested DoS bin width.")
    count = int(round((energies[-1] - energies[0]) / bin_width))
    edges = energies[0] + np.arange(count + 1) * bin_width
    if not np.isclose(edges[-1], energies[-1], atol=1e-10):
        raise ValueError("Energy range must contain an integer number of DoS bins.")
    edges[-1] = energies[-1]
    centers = (edges[:-1] + edges[1:]) / 2
    flat = values.reshape(values.shape[0], -1)
    averaged = np.empty((count, flat.shape[1]), dtype=np.float64)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        internal = (energies > left) & (energies < right)
        x = np.concatenate(([left], energies[internal], [right]))
        y = np.vstack(
            ([np.interp(left, energies, flat[:, column]) for column in range(flat.shape[1])],
             flat[internal],
             [np.interp(right, energies, flat[:, column]) for column in range(flat.shape[1])])
        )
        averaged[index] = np.trapezoid(y, x, axis=0) / (right - left)
    if not np.allclose(
        np.trapezoid(flat, energies, axis=0),
        np.sum(averaged * np.diff(edges)[:, None], axis=0),
        rtol=1e-11,
        atol=1e-11,
    ):
        raise RuntimeError("DoS binning did not conserve integrated spectral weight.")
    return centers, np.diff(edges), averaged.reshape((count, *values.shape[1:]))


def prepare_auxiliary(args: argparse.Namespace) -> None:
    entries = json.loads(args.partition.read_text(encoding="utf-8"))["nucleotides"]
    onsite = np.load(args.onsite)
    hamiltonian = np.load(args.hamiltonian, mmap_mode="r")
    records = identify_spurious_states(entries, load_geometry(args.geometry), onsite, hamiltonian)
    mask = np.load(args.base_probe_mask).copy()
    excluded = np.asarray([row["global_index_0based"] for row in records], dtype=np.int64)
    mask[excluded] = False
    energies = np.linspace(*ENERGY_GRID_EV)
    sigma = analytic_sigma_b(energies, onsite, mask, args.gamma_b, args.lambda_e)
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "energies_eV.npy", energies)
    np.save(args.output / "probe_mask.npy", mask)
    np.save(args.output / "excluded_probe_indices.npy", excluded)
    np.save(args.output / "sigma_b_eV.npy", sigma)
    (args.output / "partitions_spurious30_v2.json").write_text(
        json.dumps({"index_base": 1, "nucleotides": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output / "spurious_states.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"state_count": onsite.size, "spurious_count": len(records), "probe_count": int(mask.sum())}, indent=2))


def self_test() -> None:
    energies = np.linspace(-0.12, 0.12, 97)
    values = np.column_stack([np.ones(energies.size), energies])
    centers, widths, averaged = bin_average_dos(energies, values)
    assert np.allclose(averaged[:, 0], 1.0)
    assert np.allclose(averaged[:, 1], centers)
    assert np.isclose(np.sum(averaged[:, 0] * widths), 0.24)
    sigma = analytic_sigma_b(np.asarray([-0.05, 0.0, 0.05]), np.asarray([0.0]), np.asarray([True]))[:, 0]
    assert np.isclose(sigma[1].real, 0.0) and np.isclose(sigma[1].imag, -0.005)
    assert np.isclose(sigma[0].real, -sigma[2].real)
    print("PASS: program-independent analytic KK and integral-conserving DoS binning.")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Program-independent SPURIOUS30 V2 auxiliary preparation.")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--hamiltonian", type=Path, required=True)
    prepare.add_argument("--onsite", type=Path, required=True)
    prepare.add_argument("--partition", type=Path, required=True)
    prepare.add_argument("--geometry", type=Path, required=True)
    prepare.add_argument("--base-probe-mask", type=Path, required=True)
    prepare.add_argument("--gamma-b", type=float, default=GAMMA_B_EV)
    prepare.add_argument("--lambda-e", type=float, default=LAMBDA_EV)
    prepare.add_argument("--output", type=Path, required=True)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    self_test() if arguments.command == "self-test" else prepare_auxiliary(arguments)
