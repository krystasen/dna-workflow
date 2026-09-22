"""Gaussian F/S 到 11660 态二维 DoS 的生产实现。

正式协议见 ``PYTORCH_GAUSSIAN_11660_SPURIOUS30_V2_GUIDE.md``；旧版 601 点
指南仅保留作历史对照。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import shutil
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

import spurious30_v2 as common_v2


SCRIPT_NAME = "gaussian_full_dos_11660.py"
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
I4 = struct.Struct("<i")
I8 = struct.Struct("<q")
PARAMETERS = {
    "energy_grid_eV": [-6.0, 0.0, 1201],
    "eta_eV": 0.012,
    "dos_bin_width_eV": 0.024,
    "gamma_contact_eV": 0.6,
    "gamma_b_eV": 0.01,
    "lambda_eV": 0.05,
    "contact_model": "3prime-O orthogonal-AO projector",
    "left_contact_nucleotide": "s2_16_A",
    "right_contact_nucleotide": "s1_16_A",
}
SPURIOUS_ENERGY_WINDOW_EV = common_v2.SPURIOUS_ENERGY_WINDOW_EV


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def marker(handle) -> int:
    raw = handle.read(4)
    if len(raw) != 4:
        raise EOFError
    return I4.unpack(raw)[0]


def read_record(handle) -> bytes:
    size = marker(handle)
    data = handle.read(size)
    end = marker(handle)
    if len(data) != size or end != size:
        raise ValueError(f"Fortran record mismatch near byte {handle.tell()}: {size} != {end}.")
    return data


def skip_record(handle) -> tuple[int, int]:
    size = marker(handle)
    offset = handle.tell()
    handle.seek(size, 1)
    end = marker(handle)
    if end != size:
        raise ValueError(f"Fortran record mismatch near byte {handle.tell()}: {size} != {end}.")
    return offset, size


def text64(raw: bytes) -> str:
    return raw[:64].decode("ascii", errors="replace").rstrip(" \x00")


def scan_mat(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    with path.open("rb") as handle:
        first = read_record(handle)
        if len(first) != 144 or text64(first) != "Gaussian matrix elements":
            raise ValueError("Input is not a supported Gaussian matrix-element file.")
        version, nlab = struct.unpack_from("<qq", first, 64)
        program = text64(first[80:])
        second = read_record(handle)
        if len(second) < 136:
            raise ValueError("Gaussian system header is truncated.")
        title = text64(second)
        values = struct.unpack_from("<9q", second, 64)
        natoms, nbasis, nbsuse, charge, multiplicity, electrons, len12, len4, iopcl = values
        if version != 2 or len12 != 8:
            raise ValueError(f"Unsupported Gaussian MAT variant: version={version}, Len12L={len12}.")
        labels = [read_record(handle) for _ in range(2, nlab)]
        if len(labels) < 5:
            raise ValueError("Gaussian MAT lacks AO-to-atom metadata.")
        atomic_numbers = np.frombuffer(labels[0], dtype="<i8", count=natoms).astype(np.int64)
        coordinates = (
            np.frombuffer(labels[3], dtype="<f8", count=3 * natoms)
            .reshape(natoms, 3)
            .copy()
            * BOHR_TO_ANGSTROM
        )
        ao_record = np.frombuffer(labels[4], dtype="<i8")
        if ao_record.size < 2 * nbasis:
            raise ValueError("Gaussian AO metadata record is shorter than 2*NBasis.")
        ao_to_atom = ao_record[:nbasis].astype(np.int64)
        ao_types = ao_record[nbasis : 2 * nbasis].astype(np.int64)
        header = {
            "label": text64(first),
            "version": int(version),
            "program": program,
            "title": title,
            "n_atoms": int(natoms),
            "n_basis": int(nbasis),
            "n_basis_used": int(nbsuse),
            "charge": int(charge),
            "multiplicity": int(multiplicity),
            "electrons": int(electrons),
            "len12": int(len12),
            "len4": int(len4),
            "iopcl": int(iopcl),
            "atomic_numbers": atomic_numbers,
            "coordinates_A": coordinates,
            "ao_to_atom_1based": ao_to_atom,
            "ao_types": ao_types,
        }
        matrices: dict[str, dict[str, Any]] = {}
        while True:
            try:
                raw = read_record(handle)
            except EOFError:
                break
            if len(raw) < 136:
                raise ValueError(f"Short Gaussian one-electron header near byte {handle.tell()}.")
            name = text64(raw)
            ni, nr, ntot, lenbuf, n1, n2, n3, n4, n5 = struct.unpack_from("<9q", raw, 64)
            info = {
                "label": name,
                "ni": int(ni),
                "nr": int(nr),
                "ntot": int(ntot),
                "lenbuf": int(lenbuf),
                "shape_fields": [int(n1), int(n2), int(n3), int(n4), int(n5)],
                "records": [],
            }
            if name == "END":
                break
            if lenbuf <= 0:
                raise ValueError(f"Invalid LenBuf for Gaussian matrix {name!r}.")
            remaining = int(ntot)
            while remaining > 0:
                offset, size = skip_record(handle)
                valid = min(int(lenbuf), remaining)
                info["records"].append((offset, size, valid))
                remaining -= valid
            matrices[name] = info
    return header, matrices


def matrix_summary(header: dict[str, Any], matrices: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        key: value
        for key, value in header.items()
        if key not in {"atomic_numbers", "coordinates_A", "ao_to_atom_1based", "ao_types"}
    } | {
        "matrices": {
            name: {
                "ntot": info["ntot"],
                "shape_fields": info["shape_fields"],
                "ni": info["ni"],
                "nr": info["nr"],
            }
            for name, info in matrices.items()
        }
    }


def valid_npy(path: Path, shape: tuple[int, ...]) -> bool:
    if not path.exists():
        return False
    try:
        return np.load(path, mmap_mode="r").shape == shape
    except (OSError, ValueError):
        return False


def extract_real_matrix(
    mat_path: Path,
    info: dict[str, Any],
    destination: Path,
    shape: tuple[int, ...],
    fortran_order: bool = False,
) -> np.memmap:
    if valid_npy(destination, shape):
        print(f"reuse {destination.name}", flush=True)
        return np.load(destination, mmap_mode="r+")
    if info["ni"] != 0 or info["nr"] != 1 or int(np.prod(shape)) != info["ntot"]:
        raise ValueError(f"Unexpected storage for {info['label']}: {info}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = open_memmap(destination, mode="w+", dtype=np.float64, shape=shape, fortran_order=fortran_order)
    flat = output.reshape(-1, order="F" if fortran_order else "C")
    cursor = 0
    with mat_path.open("rb") as handle:
        for record_index, (offset, size, valid) in enumerate(info["records"], 1):
            handle.seek(offset)
            values = np.fromfile(handle, dtype="<f8", count=size // 8)
            if values.size < valid:
                raise ValueError(f"Truncated data record in {info['label']}.")
            flat[cursor : cursor + valid] = values[:valid]
            cursor += valid
            if record_index % 5000 == 0:
                print(f"{info['label']}: {cursor}/{info['ntot']}", flush=True)
    output.flush()
    if cursor != info["ntot"]:
        raise ValueError(f"Extracted {cursor} values from {info['label']}; expected {info['ntot']}.")
    return output


def expand_packed_lower(packed_path: Path, destination: Path, size: int) -> np.memmap:
    if valid_npy(destination, (size, size)):
        print(f"reuse {destination.name}", flush=True)
        return np.load(destination, mmap_mode="r+")
    packed = np.load(packed_path, mmap_mode="r")
    if packed.shape != (size * (size + 1) // 2,):
        raise ValueError(f"Packed matrix has wrong size: {packed.shape}.")
    output = open_memmap(destination, mode="w+", dtype=np.float64, shape=(size, size))
    for row in range(size):
        start = row * (row + 1) // 2
        values = packed[start : start + row + 1]
        output[row, : row + 1] = values
        output[:row, row] = values[:row]
        if (row + 1) % 1000 == 0:
            print(f"unpack {destination.name}: {row + 1}/{size}", flush=True)
    output.flush()
    return output


def block_symmetry_error(matrix: np.ndarray, block: int = 512) -> float:
    error = 0.0
    for start in range(0, matrix.shape[0], block):
        stop = min(start + block, matrix.shape[0])
        error = max(error, float(np.max(np.abs(matrix[start:stop] - matrix[:, start:stop].T))))
    return error


def symmetrize_in_place(matrix: np.memmap, block: int = 512) -> None:
    size = matrix.shape[0]
    for row_start in range(0, size, block):
        row_stop = min(row_start + block, size)
        diagonal = np.asarray(matrix[row_start:row_stop, row_start:row_stop])
        matrix[row_start:row_stop, row_start:row_stop] = (diagonal + diagonal.T) / 2
        for col_start in range(0, row_start, block):
            col_stop = min(col_start + block, size)
            lower = np.asarray(matrix[row_start:row_stop, col_start:col_stop])
            upper_t = np.asarray(matrix[col_start:col_stop, row_start:row_stop]).T
            average = (lower + upper_t) / 2
            matrix[row_start:row_stop, col_start:col_stop] = average
            matrix[col_start:col_stop, row_start:row_stop] = average.T
    matrix.flush()


def frobenius_squared(matrix: np.ndarray, block: int = 512) -> float:
    total = 0.0
    for start in range(0, matrix.shape[0], block):
        values = np.asarray(matrix[start : start + block])
        total += float(np.sum(values * values))
    return total


def gaussian_geometry(path: Path) -> list[tuple[str, np.ndarray]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    charge_line = next(
        index for index, line in enumerate(lines) if re.fullmatch(r"\s*-?\d+\s+\d+\s*", line)
    )
    atoms: list[tuple[str, np.ndarray]] = []
    for line in lines[charge_line + 1 :]:
        fields = line.split()
        if len(fields) < 4:
            break
        atoms.append((fields[0], np.asarray(fields[1:4], dtype=float)))
    return atoms


def neighbors(
    atoms: list[tuple[str, np.ndarray]], atom_index_1based: int, first: int, last: int
) -> list[tuple[int, str, float]]:
    center_element, center = atoms[atom_index_1based - 1]
    found = []
    for other in range(first, last + 1):
        if other == atom_index_1based:
            continue
        element, coordinate = atoms[other - 1]
        distance = float(np.linalg.norm(center - coordinate))
        if distance < 1.25 * (RADII[center_element] + RADII[element]):
            found.append((other, element, distance))
    return found


def terminal_oxygen(
    atoms: list[tuple[str, np.ndarray]], nucleotide: dict[str, Any]
) -> tuple[int, list[tuple[int, str, float]]]:
    first, last = map(int, nucleotide["atom_range_1based"])
    candidates = []
    for atom_index in range(first, last + 1):
        if atoms[atom_index - 1][0] != "O":
            continue
        bonded = neighbors(atoms, atom_index, first, last)
        if {"H", "C"}.issubset({item[1] for item in bonded}):
            candidates.append((atom_index, bonded))
    if len(candidates) != 1:
        raise ValueError(f"Expected one terminal O-H/C anchor in {nucleotide['name']}; found {candidates}.")
    return candidates[0]


def gaussian_partition(
    template_path: Path, header: dict[str, Any], geometry_path: Path
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    ao_to_atom = np.asarray(header["ao_to_atom_1based"], dtype=np.int64)
    natoms = int(header["n_atoms"])
    nbasis = int(header["n_basis"])
    if ao_to_atom.shape != (nbasis,) or np.any((ao_to_atom < 1) | (ao_to_atom > natoms)):
        raise ValueError("Gaussian AO-to-atom map is invalid.")
    counts = np.bincount(ao_to_atom, minlength=natoms + 1)[1:]
    expected = np.repeat(np.arange(1, natoms + 1), counts)
    if not np.array_equal(ao_to_atom, expected):
        raise ValueError("Gaussian AOs are not contiguous by atom; atom-range partitioning would be invalid.")
    prefix = np.cumsum(np.r_[0, counts])
    atoms = gaussian_geometry(geometry_path)
    if len(atoms) != natoms:
        raise ValueError(f"Geometry has {len(atoms)} atoms; MAT has {natoms}.")
    element_z = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15}
    geometry_numbers = np.asarray([element_z[item[0]] for item in atoms])
    if not np.array_equal(geometry_numbers, header["atomic_numbers"]):
        raise ValueError("Gaussian geometry element order differs from the MAT atom order.")
    coordinate_error = float(np.max(np.abs(np.asarray([item[1] for item in atoms]) - header["coordinates_A"])))
    if coordinate_error > 1e-6:
        raise ValueError(f"Gaussian geometry/MAT coordinates differ by {coordinate_error:.3e} Angstrom.")
    entries = []
    for source in template["nucleotides"]:
        first, last = map(int, source["atom_range_1based"])
        entry = {
            key: value
            for key, value in source.items()
            if key not in {"ao_ranges", "contact", "exclude_probe_local"}
        }
        entry["ao_ranges"] = [[int(prefix[first - 1] + 1), int(prefix[last])]]
        entry["probe"] = source["name"] not in {
            PARAMETERS["left_contact_nucleotide"],
            PARAMETERS["right_contact_nucleotide"],
        }
        entries.append(entry)
    partition = {
        "index_base": 1,
        "source": "Gaussian MAT AO-to-atom map plus ORCA-validated nucleotide atom ranges",
        "pairing": template.get("pairing"),
        "strand_sequences_5to3": template.get("strand_sequences_5to3"),
        "contact_assumption": {
            "model": PARAMETERS["contact_model"],
            "left": PARAMETERS["left_contact_nucleotide"],
            "right": PARAMETERS["right_contact_nucleotide"],
        },
        "nucleotides": entries,
    }
    anchors = {}
    for name in (PARAMETERS["left_contact_nucleotide"], PARAMETERS["right_contact_nucleotide"]):
        nucleotide = next(item for item in entries if item["name"] == name)
        atom_index, bonded = terminal_oxygen(atoms, nucleotide)
        anchors[name] = {
            "anchor_atom_1based": atom_index,
            "anchor_ao_range_0based": [int(prefix[atom_index - 1]), int(prefix[atom_index])],
            "bonded_atoms": [
                {"atom_1based": i, "element": e, "distance_A": d} for i, e, d in bonded
            ],
        }
    diagnostics = {
        "atom_ao_prefix_0based": prefix.tolist(),
        "geometry_mat_coordinate_max_abs_A": coordinate_error,
        "anchors": anchors,
    }
    return partition, prefix, diagnostics


def partition_indices(entry: dict[str, Any]) -> np.ndarray:
    return common_v2.partition_indices(entry)


def partition_probe_mask(
    entries: list[dict[str, Any]], size: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    return common_v2.partition_probe_mask(entries, size)


def identify_spurious_states(
    entries: list[dict[str, Any]],
    atoms: list[tuple[str, np.ndarray]],
    onsite: np.ndarray,
    hamiltonian: np.ndarray,
) -> list[dict[str, Any]]:
    return common_v2.identify_spurious_states(entries, atoms, onsite, hamiltonian)


def display_labels(entries: list[dict[str, Any]], order: list[str]) -> list[str]:
    labels = []
    for pair in order:
        bases = {}
        for entry in entries:
            if str(entry["base_pair"]) == str(pair):
                fields = entry["name"].split("_")
                bases[fields[0]] = fields[-1]
        labels.append(bases.get("s1", "") + bases.get("s2", "") or str(pair))
    return labels


def kramers_kronig_grid(energies: np.ndarray, onsite: np.ndarray, probe_mask: np.ndarray) -> np.ndarray:
    return common_v2.analytic_sigma_b(
        energies, onsite, probe_mask, PARAMETERS["gamma_b_eV"], PARAMETERS["lambda_eV"]
    )


def bin_average_dos(
    energies: np.ndarray, values: np.ndarray, bin_width: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return common_v2.bin_average_dos(energies, values, bin_width)


def prepare_package(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        print(f"Package already prepared: {args.output}", flush=True)
        return
    work = args.work or args.output / "_prepare_work"
    work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("Scanning Gaussian MAT records", flush=True)
    header, matrices = scan_mat(args.mat)
    size = int(header["n_basis"])
    required = {"OVERLAP", "ORTHOGONAL BASIS", "ALPHA FOCK MATRIX", "ALPHA ORBITAL ENERGIES"}
    missing = required - matrices.keys()
    if missing:
        raise ValueError(f"Gaussian MAT lacks required arrays: {sorted(missing)}.")
    partition, atom_prefix, partition_diagnostics = gaussian_partition(
        args.partition_template, header, args.geometry
    )
    (args.output / "gaussian_partitions.json").write_text(
        json.dumps(partition, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    overlap_packed_path = work / "overlap_packed.npy"
    overlap_packed = extract_real_matrix(
        args.mat, matrices["OVERLAP"], overlap_packed_path, (size * (size + 1) // 2,)
    )
    del overlap_packed
    overlap = expand_packed_lower(overlap_packed_path, work / "overlap.npy", size)
    gaussian_orthogonal = extract_real_matrix(
        args.mat, matrices["ORTHOGONAL BASIS"], work / "orthogonal_basis.npy", (size, size), True
    )
    rng = np.random.default_rng(20260831)
    probes = rng.normal(size=(size, 3))
    transformed = gaussian_orthogonal.T @ (overlap @ (gaussian_orthogonal @ probes))
    gaussian_orthogonal_identity_error = float(
        np.linalg.norm(transformed - probes) / np.linalg.norm(probes)
    )
    gaussian_orthogonal_symmetry_error = block_symmetry_error(gaussian_orthogonal)
    if gaussian_orthogonal_identity_error > 1e-8:
        raise ValueError(
            "Gaussian ORTHOGONAL BASIS failed X.T @ S @ X = I: "
            f"relative error={gaussian_orthogonal_identity_error:.3e}."
        )
    lowdin_path = work / "lowdin_inverse_sqrt.npy"
    if valid_npy(lowdin_path, (size, size)):
        lowdin = np.load(lowdin_path, mmap_mode="r+")
        overlap_eigenvalues = np.load(work / "overlap_eigenvalues.npy")
    else:
        from scipy.linalg import eigh

        print("Diagonalizing S to construct the symmetric Lowdin S^(-1/2)", flush=True)
        overlap_work = np.array(overlap, dtype=np.float64, order="F", copy=True)
        overlap_eigenvalues, overlap_eigenvectors = eigh(
            overlap_work, overwrite_a=True, check_finite=False, driver="evd"
        )
        if overlap_eigenvalues[0] <= 1e-12 * overlap_eigenvalues[-1]:
            raise ValueError(
                "S is singular or numerically unstable for full-space Lowdin orthogonalization: "
                f"min={overlap_eigenvalues[0]:.3e}, max={overlap_eigenvalues[-1]:.3e}."
            )
        np.save(work / "overlap_eigenvalues.npy", overlap_eigenvalues)
        inverse_roots = overlap_eigenvalues**-0.5
        lowdin = open_memmap(
            lowdin_path, mode="w+", dtype=np.float64, shape=(size, size), fortran_order=True
        )
        for start in range(0, size, 512):
            stop = min(start + 512, size)
            lowdin[start:stop] = (
                overlap_eigenvectors[start:stop] * inverse_roots[None, :]
            ) @ overlap_eigenvectors.T
            lowdin.flush()
            print(f"Lowdin rows {stop}/{size}", flush=True)
        del overlap_work, overlap_eigenvectors
    lowdin_probe = lowdin.T @ (overlap @ (lowdin @ probes))
    lowdin_identity_error = float(np.linalg.norm(lowdin_probe - probes) / np.linalg.norm(probes))
    lowdin_symmetry_error = block_symmetry_error(lowdin)
    if lowdin_identity_error > 1e-10 or lowdin_symmetry_error > 1e-10:
        raise ValueError(
            "Constructed Lowdin matrix failed validation: "
            f"identity={lowdin_identity_error:.3e}, symmetry={lowdin_symmetry_error:.3e}."
        )
    del lowdin_probe, overlap

    fock_packed_path = work / "fock_packed.npy"
    fock_packed = extract_real_matrix(
        args.mat, matrices["ALPHA FOCK MATRIX"], fock_packed_path, (size * (size + 1) // 2,)
    )
    del fock_packed
    fock = expand_packed_lower(fock_packed_path, work / "fock_hartree.npy", size)
    canonical = extract_real_matrix(
        args.mat, matrices["ALPHA ORBITAL ENERGIES"], work / "canonical_hartree.npy", (size,)
    )
    canonical_eV = np.asarray(canonical) * HARTREE_TO_EV
    np.save(args.output / "canonical_orbital_energies_eV.npy", canonical_eV)

    fx_path = work / "fock_times_x.npy"
    if valid_npy(fx_path, (size, size)):
        fx = np.load(fx_path, mmap_mode="r+")
    else:
        print("Computing F @ S^(-1/2)", flush=True)
        fx = open_memmap(fx_path, mode="w+", dtype=np.float64, shape=(size, size), fortran_order=True)
        np.matmul(fock, lowdin, out=fx)
        fx.flush()
    h0_path = work / "h0_eV.npy"
    if valid_npy(h0_path, (size, size)):
        h0 = np.load(h0_path, mmap_mode="r+")
    else:
        print("Computing S^(-1/2) @ F @ S^(-1/2)", flush=True)
        h0 = open_memmap(h0_path, mode="w+", dtype=np.float64, shape=(size, size), fortran_order=True)
        np.matmul(lowdin, fx, out=h0)
        h0 *= HARTREE_TO_EV
        symmetrize_in_place(h0)
    h0_symmetry_error = block_symmetry_error(h0)
    trace_error = float(abs(np.trace(h0) - canonical_eV.sum()))
    frobenius_error = float(abs(frobenius_squared(h0) - np.dot(canonical_eV, canonical_eV)))
    frobenius_relative_error = frobenius_error / max(float(np.dot(canonical_eV, canonical_eV)), 1.0)
    if h0_symmetry_error > 1e-7 or trace_error > 1e-4 or frobenius_relative_error > 1e-10:
        raise ValueError(
            "Orthogonal Hamiltonian is inconsistent with Gaussian orbital energies: "
            f"symmetry={h0_symmetry_error:.3e}, trace={trace_error:.3e} eV, "
            f"frobenius_rel={frobenius_relative_error:.3e}."
        )

    entries = partition["nucleotides"]
    vectors: dict[str, np.ndarray] = {}
    onsite = np.empty(size, dtype=np.float64)
    base_pair_order: list[str] = []
    for entry in entries:
        label = str(entry["base_pair"])
        if label not in base_pair_order:
            base_pair_order.append(label)
        indices = partition_indices(entry)
        block = np.asarray(h0[np.ix_(indices, indices)])
        values, block_vectors = np.linalg.eigh((block + block.T) / 2)
        onsite[indices] = values
        vectors[entry["name"]] = block_vectors
        print(f"localized {entry['name']}: {indices.size} states", flush=True)

    right_path = work / "h0_times_u.npy"
    if valid_npy(right_path, (size, size)):
        right = np.load(right_path, mmap_mode="r+")
    else:
        right = open_memmap(right_path, mode="w+", dtype=np.float64, shape=(size, size))
        for entry in entries:
            indices = partition_indices(entry)
            right[:, indices] = np.asarray(h0[:, indices]) @ vectors[entry["name"]]
            right.flush()

    local_path = work / "hamiltonian_eV.npy"
    local = open_memmap(local_path, mode="w+", dtype=np.float64, shape=(size, size))
    for entry in entries:
        indices = partition_indices(entry)
        local[indices, :] = vectors[entry["name"]].T @ np.asarray(right[indices, :])
        local.flush()
    symmetrize_in_place(local)
    diagonal_error = float(np.max(np.abs(np.diag(local) - onsite)))
    local_symmetry_error = block_symmetry_error(local)
    invariant_relative_error = abs(frobenius_squared(local) - frobenius_squared(h0)) / max(
        frobenius_squared(h0), 1.0
    )
    if diagonal_error > 1e-7 or local_symmetry_error > 1e-7 or invariant_relative_error > 1e-10:
        raise ValueError(
            "Localized Hamiltonian validation failed: "
            f"diagonal={diagonal_error:.3e}, symmetry={local_symmetry_error:.3e}, "
            f"frobenius_rel={invariant_relative_error:.3e}."
        )

    spurious_states = identify_spurious_states(
        entries, gaussian_geometry(args.geometry), onsite, local
    )
    (args.output / "gaussian_partitions.json").write_text(
        json.dumps(partition, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "spurious_states.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(spurious_states[0]))
        writer.writeheader()
        writer.writerows(spurious_states)

    base_pair_index = np.full(size, -1, dtype=np.int16)
    for column, pair in enumerate(base_pair_order):
        for entry in entries:
            if str(entry["base_pair"]) == pair:
                base_pair_index[partition_indices(entry)] = column
    if np.any(base_pair_index < 0):
        raise ValueError("Base-pair groups do not cover every Gaussian localized state.")
    np.save(args.output / "base_pair_index.npy", base_pair_index)
    np.save(args.output / "onsite_eV.npy", onsite)

    anchors = partition_diagnostics["anchors"]
    contacts = []
    for side, name in (
        ("left", PARAMETERS["left_contact_nucleotide"]),
        ("right", PARAMETERS["right_contact_nucleotide"]),
    ):
        entry = next(item for item in entries if item["name"] == name)
        indices = partition_indices(entry)
        atom_start, atom_stop = anchors[name]["anchor_ao_range_0based"]
        anchor_rows = np.flatnonzero((indices >= atom_start) & (indices < atom_stop))
        block_vectors = vectors[name]
        projector = block_vectors[anchor_rows, :].T @ block_vectors[anchor_rows, :]
        projector = (projector + projector.T) / 2
        eigenvalues = np.linalg.eigvalsh(projector)
        rank = int(np.count_nonzero(eigenvalues > 1e-8))
        np.save(args.output / f"contact_{side}_indices.npy", indices)
        np.save(args.output / f"contact_{side}_projector.npy", projector)
        contacts.append({
            "side": side,
            "nucleotide": name,
            "state_count": int(indices.size),
            "anchor_ao_count": int(anchor_rows.size),
            "projector_rank": rank,
            "projector_idempotency_error": float(np.linalg.norm(projector @ projector - projector)),
            "projector_min_eigenvalue": float(eigenvalues[0]),
            "projector_max_eigenvalue": float(eigenvalues[-1]),
        })

    probe_mask, excluded_probe_states = partition_probe_mask(entries, size)
    expected_spurious_states = len(entries) - 2
    if len(excluded_probe_states) != expected_spurious_states:
        print(
            f"WARNING: configured {len(excluded_probe_states)} exclude_probe_local states; "
            f"the nucleotide partition predicts {expected_spurious_states} dangling-bond states. "
            "Identify them before the final production run.",
            file=sys.stderr,
        )
    energies = np.linspace(*PARAMETERS["energy_grid_eV"])
    sigma_b = kramers_kronig_grid(energies, onsite, probe_mask)
    np.save(args.output / "energies_eV.npy", energies)
    np.save(args.output / "probe_mask.npy", probe_mask)
    np.save(
        args.output / "excluded_probe_indices.npy",
        np.asarray([item["global_index_0based"] for item in excluded_probe_states], dtype=np.int64),
    )
    np.save(args.output / "sigma_b_eV.npy", sigma_b)

    local.flush()
    destination_h = args.output / "hamiltonian_eV.npy"
    if destination_h.exists():
        destination_h.unlink()
    del local
    os.replace(local_path, destination_h)
    labels = display_labels(entries, base_pair_order)
    metadata = {
        "schema_version": 2,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "model": "GGCC Gaussian full localized-orbital space",
        "source_program": header["program"],
        "source_title": header["title"],
        "source_mat": str(args.mat),
        "source_geometry": str(args.geometry),
        "source_partition_template": str(args.partition_template),
        "state_count": size,
        "atom_count": int(header["n_atoms"]),
        "charge": int(header["charge"]),
        "multiplicity": int(header["multiplicity"]),
        "basis_note": "Gaussian 6-31G(d,p) uses Cartesian 6D here; ORCA comparison used spherical 5D.",
        "energy_grid_eV": PARAMETERS["energy_grid_eV"],
        "eta_eV": PARAMETERS["eta_eV"],
        "eta_role": "numerical Lorentzian half-width for the 0.024 eV resolved DoS",
        "dos_bin_width_eV": PARAMETERS["dos_bin_width_eV"],
        "contact_model": PARAMETERS["contact_model"],
        "contact_nucleotides": [PARAMETERS["left_contact_nucleotide"], PARAMETERS["right_contact_nucleotide"]],
        "contacts": contacts,
        "gamma_contact_eV": PARAMETERS["gamma_contact_eV"],
        "gamma_b_eV": PARAMETERS["gamma_b_eV"],
        "lambda_eV": PARAMETERS["lambda_eV"],
        "kramers_kronig_real": True,
        "kramers_kronig_method": "analytic infinite-domain transform of exp(-abs(E-epsilon)/lambda)",
        "contact_state_count": int(sum(item["state_count"] for item in contacts)),
        "expected_spurious_state_count": expected_spurious_states,
        "excluded_probe_state_count": len(excluded_probe_states),
        "excluded_probe_states": excluded_probe_states,
        "spurious_state_identification": spurious_states,
        "spurious_state_energy_window_eV": list(SPURIOUS_ENERGY_WINDOW_EV),
        "probe_state_count": int(probe_mask.sum()),
        "base_pair_order": base_pair_order,
        "base_pair_display_labels": labels,
        "hamiltonian_dtype": "float64",
        "sigma_b_dtype": "complex128",
        "mathematical_chain": [
            "Gaussian ORTHOGONAL BASIS is validation-only; X = S^(-1/2) is built from S",
            "H0 = X.T @ F @ X",
            "each nucleotide block: V_m.T @ H0_mm @ V_m = diag(epsilon_m)",
            "H_local = U.T @ H0 @ U, U = direct_sum(V_m)",
            "Re Sigma_B is the infinite-domain analytic KK transform of its exponential imaginary part",
            "G^r(E) = [(E+i*eta_resolution)I-H_local-Sigma_L-Sigma_R-Sigma_B(E)]^(-1)",
            "DoS_bp(E) = sum_{k in bp} -Im G^r_kk(E)/pi",
            "reported quantitative DoS is averaged into 0.024 eV energy bins",
        ],
        "validation": {
            "gaussian_orthogonalizer_identity_random_relative_error": gaussian_orthogonal_identity_error,
            "gaussian_orthogonalizer_symmetry_max_abs": gaussian_orthogonal_symmetry_error,
            "s_min_eigenvalue": float(overlap_eigenvalues[0]),
            "s_max_eigenvalue": float(overlap_eigenvalues[-1]),
            "s_condition_number": float(overlap_eigenvalues[-1] / overlap_eigenvalues[0]),
            "lowdin_identity_random_relative_error": lowdin_identity_error,
            "lowdin_symmetry_max_abs": lowdin_symmetry_error,
            "h0_symmetry_max_abs_eV": h0_symmetry_error,
            "h0_trace_vs_orbital_energies_abs_eV": trace_error,
            "h0_frobenius_invariant_relative_error": frobenius_relative_error,
            "localized_diagonal_max_abs_error_eV": diagonal_error,
            "localized_symmetry_max_abs_eV": local_symmetry_error,
            "localized_frobenius_invariant_relative_error": invariant_relative_error,
            "geometry_mat_coordinate_max_abs_A": partition_diagnostics["geometry_mat_coordinate_max_abs_A"],
        },
        "preparation_minutes": (time.perf_counter() - started) / 60,
        "model_boundary": (
            "The electronic Hamiltonian is Gaussian/IEFPCM. Contact and decoherence self-energies are the "
            "same fixed phenomenological assumptions used for the ORCA comparison."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    source_script = Path(__file__).resolve()
    destination_script = (args.output / SCRIPT_NAME).resolve()
    if source_script != destination_script:
        shutil.copy2(source_script, destination_script)
    common_script = Path(common_v2.__file__).resolve()
    common_destination = (args.output / common_script.name).resolve()
    if common_script != common_destination:
        shutil.copy2(common_script, common_destination)
    if args.guide:
        shutil.copy2(args.guide, args.output / args.guide.name)
    (args.output / "preparation_report.json").write_text(
        json.dumps(
            {
                "mat": matrix_summary(header, matrices),
                "atom_ao_prefix_0based": atom_prefix.tolist(),
                "partition_diagnostics": partition_diagnostics,
                "metadata": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.keep_work:
        del canonical, fock, fx, gaussian_orthogonal, h0, lowdin, right
        gc.collect()
        shutil.rmtree(work)
    print(f"Prepared {args.output}: {size} states, {energies.size} energies.", flush=True)


def load_metadata(data_dir: Path) -> dict[str, Any]:
    return json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))


def import_torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed in this Python environment.") from error
    return torch


def torch_dtypes(torch: Any, name: str) -> tuple[Any, Any]:
    if name == "complex64":
        return torch.float32, torch.complex64
    if name == "complex128":
        return torch.float64, torch.complex128
    raise ValueError(f"Unsupported dtype: {name}")


def inverse_diagonal_from_lu(torch: Any, lu: Any, pivots: Any, size: int, rhs_chunk: int) -> Any:
    diagonal = torch.empty(size, dtype=lu.dtype, device=lu.device)
    for start in range(0, size, rhs_chunk):
        stop = min(start + rhs_chunk, size)
        width = stop - start
        rows = torch.arange(start, stop, device=lu.device)
        columns = torch.arange(width, device=lu.device)
        rhs = torch.zeros((size, width), dtype=lu.dtype, device=lu.device)
        rhs[rows, columns] = 1
        solved = torch.linalg.lu_solve(lu, pivots, rhs)
        diagonal[start:stop] = solved[rows, columns]
    return diagonal


def scaled_residual(torch: Any, matrix: Any, lu: Any, pivots: Any) -> float:
    size = matrix.shape[0]
    rows = torch.as_tensor([0, size // 2, size - 1], device=matrix.device)
    columns = torch.arange(rows.numel(), device=matrix.device)
    rhs = torch.zeros((size, rows.numel()), dtype=matrix.dtype, device=matrix.device)
    rhs[rows, columns] = 1
    solved = torch.linalg.lu_solve(lu, pivots, rhs)
    numerator = torch.linalg.vector_norm(matrix @ solved - rhs)
    denominator = torch.linalg.matrix_norm(matrix, ord="fro") * torch.linalg.vector_norm(solved) + torch.linalg.vector_norm(rhs)
    return float((numerator / denominator).item())


def selected_energy_indices(args: argparse.Namespace, count: int) -> list[int]:
    indices = sorted(set(args.indices)) if args.indices else list(range(args.start, count if args.stop is None else args.stop))
    if not indices or indices[0] < 0 or indices[-1] >= count:
        raise ValueError(f"Energy indices must lie in 0..{count - 1}.")
    return indices


def probe_gpu(args: argparse.Namespace) -> None:
    torch = import_torch()
    metadata = load_metadata(args.data)
    size = int(metadata["state_count"])
    real_bytes, complex_bytes = ((4, 8) if args.dtype == "complex64" else (8, 16))
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"state_count={size}, dtype={args.dtype}, rhs_chunk={args.rhs_chunk}")
    print(f"resident_real_hamiltonian_GiB={size * size * real_bytes / 2**30:.3f}")
    print(f"one_complex_square_matrix_GiB={size * size * complex_bytes / 2**30:.3f}")
    print(f"one_rhs_chunk_GiB={size * args.rhs_chunk * complex_bytes / 2**30:.3f}")
    print("LU workspace is backend/GPU dependent; verify with a single energy point.")
    if torch.cuda.is_available():
        device = torch.device(args.device)
        free, total = torch.cuda.mem_get_info(device)
        print(f"gpu={torch.cuda.get_device_name(device)}")
        print(f"free_vram_GiB={free / 2**30:.2f}, total_vram_GiB={total / 2**30:.2f}")


def run_gpu(args: argparse.Namespace) -> None:
    torch = import_torch()
    metadata = load_metadata(args.data)
    size = int(metadata["state_count"])
    energies = np.load(args.data / "energies_eV.npy")
    sigma_b = np.load(args.data / "sigma_b_eV.npy", mmap_mode="r")
    base_pair_index = np.load(args.data / "base_pair_index.npy")
    labels = metadata["base_pair_display_labels"]
    if base_pair_index.shape != (size,) or sigma_b.shape != (energies.size, size):
        raise ValueError("Material dimensions do not match metadata.")
    indices_to_run = selected_energy_indices(args, energies.size)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    real_dtype, complex_dtype = torch_dtypes(torch, args.dtype)
    hamiltonian = torch.tensor(np.load(args.data / "hamiltonian_eV.npy", mmap_mode="r"), dtype=real_dtype, device=device)
    contact_data = []
    for side in ("left", "right"):
        contact_data.append((
            torch.as_tensor(np.load(args.data / f"contact_{side}_indices.npy"), dtype=torch.int64, device=device),
            torch.as_tensor(np.load(args.data / f"contact_{side}_projector.npy"), dtype=real_dtype, device=device),
        ))
    points = args.output / "points"
    points.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {size} states on {device}; {len(indices_to_run)} point(s), {args.dtype}.", flush=True)
    for position, energy_index in enumerate(indices_to_run, 1):
        point_path = points / f"energy_{energy_index:04d}.npz"
        if args.resume and point_path.exists():
            print(f"skip {energy_index}: {point_path.name}", flush=True)
            continue
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        matrix = hamiltonian.to(dtype=complex_dtype)
        matrix.neg_()
        shift = torch.as_tensor(
            (energies[energy_index] + 1j * float(metadata["eta_eV"])) - sigma_b[energy_index],
            dtype=complex_dtype,
            device=device,
        )
        matrix.diagonal().add_(shift)
        for indices, projector in contact_data:
            matrix[indices[:, None], indices[None, :]] += 0.5j * float(metadata["gamma_contact_eV"]) * projector
        lu, pivots, info = torch.linalg.lu_factor_ex(matrix, check_errors=False)
        if int(info.item()) != 0:
            raise RuntimeError(f"LU failed at energy index {energy_index}: info={info.item()}.")
        residual = np.nan
        if energy_index in {0, energies.size // 6, energies.size // 2, energies.size - 1}:
            residual = scaled_residual(torch, matrix, lu, pivots)
            limit = 5e-5 if args.dtype == "complex64" else 1e-10
            if not np.isfinite(residual) or residual > limit:
                raise RuntimeError(f"Scaled residual {residual:.3e} exceeds {limit:.1e} at {energy_index}.")
        del matrix
        green_diagonal = inverse_diagonal_from_lu(torch, lu, pivots, size, args.rhs_chunk)
        raw_ldos = -green_diagonal.imag / np.pi
        raw_min = float(raw_ldos.min().item())
        raw_scale = max(float(raw_ldos.abs().max().item()), 1.0)
        negative_limit = (1e-5 if args.dtype == "complex64" else 1e-10) * raw_scale
        if raw_min < -negative_limit:
            raise RuntimeError(f"LDOS minimum {raw_min:.3e} is below tolerance {-negative_limit:.3e}.")
        ldos = torch.clamp(raw_ldos, min=0).to(torch.float64).cpu().numpy()
        dos = np.bincount(base_pair_index, weights=ldos, minlength=len(labels))
        total = float(ldos.sum())
        if not np.isclose(dos.sum(), total, rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"Base-pair DoS does not conserve total DoS at index {energy_index}.")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
        else:
            peak_gib = 0.0
        elapsed = time.perf_counter() - started
        atomic_savez(
            point_path,
            energy_index=np.int64(energy_index), energy_eV=np.float64(energies[energy_index]),
            dos2d=dos, total_dos=np.float64(total), elapsed_seconds=np.float64(elapsed),
            peak_allocated_GiB=np.float64(peak_gib), raw_ldos_min=np.float64(raw_min),
            scaled_residual=np.float64(residual), dtype=np.asarray(args.dtype), rhs_chunk=np.int64(args.rhs_chunk),
        )
        print(
            f"{position}/{len(indices_to_run)} index={energy_index} E={energies[energy_index]:.3f} eV "
            f"total={total:.8g} time={elapsed:.1f}s peak={peak_gib:.2f}GiB residual={residual:.3e}",
            flush=True,
        )
        del lu, pivots, green_diagonal


def merge_results(args: argparse.Namespace) -> None:
    metadata = load_metadata(args.data)
    energies = np.load(args.data / "energies_eV.npy")
    labels = metadata["base_pair_display_labels"]
    dos = np.empty((energies.size, len(labels)))
    total = np.empty(energies.size)
    elapsed = np.empty(energies.size)
    residual = np.full(energies.size, np.nan)
    dtypes: set[str] = set()
    missing = []
    for index in range(energies.size):
        path = args.output / "points" / f"energy_{index:04d}.npz"
        if not path.exists():
            missing.append(index)
            continue
        with np.load(path, allow_pickle=False) as point:
            dos[index] = point["dos2d"]
            total[index] = point["total_dos"]
            elapsed[index] = point["elapsed_seconds"]
            residual[index] = point["scaled_residual"]
            dtypes.add(str(point["dtype"]))
    if missing:
        raise ValueError(f"Missing {len(missing)} point(s), beginning with {missing[:12]}.")
    if len(dtypes) != 1:
        raise ValueError(f"Refusing to merge mixed precisions: {sorted(dtypes)}.")
    atomic_savez(
        args.output / "dos2d_results.npz", energy_eV=energies,
        base_pair_display_labels=np.asarray(labels), dos2d=dos, total_dos=total,
        elapsed_seconds=elapsed, scaled_residual=residual, dtype=np.asarray(next(iter(dtypes))),
    )
    with (args.output / "dos2d.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["energy_eV", *labels, "total_dos_1_per_eV"])
        writer.writerows(np.column_stack([energies, dos, total]))
    bin_width = float(metadata["dos_bin_width_eV"])
    bin_centers, bin_widths, binned = bin_average_dos(
        energies, np.column_stack([dos, total]), bin_width
    )
    binned_dos, binned_total = binned[:, :-1], binned[:, -1]
    bin_tag = f"{bin_width:.3f}".replace(".", "p")
    atomic_savez(
        args.output / f"dos2d_binned_{bin_tag}.npz",
        energy_eV=bin_centers,
        bin_width_eV=bin_widths,
        base_pair_display_labels=np.asarray(labels),
        dos2d=binned_dos,
        integrated_states_per_bin=binned_dos * bin_widths[:, None],
        total_dos=binned_total,
        dtype=np.asarray(next(iter(dtypes))),
    )
    with (args.output / f"dos2d_binned_{bin_tag}.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["energy_eV", "bin_width_eV", *labels, "total_dos_1_per_eV"])
        writer.writerows(np.column_stack([bin_centers, bin_widths, binned_dos, binned_total]))
    summary = {
        "state_count": int(metadata["state_count"]), "energy_count": int(energies.size),
        "base_pair_count": len(labels), "dtype": next(iter(dtypes)), "all_finite": bool(np.isfinite(dos).all()),
        "total_elapsed_hours": float(elapsed.sum() / 3600), "mean_seconds_per_energy": float(elapsed.mean()),
        "max_representative_scaled_residual": float(np.nanmax(residual)),
        "dos_bin_width_eV": bin_width,
        "binned_energy_count": int(bin_centers.size),
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merged {energies.size} points into {args.output / 'dos2d_results.npz'}.")


def report_spurious_states(args: argparse.Namespace) -> None:
    metadata = load_metadata(args.data)
    partition_path = args.data / "gaussian_partitions.json"
    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    geometry = args.geometry or Path(metadata["source_geometry"])
    records = identify_spurious_states(
        partition["nucleotides"],
        gaussian_geometry(geometry),
        np.load(args.data / "onsite_eV.npy"),
        np.load(args.data / "hamiltonian_eV.npy", mmap_mode="r"),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(records, ensure_ascii=False, indent=2))


def self_test() -> None:
    rng = np.random.default_rng(31)
    size = 9
    raw = rng.normal(size=(size, size))
    overlap = raw.T @ raw + np.eye(size)
    values, vectors = np.linalg.eigh(overlap)
    x = (vectors * values**-0.5) @ vectors.T
    h0 = rng.normal(size=(size, size))
    h0 = (h0 + h0.T) / 2
    fock = np.linalg.inv(x) @ h0 @ np.linalg.inv(x)
    recovered = x.T @ fock @ x
    assert np.allclose(x.T @ overlap @ x, np.eye(size), atol=1e-12)
    assert np.allclose(recovered, h0, atol=1e-11)
    partitions = [np.arange(0, 3), np.arange(3, 6), np.arange(6, 9)]
    unitary = np.zeros((size, size))
    onsite = np.empty(size)
    for indices in partitions:
        onsite[indices], unitary[np.ix_(indices, indices)] = np.linalg.eigh(h0[np.ix_(indices, indices)])
    localized = unitary.T @ h0 @ unitary
    assert np.allclose(np.diag(localized), onsite, atol=1e-12)
    assert np.allclose(np.linalg.eigvalsh(localized), np.linalg.eigvalsh(h0), atol=1e-12)
    probe_mask, excluded = partition_probe_mask(
        [
            {"name": "probe", "ao_ranges": [[1, 3]], "probe": True, "exclude_probe_local": [2]},
            {"name": "contact", "ao_ranges": [[4, 9]], "probe": False},
        ],
        size,
    )
    assert np.array_equal(probe_mask, [True, False, True, False, False, False, False, False, False])
    assert excluded[0]["global_index_0based"] == 1
    test_energies = np.asarray([-0.05, 0.0, 0.05])
    sigma = kramers_kronig_grid(test_energies, np.asarray([0.0]), np.asarray([True]))[:, 0]
    assert np.all(np.isfinite(sigma))
    assert np.isclose(sigma[1].real, 0.0) and np.isclose(sigma[1].imag, -0.005)
    assert np.isclose(sigma[0].real, -sigma[2].real)
    fine_energies = np.linspace(-0.12, 0.12, 97)
    test_values = np.column_stack([np.ones(fine_energies.size), fine_energies])
    centers, widths, averaged = bin_average_dos(fine_energies, test_values, 0.024)
    assert np.allclose(averaged[:, 0], 1.0)
    assert np.allclose(averaged[:, 1], centers)
    assert np.isclose(np.sum(averaged[:, 0] * widths), 0.24)
    print("PASS: Lowdin/localization, analytic KK, probe exclusions, and 0.024 eV DoS binning.")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Gaussian MAT to full-space GGCC 2D DoS PyTorch package.")
    commands = root.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--mat", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--mat", type=Path, required=True)
    prepare.add_argument("--geometry", type=Path, required=True)
    prepare.add_argument("--partition-template", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--work", type=Path)
    prepare.add_argument("--guide", type=Path)
    prepare.add_argument("--keep-work", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")
    commands.add_parser("self-test")
    probe = commands.add_parser("probe")
    probe.add_argument("--data", type=Path, required=True)
    probe.add_argument("--device", default="cuda")
    probe.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    probe.add_argument("--rhs-chunk", type=int, default=256)
    run = commands.add_parser("run")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    run.add_argument("--rhs-chunk", type=int, default=256)
    run.add_argument("--start", type=int, default=0)
    run.add_argument("--stop", type=int)
    run.add_argument("--indices", type=int, nargs="*")
    run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--data", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    spurious = commands.add_parser("spurious")
    spurious.add_argument("--data", type=Path, required=True)
    spurious.add_argument("--geometry", type=Path)
    spurious.add_argument("--output", type=Path)
    return root


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "rhs_chunk", 1) < 1:
        raise ValueError("--rhs-chunk must be positive.")
    if args.command == "inspect":
        header, matrices = scan_mat(args.mat)
        print(json.dumps(matrix_summary(header, matrices), ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        prepare_package(args)
    elif args.command == "self-test":
        self_test()
    elif args.command == "probe":
        probe_gpu(args)
    elif args.command == "run":
        run_gpu(args)
    elif args.command == "spurious":
        report_spurious_states(args)
    else:
        merge_results(args)


if __name__ == "__main__":
    try:
        main()
    except (ImportError, KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
