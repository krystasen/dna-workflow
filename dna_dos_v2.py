"""Portable, full-space Gaussian/ORCA DNA DoS using the GGCC V2 mathematics.

Local: inspect/export. Compute node: prepare (float64), run (CUDA), merge.
Local again: verify/plot. No Gaussian executable is needed after export.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np
from numpy.lib.format import open_memmap

import gaussian_full_dos_11660 as gaussian
import spurious30_v2 as v2


HERE = Path(__file__).resolve().parent
PROTOCOL = "dna-gaussian-full-dos-v2"
ORCA_PROTOCOL = "dna-orca-full-dos-v2"
DEFAULTS = {
    "energy_min_eV": -6.0, "energy_max_eV": 0.0, "energy_step_eV": 0.005,
    "eta_eV": 0.012, "dos_bin_width_eV": 0.024,
    "gamma_contact_eV": 0.6, "gamma_b_eV": 0.01, "lambda_eV": 0.05,
    "temperature_K": 300.0, "spurious_min_eV": -6.05, "spurious_max_eV": -5.65,
}
ARRAY_NAMES = {
    "OVERLAP": "overlap_packed.npy",
    "ALPHA FOCK MATRIX": "fock_packed_hartree.npy",
    "ORTHOGONAL BASIS": "gaussian_orthogonal_basis.npy",
    "ALPHA ORBITAL ENERGIES": "orbital_energies_hartree.npy",
}
ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O", 15: "P", 17: "Cl"}
BASES = {"DA": "A", "DC": "C", "DG": "G", "DT": "T"}
MODIFIED_BASES = {prefix + suffix: parent
                 for prefix, parent in (('OG','G'), ('MC','C'), ('HC','C'), ('LC','C'),
                                        ('DI','A'), ('MA','A'), ('XG','G'), ('CG','G'))
                 for suffix in '653N'}
BASES.update(MODIFIED_BASES)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, fields: list[str], rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def file_states(files: dict[str, Path]) -> dict:
    return {key: {"path": str(path.resolve()), "size": path.stat().st_size,
                  "mtime_ns": path.stat().st_mtime_ns} for key, path in files.items()}


def source_files(source: Path) -> dict[str, Path]:
    def unique(preferred: str, pattern: str) -> Path:
        if (source / preferred).is_file():
            return source / preferred
        matches = list(source.glob(pattern))
        if len(matches) != 1:
            raise ValueError(f"{source}: expected one {pattern} file or {preferred}; found {len(matches)}.")
        return matches[0]
    files = {"mat": unique("dna.mat", "*.mat"), "geometry": unique("dna.gjf", "*.gjf"),
             "pdb": unique("dna_selected.pdb", "*.pdb"), "mapping": source / "atom_mapping.csv"}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return files


def energy_grid(parameters: dict) -> np.ndarray:
    if not all(np.isfinite(float(value)) for value in parameters.values()):
        raise ValueError("All numerical parameters must be finite.")
    for key in ("energy_step_eV", "eta_eV", "dos_bin_width_eV", "lambda_eV", "temperature_K"):
        if parameters[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    if parameters["gamma_contact_eV"] < 0 or parameters["gamma_b_eV"] < 0:
        raise ValueError("Gamma must be nonnegative.")
    if parameters["spurious_min_eV"] >= parameters["spurious_max_eV"]:
        raise ValueError("The spurious-candidate energy interval is empty.")
    width = parameters["energy_max_eV"] - parameters["energy_min_eV"]
    step, bin_width = parameters["energy_step_eV"], parameters["dos_bin_width_eV"]
    if width <= 0 or step > bin_width / 2:
        raise ValueError("Energy range must be positive; grid step must be <= half the bin width.")
    if not np.isclose(width / step, round(width / step), atol=1e-9, rtol=0):
        raise ValueError("Energy range must contain an integer number of grid intervals.")
    if not np.isclose(width / bin_width, round(width / bin_width), atol=1e-9, rtol=0):
        raise ValueError("Energy range must contain an integer number of DoS bins.")
    return np.linspace(parameters["energy_min_eV"], parameters["energy_max_eV"], round(width / step) + 1)


def result_filenames(metadata: dict) -> tuple[str, str]:
    return ("dna.inp", "orca_partitions.json") if metadata["protocol"] == ORCA_PROTOCOL else ("dna.gjf", "gaussian_partitions.json")


def make_partition(header: dict, mapping: Path, geometry: Path, pdb: Path,
                   atom_index_key="gaussian_atom_1based", atoms=None) -> dict:
    atoms = gaussian.gaussian_geometry(geometry) if atoms is None else atoms
    n, nat = int(header["n_basis"]), int(header["n_atoms"])
    if len(atoms) != nat or any(element not in ELEMENTS.values() for element, _ in atoms):
        raise ValueError("Geometry must contain exactly the MAT DNA atoms (H/C/N/O/P/Cl).")
    xyz = np.asarray([xyz for _, xyz in atoms])
    if [element for element, _ in atoms] != [ELEMENTS.get(int(z)) for z in header["atomic_numbers"]]:
        raise ValueError("Geometry/MAT element order differs.")
    coordinate_error = float(np.max(np.abs(xyz - header["coordinates_A"])))
    if coordinate_error > 1e-6:
        raise ValueError(f"Geometry/MAT coordinates differ by {coordinate_error:.3g} A; use matching final geometry.")
    with mapping.open(encoding="utf-8-sig", newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row[atom_index_key]))
    if [int(row[atom_index_key]) for row in rows] != list(range(1, nat + 1)):
        raise ValueError("Atom mapping must cover every MAT atom exactly once.")
    pdb_rows = [line for line in pdb.read_text(encoding="utf-8").splitlines() if line.startswith(("ATOM  ", "HETATM"))]
    if len(pdb_rows) != nat:
        raise ValueError("PDB atom count differs from the Gaussian atom mapping.")
    for row, line, (_, point) in zip(rows, pdb_rows, atoms):
        identity = (line[12:16].strip(), line[17:20].strip(), line[21:22], int(line[22:26]))
        expected = (row["atom_name"], row["resname"], row["chain"], int(row["nucleotide_1based"]))
        if identity != expected or not np.allclose([float(line[30:38]), float(line[38:46]), float(line[46:54])], point, atol=0.00051, rtol=0):
            raise ValueError(f"PDB/mapping/geometry mismatch at atom {row[atom_index_key]}.")
    ao_atoms = np.asarray(header["ao_to_atom_1based"], dtype=np.int64)
    if ao_atoms.shape != (n,) or np.any((ao_atoms < 1) | (ao_atoms > nat)):
        raise ValueError("Invalid MAT AO-to-atom map.")
    counts = np.bincount(ao_atoms, minlength=nat + 1)[1:]
    if not np.array_equal(ao_atoms, np.repeat(np.arange(1, nat + 1), counts)) or np.any(counts == 0):
        raise ValueError("This adapter requires atom-contiguous AOs and at least one AO per atom.")
    prefix = np.cumsum(np.r_[0, counts])
    chains = list(dict.fromkeys(row["chain"] for row in rows))
    if len(chains) not in (1, 2):
        raise ValueError("The workflow supports one or two DNA strands.")
    entries, boundaries, sequences = [], [], []
    for strand, chain in enumerate(chains, 1):
        positions = sorted({int(row["nucleotide_1based"]) for row in rows if row["chain"] == chain})
        if positions != list(range(1, len(positions) + 1)):
            raise ValueError("Nucleotide numbering must start at 1 on each strand.")
        previous, sequence = None, ""
        for position in positions:
            selected = [row for row in rows if row["chain"] == chain and int(row["nucleotide_1based"]) == position]
            bases = {row["resname"] for row in selected}
            pairs = {int(row["nucleotide_1based"] if len(chains) == 1 else row["bp_1based"])
                     for row in selected}
            if len(bases) != 1 or next(iter(bases)) not in BASES or len(pairs) != 1:
                raise ValueError("Each nucleotide needs one bp assignment and a canonical or registered modified-DNA identity.")
            residue_name = next(iter(bases))
            base = BASES[residue_name]
            sequence += base
            ids = [int(row[atom_index_key]) for row in selected]
            first, last = ids[0], ids[-1]
            if ids != list(range(first, last + 1)):
                raise ValueError("Nucleotide atoms must be contiguous in Gaussian input order.")
            names = {row["atom_name"]: int(row[atom_index_key]) for row in selected}
            if len(names) != len(selected):
                raise ValueError(f"Duplicate atom names in {chain}:{position}.")
            label = residue_name if residue_name in MODIFIED_BASES else base
            entry = {"name": f"s{strand}_{position:02d}_{label}", "chain": chain, "position": position,
                     "residue_name": residue_name,
                     "base": base, "base_pair": str(next(iter(pairs))), "atom_range_1based": [first, last],
                     "ao_ranges": [[int(prefix[first - 1] + 1), int(prefix[last])]], "probe": True,
                     "atom_names": names}
            if previous is not None:
                try:
                    oxygen, phosphorus = previous["atom_names"]["O3'"], names["P"]
                except KeyError as error:
                    raise ValueError("An intact O3'-P backbone is required; nicked or capped fragments need an explicit partition.") from error
                distance = float(np.linalg.norm(xyz[oxygen - 1] - xyz[phosphorus - 1]))
                if not 1.1 < distance < 2.2:
                    raise ValueError(f"Invalid O3'-P boundary {previous['name']}->{entry['name']}: {distance:.3f} A.")
                boundaries.append({"previous": previous["name"], "current": entry["name"],
                                   "atoms_1based": [oxygen, phosphorus], "distance_A": distance})
            entries.append(entry)
            previous = entry
        sequences.append(sequence)
    order = sorted({int(entry["base_pair"]) for entry in entries})
    if order != list(range(1, len(order) + 1)):
        raise ValueError("bp positions must be consecutive from 1.")
    for pair in order:
        group = [entry for entry in entries if entry["base_pair"] == str(pair)]
        if len(group) != len(chains) or len({entry["chain"] for entry in group}) != len(chains):
            raise ValueError(f"bp {pair} must have exactly one nucleotide from each declared strand.")
    # Contact chemistry is an explicit V2 assumption, independent of spatial bending/fraying.
    first_chain = [entry for entry in entries if entry["chain"] == chains[0]]
    second_chain = [entry for entry in entries if entry["chain"] == chains[-1]]
    anchors = [("left", second_chain[-1], "O3'") if len(chains) == 2 else ("left", first_chain[0], "O5'"),
               ("right", first_chain[-1], "O3'")]
    contacts = []
    for side, entry, atom_name in anchors:
        if atom_name not in entry["atom_names"]:
            raise ValueError(f"Contact {entry['name']} lacks {atom_name}.")
        atom_id = entry["atom_names"][atom_name]
        entry["probe"] = False
        contacts.append({"side": side, "nucleotide": entry["name"], "chain": entry["chain"],
                         "position": entry["position"], "base": entry["base"], "base_pair": entry["base_pair"],
                         "atom_name": atom_name, "anchor_atom_1based": atom_id,
                         "anchor_ao_indices_0based": np.flatnonzero(ao_atoms == atom_id).tolist()})
    coverage = np.zeros(n, dtype=np.int8)
    for entry in entries:
        coverage[v2.partition_indices(entry)] += 1
    if np.any(coverage != 1):
        raise ValueError("Nucleotide AO partition does not cover every state exactly once.")
    labels = ["".join(entry["base"] for entry in entries if entry["base_pair"] == str(pair)) for pair in order]
    return {"index_base": 1, "nucleotides": entries, "contacts": contacts, "boundaries": boundaries,
            "strand_sequences_5to3": sequences, "base_pair_order": list(map(str, order)),
            "base_pair_display_labels": labels, "coordinate_max_abs_A": coordinate_error,
            "contact_model": "terminal-O orthogonal-AO projector (GGCC V2 modelling assumption)"}


def inspect_source(source: Path, output: Path) -> dict:
    files = source_files(source)
    states = file_states(files)
    report_path = output / "inspection.json"
    if report_path.is_file() and (output / "record_index.npz").is_file():
        previous = read_json(report_path)
        if previous.get("source_files") == states:
            print("Reusing unchanged MAT record index and atom mapping.", flush=True)
            return previous
    print("Scanning MAT record headers; skipping matrix payloads.", flush=True)
    header, matrices = gaussian.scan_mat(files["mat"])
    n = int(header["n_basis"])
    if int(header["n_basis_used"]) != n:
        raise ValueError("Gaussian removed basis directions; full-space Lowdin requires a separate rank-reduction decision.")
    if header["multiplicity"] != 1 or "BETA FOCK MATRIX" in matrices:
        raise ValueError("This adapter currently handles closed-shell Gaussian results only.")
    if set(ARRAY_NAMES) - matrices.keys():
        raise ValueError(f"MAT lacks required arrays: {sorted(set(ARRAY_NAMES) - matrices.keys())}.")
    for label in ARRAY_NAMES:
        info = matrices[label]
        expected = n if label == "ALPHA ORBITAL ENERGIES" else n * n if label == "ORTHOGONAL BASIS" else n * (n + 1) // 2
        if info["ntot"] != expected or info["ni"] != 0 or info["nr"] != 1:
            raise ValueError(f"Unexpected {label} storage: {info['shape_fields']}.")
    if int(np.sum(header["atomic_numbers"])) - header["charge"] != header["electrons"]:
        raise ValueError("Charge, atomic numbers and electron count are inconsistent.")
    partition = make_partition(header, files["mapping"], files["geometry"], files["pdb"])
    if file_states(files) != states:
        raise ValueError("Input files changed during inspection; inspect again after Gaussian export completes.")
    output.mkdir(parents=True, exist_ok=True)
    gaussian.atomic_savez(output / "record_index.npz", **{
        ARRAY_NAMES[label]: np.asarray(matrices[label]["records"], dtype=np.int64) for label in ARRAY_NAMES})
    gaussian.atomic_savez(output / "source_arrays.npz", **{key: header[key] for key in
        ("atomic_numbers", "coordinates_A", "ao_to_atom_1based", "ao_types")})
    save_json(output / "partitions.json", partition)
    report = {"protocol": PROTOCOL, "source": str(source.resolve()), "source_files": states,
              "state_count": n, "atom_count": int(header["n_atoms"]), "charge": int(header["charge"]),
              "multiplicity": int(header["multiplicity"]), "electron_count": int(header["electrons"]),
              "nucleotide_count": len(partition["nucleotides"]), "base_pair_count": len(partition["base_pair_order"]),
              "strand_sequences_5to3": partition["strand_sequences_5to3"], "contacts": partition["contacts"],
              "expected_boundary_candidates": len(partition["boundaries"]),
              "geometry_mat_max_abs_A": partition["coordinate_max_abs_A"], "ao_coverage_exactly_once": True,
              "source_program": header["program"], "source_title": header["title"],
              "cartesian_d_ao_present": bool(np.any(header["ao_types"] == 2006)),
              "scf_convergence": "Not independently verified; retain the Gaussian log for the scientific record.",
              "float64_square_GiB": n * n * 8 / 2**30,
              "c64_H_A_LU_GiB_excluding_workspace": n * n * 20 / 2**30,
              "matrices": {label: {key: value for key, value in matrices[label].items() if key != "records"}
                           for label in ARRAY_NAMES}}
    save_json(report_path, report)
    print(f"Inspection passed: {n} states, {report['base_pair_count']} positions.", flush=True)
    return report


def export_package(inspection: Path, output: Path, parameters: dict) -> dict:
    energies = energy_grid(parameters)
    report = read_json(inspection / "inspection.json")
    if report.get("protocol") == ORCA_PROTOCOL:
        from orca_dos_v2 import export_package as export_orca
        return export_orca(inspection, output, parameters)
    files = {key: Path(value["path"]) for key, value in report["source_files"].items()}
    if file_states(files) != report["source_files"]:
        raise ValueError("Inputs changed after inspection; run inspect again.")
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    manifest = {"protocol": PROTOCOL, "run_id": str(uuid.uuid4()), "phase": "exporting",
                "state_count": report["state_count"], "atom_count": report["atom_count"],
                "base_pair_count": report["base_pair_count"], "parameters": parameters,
                "source": report, "units": {"fock": "hartree", "overlap": "dimensionless", "geometry": "angstrom"}}
    save_json(output / "manifest.json", manifest)
    n = report["state_count"]
    with np.load(inspection / "record_index.npz", allow_pickle=False) as records:
        for label, name in ARRAY_NAMES.items():
            info = report["matrices"][label] | {"records": records[name].tolist()}
            shape = (n,) if label == "ALPHA ORBITAL ENERGIES" else (n, n) if label == "ORTHOGONAL BASIS" else (n * (n + 1) // 2,)
            temporary = inputs / (name + ".partial.npy")
            print(f"Extracting {label} -> {name}", flush=True)
            array = gaussian.extract_real_matrix(files["mat"], info, temporary, shape, label == "ORTHOGONAL BASIS")
            array.flush()
            del array
            os.replace(temporary, inputs / name)
    shutil.copy2(inspection / "source_arrays.npz", inputs / "source_arrays.npz")
    shutil.copy2(inspection / "partitions.json", output / "partitions.json")
    for role, name in (("geometry", "dna.gjf"), ("pdb", "dna_selected.pdb"), ("mapping", "atom_mapping.csv")):
        shutil.copy2(files[role], inputs / name)
    np.save(output / "energies_eV.npy", energies)
    write_package_runtime(output)
    if file_states(files) != report["source_files"]:
        raise ValueError("Source changed while extracting matrices; package remains marked incomplete.")
    manifest["phase"] = "exported"
    (output / "GPU_GUIDE.md").write_text(gpu_guide(manifest), encoding="utf-8")
    save_json(output / "manifest.json", manifest)
    print(f"GPU package exported: {output}\nNext: run prepare.sh on the compute node, then run_gpu.sh.", flush=True)
    return manifest


def write_package_runtime(output: Path) -> None:
    for name in ("dna_dos_v2.py", "orca_dos_v2.py", "gaussian_full_dos_11660.py", "spurious30_v2.py", "plot_dna_dos_v2.py",
                 "plot_figure3_abcd_gaussian_spurious30_v2.py"):
        shutil.copy2(HERE / name, output / name)
    (output / "requirements.txt").write_text("numpy>=2.0\nscipy>=1.14\n# Install a CUDA build of PyTorch separately on the compute node.\n", encoding="utf-8")
    scripts = {
        "prepare.sh": 'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"\nexport OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"\nexport MKL_NUM_THREADS="$OMP_NUM_THREADS"\n"${PYTHON:-python}" -u dna_dos_v2.py prepare --data . "$@"\n',
        "run_gpu.sh": '"${PYTHON:-python}" -u dna_dos_v2.py run --data . --output results_c64 --device cuda --dtype complex64 "$@"\n',
        "merge.sh": '"${PYTHON:-python}" -u dna_dos_v2.py merge --results results_c64 "$@"\n',
    }
    for name, body in scripts.items():
        (output / name).write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd -- "$(dirname -- "${BASH_SOURCE[0]}")"\n' + body, encoding="utf-8", newline="\n")


def build_square(path: Path, size: int, fill) -> np.memmap:
    """Commit a large stage only after it is written; interrupted stages are not reused."""
    if gaussian.valid_npy(path, (size, size)):
        print(f"Reuse completed stage: {path.name}", flush=True)
        return np.load(path, mmap_mode="r")
    temporary = path.with_name(path.stem + ".partial.npy")
    matrix = open_memmap(temporary, mode="w+", dtype=np.float64, shape=(size, size))
    fill(matrix)
    matrix.flush()
    del matrix
    os.replace(temporary, path)
    return np.load(path, mmap_mode="r")


def unpack_matrix(source: Path, output: Path, size: int) -> np.memmap:
    if not gaussian.valid_npy(output, (size, size)):
        temporary = output.with_name(output.stem + ".partial.npy")
        # expand_packed_lower also reuses files; discard only this incomplete stage first.
        if temporary.exists():
            temporary.unlink()
        matrix = gaussian.expand_packed_lower(source, temporary, size)
        del matrix
        os.replace(temporary, output)
    result = np.load(output, mmap_mode="r")
    for start in range(0, size, 512):
        if not np.isfinite(result[start:start + 512]).all():
            raise ValueError(f"Nonfinite matrix values in {source.name}.")
    return result


def probe_candidates(partition: dict, onsite: np.ndarray, hamiltonian: np.ndarray, parameters: dict) -> list[dict]:
    by_name = {entry["name"]: entry for entry in partition["nucleotides"]}
    rows = []
    for boundary in partition["boundaries"]:
        current, previous = (v2.partition_indices(by_name[boundary[key]]) for key in ("current", "previous"))
        strengths = np.linalg.norm(hamiltonian[np.ix_(current, previous)], axis=1)
        energies = onsite[current]
        distance = np.maximum(np.maximum(parameters["spurious_min_eV"] - energies,
                                         energies - parameters["spurious_max_eV"]), 0.0)
        in_window = distance == 0.0
        # Deep core orbitals dominate full-space coupling; show the nearest energies instead.
        candidates = sorted(set(np.argsort(distance, kind="stable")[:6].tolist()) | set(np.flatnonzero(in_window).tolist()),
                            key=lambda index: strengths[index], reverse=True)
        for local in candidates:
            rows.append({"nucleotide": boundary["current"], "local_index_1based": int(local + 1),
                         "global_index_0based": int(current[local]), "onsite_eV": float(onsite[current[local]]),
                         "coupling_norm_to_previous_eV": float(strengths[local]), "in_v2_window": bool(in_window[local])})
    return rows


def configure_probes(partition: dict, atoms: list, onsite: np.ndarray, hamiltonian: np.ndarray,
                     parameters: dict, overrides: Path | None = None) -> tuple[np.ndarray, list[dict], list[dict]]:
    entries = partition["nucleotides"]
    if overrides is None:
        records = v2.identify_spurious_states(entries, atoms, onsite, hamiltonian,
                                            (parameters["spurious_min_eV"], parameters["spurious_max_eV"]))
    else:
        selected = read_json(overrides)
        expected = {boundary["current"] for boundary in partition["boundaries"]}
        if set(selected) != expected:
            raise ValueError("Probe overrides must explicitly assign one local state to each O-P boundary nucleotide.")
        records = []
        for entry in entries:
            if entry["name"] not in selected:
                continue
            local = selected[entry["name"]]
            indices = v2.partition_indices(entry)
            if isinstance(local, bool) or not isinstance(local, int) or not 1 <= local <= len(indices):
                raise ValueError(f"Invalid local state for {entry['name']}: {local}.")
            entry["exclude_probe_local"] = [local]
            records.append({"nucleotide": entry["name"], "local_index_1based": local,
                            "global_index_0based": int(indices[local - 1]),
                            "onsite_eV": float(onsite[indices[local - 1]]), "identification": "explicit reviewed override"})
    mask, excluded = v2.partition_probe_mask(entries, onsite.size)
    if len(excluded) != len(partition["boundaries"]):
        raise ValueError("Boundary/probe exclusion count mismatch.")
    return mask, excluded, records


def prepare_material(data: Path, overrides: Path | None = None) -> dict:
    from scipy.linalg import eigh

    manifest = read_json(data / "manifest.json")
    if manifest.get("protocol") not in (PROTOCOL, ORCA_PROTOCOL) or manifest.get("phase") != "exported":
        raise ValueError("Package export is incomplete or has a different protocol.")
    is_orca = manifest["protocol"] == ORCA_PROTOCOL
    geometry_name, partition_name = result_filenames(manifest)
    if (data / "metadata.json").is_file():
        prepared = read_json(data / "metadata.json")
        if prepared["run_id"] != manifest["run_id"] or prepared["parameters"] != manifest["parameters"]:
            raise ValueError("Prepared data does not match this package; use a fresh package directory.")
        if overrides is not None:
            selected = read_json(overrides)
            partition = read_json(data / partition_name)
            recorded = {entry["name"]: entry["exclude_probe_local"][0]
                        for entry in partition["nucleotides"] if entry.get("exclude_probe_local")}
            if selected != recorded or any(type(value) is not int for value in selected.values()):
                raise ValueError("Preparation is already complete. Changed probe choices require a new run/package.")
        print("Prepared material already complete.", flush=True)
        return prepared
    started = time.perf_counter()
    n, parameters = int(manifest["state_count"]), manifest["parameters"]
    energies = energy_grid(parameters)
    partition = read_json(data / "partitions.json")
    work = data / "_prepare_work"
    work.mkdir(exist_ok=True)
    validation = {}
    local_path, vectors_path = data / "hamiltonian_eV.npy", work / "local_vectors.npz"
    if local_path.is_file() and vectors_path.is_file() and (work / "matrix_validation.json").is_file():
        validation = read_json(work / "matrix_validation.json")
        local = np.load(local_path, mmap_mode="r")
        onsite = np.load(data / "onsite_eV.npy")
        with np.load(vectors_path, allow_pickle=False) as loaded:
            vectors = {key: loaded[key] for key in loaded.files}
        print("Reuse completed Hamiltonian; continuing probe/contact preparation.", flush=True)
    else:
        overlap = np.load(data / "inputs/overlap.npy", mmap_mode="r") if is_orca else unpack_matrix(data / "inputs/overlap_packed.npy", work / "overlap.npy", n)
        if is_orca and (overlap.shape != (n, n) or gaussian.block_symmetry_error(overlap) > 1e-10):
            raise ValueError("Overlap shape or symmetry validation failed.")
        rng = np.random.default_rng(20260831)
        probes = rng.normal(size=(n, 3))
        orthogonal = np.load(data / ("inputs/mo_coefficients.npy" if is_orca else "inputs/gaussian_orthogonal_basis.npy"), mmap_mode="r")
        if orthogonal.shape != (n, n):
            raise ValueError("Complete square MO/orthogonalizer data is required.")
        error = float(np.linalg.norm(orthogonal.T @ (overlap @ (orthogonal @ probes)) - probes) / np.linalg.norm(probes))
        if not np.isfinite(error) or error > 1e-8:
            raise ValueError(f"{'ORCA C.T S C' if is_orca else 'Gaussian orthogonalizer X.T S X'} validation failed: {error:.3e}.")
        validation["orca_mo_orthogonality_random_relative_error" if is_orca else "gaussian_orthogonalizer_random_relative_error"] = error
        del orthogonal
        x_path = work / "lowdin_inverse_sqrt.npy"
        if gaussian.valid_npy(x_path, (n, n)):
            eigenvalues = np.load(work / "overlap_eigenvalues.npy")
            lowdin = np.load(x_path, mmap_mode="r")
        else:
            print(f"float64 symmetric eigendecomposition of S ({n} states).", flush=True)
            eigenvalues, eigenvectors = eigh(np.array(overlap, order="F"), overwrite_a=True,
                                             check_finite=False, driver="evd")
            if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
                raise ValueError("Overlap is numerically rank deficient; states will not be silently truncated.")
            np.save(work / "overlap_eigenvalues.npy", eigenvalues)
            roots = eigenvalues ** -0.5

            def fill_lowdin(out):
                for start in range(0, n, 512):
                    stop = min(n, start + 512)
                    out[start:stop] = (eigenvectors[start:stop] * roots) @ eigenvectors.T
                    print(f"Lowdin: {stop}/{n}", flush=True)

            lowdin = build_square(x_path, n, fill_lowdin)
            del eigenvectors, roots
        error = float(np.linalg.norm(lowdin.T @ (overlap @ (lowdin @ probes)) - probes) / np.linalg.norm(probes))
        symmetry = gaussian.block_symmetry_error(lowdin)
        if not np.isfinite(error) or error > 1e-10 or symmetry > 1e-10:
            raise ValueError(f"Lowdin validation failed: residual={error:.3e}, symmetry={symmetry:.3e}.")
        validation.update(lowdin_random_relative_error=error, lowdin_symmetry_error=symmetry,
                          s_min_eigenvalue=float(eigenvalues[0]), s_max_eigenvalue=float(eigenvalues[-1]))
        if is_orca:
            coefficients = np.load(data / "inputs/mo_coefficients.npy", mmap_mode="r")
            sc = build_square(work / "sc.npy", n, lambda out: np.matmul(overlap, coefficients, out=out))
            q = build_square(work / "orthogonal_mos.npy", n, lambda out: np.matmul(lowdin, sc, out=out))
            error = float(np.linalg.norm(q.T @ (q @ probes) - probes) / np.linalg.norm(probes))
            if not np.isfinite(error) or error > 1e-8:
                raise ValueError(f"ORCA Q=S^1/2 C is not orthogonal: {error:.3e}.")
            validation.update(orca_q_orthogonality_random_relative_error=error,
                              h0_method="Q=S^1/2 C; H0=Q epsilon Q.T, equivalent to Lowdin F/S for complete orthonormal MOs")
            orbital = np.load(data / "inputs/orbital_energies_hartree.npy") * gaussian.HARTREE_TO_EV

            def fill_h0(out):
                for start in range(0, n, 512):
                    stop = min(n, start + 512)
                    out[start:stop] = (q[start:stop] * orbital) @ q.T
                gaussian.symmetrize_in_place(out)

            h0 = build_square(work / "h0_eV.npy", n, fill_h0)
            del coefficients, sc, q, overlap
        else:
            del overlap
            fock = unpack_matrix(data / "inputs/fock_packed_hartree.npy", work / "fock.npy", n)
            print("float64 F @ X", flush=True)
            fx = build_square(work / "fx.npy", n, lambda out: np.matmul(fock, lowdin, out=out))
            print("float64 H0 = X.T @ F @ X", flush=True)

            def fill_h0(out):
                np.matmul(lowdin.T, fx, out=out)
                out *= gaussian.HARTREE_TO_EV
                gaussian.symmetrize_in_place(out)

            h0 = build_square(work / "h0_eV.npy", n, fill_h0)
            del fock, fx
        del lowdin
        gc.collect()
        orbital_eV = np.load(data / "inputs/orbital_energies_hartree.npy") * gaussian.HARTREE_TO_EV
        trace_error = float(abs(np.trace(h0) - orbital_eV.sum()))
        h0_norm = gaussian.frobenius_squared(h0)
        spectrum_norm = float(orbital_eV @ orbital_eV)
        spectrum_error = abs(h0_norm - spectrum_norm) / max(spectrum_norm, 1.0)
        if trace_error > 1e-4 or spectrum_error > 1e-10:
            raise ValueError(f"Transformed spectrum disagrees with MO energies: trace={trace_error:.3g}, norm={spectrum_error:.3g}.")
        validation.update(h0_trace_error_eV=trace_error, h0_spectral_norm_relative_error=spectrum_error)
        vectors, onsite, unitary_error = {}, np.empty(n, dtype=np.float64), 0.0
        for entry in partition["nucleotides"]:
            indices = v2.partition_indices(entry)
            values, block_vectors = np.linalg.eigh(h0[np.ix_(indices, indices)])
            onsite[indices] = values
            vectors[entry["name"]] = block_vectors
            unitary_error = max(unitary_error, float(np.max(np.abs(block_vectors.T @ block_vectors - np.eye(len(indices))))))
            print(f"Localized {entry['name']}: {len(indices)} states", flush=True)

        def fill_right(out):
            for entry in partition["nucleotides"]:
                indices = v2.partition_indices(entry)
                out[:, indices] = h0[:, indices] @ vectors[entry["name"]]

        right = build_square(work / "h0_u.npy", n, fill_right)

        def fill_local(out):
            for entry in partition["nucleotides"]:
                indices = v2.partition_indices(entry)
                out[indices, :] = vectors[entry["name"]].T @ right[indices, :]
            gaussian.symmetrize_in_place(out)

        if local_path.exists():
            # H without its saved local vectors/validation is an unfinished stage.
            local_path.unlink()
        local = build_square(local_path, n, fill_local)
        diagonal_error = float(np.max(np.abs(np.diag(local) - onsite)))
        invariant_error = abs(gaussian.frobenius_squared(local) - h0_norm) / max(h0_norm, 1.0)
        symmetry = gaussian.block_symmetry_error(local)
        if unitary_error > 1e-10 or diagonal_error > 1e-7 or invariant_error > 1e-10 or symmetry > 1e-7:
            raise ValueError("Localized Hamiltonian failed unitarity, diagonal, Hermitian or spectral-invariant validation.")
        validation.update(local_unitarity_max_abs=unitary_error, local_diagonal_error_eV=diagonal_error,
                          local_spectral_norm_relative_error=invariant_error, local_symmetry_error_eV=symmetry)
        np.save(data / "onsite_eV.npy", onsite)
        gaussian.atomic_savez(vectors_path, **vectors)
        save_json(work / "matrix_validation.json", validation)
        del h0, right
        gc.collect()
    atoms = v2.load_geometry(data / "inputs" / geometry_name)
    candidates = probe_candidates(partition, onsite, local, parameters)
    fields = ["nucleotide", "local_index_1based", "global_index_0based", "onsite_eV", "coupling_norm_to_previous_eV", "in_v2_window"]
    write_csv(data / "probe_candidates.csv", fields, ([row[key] for key in fields] for row in candidates))
    try:
        mask, excluded, records = configure_probes(partition, atoms, onsite, local, parameters, overrides)
    except ValueError as error:
        raise ValueError(f"{error} Review probe_candidates.csv; explicit reviewed choices can be supplied with --probe-overrides. The Hamiltonian is retained; no energy scan was started.") from error
    fields = list(dict.fromkeys(key for row in records for key in row)) or ["nucleotide", "local_index_1based"]
    write_csv(data / "spurious_states.csv", fields, ([row.get(key, "") for key in fields] for row in records))
    groups = partition["base_pair_order"]
    bp_index = np.full(n, -1, dtype=np.int32)
    entries = {entry["name"]: entry for entry in partition["nucleotides"]}
    for entry in entries.values():
        bp_index[v2.partition_indices(entry)] = groups.index(entry["base_pair"])
    if np.any(bp_index < 0):
        raise ValueError("Incomplete base-pair state coverage.")
    for contact in partition["contacts"]:
        entry = entries[contact["nucleotide"]]
        indices = v2.partition_indices(entry)
        rows = np.flatnonzero(np.isin(indices, contact["anchor_ao_indices_0based"]))
        factor = vectors[entry["name"]][rows, :].T.copy()
        projector = factor @ factor.T
        eigenvalues = np.linalg.eigvalsh(projector)
        idempotency = float(np.max(np.abs(projector @ projector - projector)))
        if len(rows) == 0 or idempotency > 1e-10 or eigenvalues[0] < -1e-10:
            raise ValueError("Invalid contact projector.")
        side = contact["side"]
        np.save(data / f"contact_{side}_indices.npy", indices)
        np.save(data / f"contact_{side}_factor.npy", factor)
        np.save(data / f"contact_{side}_projector.npy", projector)
        contact.update(projector_rank=len(rows), projector_idempotency_error=idempotency,
                       state_count=len(indices), base_pair_index=groups.index(entry["base_pair"]))
    np.save(data / "base_pair_index.npy", bp_index)
    np.save(data / "probe_mask.npy", mask)
    np.save(data / "excluded_probe_indices.npy", np.asarray([row["global_index_0based"] for row in excluded], dtype=np.int64))
    print("Preparing infinite-domain analytic KK self-energy.", flush=True)
    sigma_b = v2.analytic_sigma_b(energies, onsite, mask, parameters["gamma_b_eV"], parameters["lambda_eV"])
    if not np.isfinite(sigma_b).all() or np.any(sigma_b.imag > 1e-14):
        raise ValueError("Nonfinite or non-retarded probe self-energy.")
    np.save(data / "sigma_b_eV.npy", sigma_b)
    save_json(data / partition_name, partition)
    metadata = {"protocol": manifest["protocol"], "run_id": manifest["run_id"], "parameters": parameters,
                "state_count": n, "atom_count": manifest["atom_count"], "base_pair_count": len(groups),
                "base_pair_order": groups, "base_pair_display_labels": partition["base_pair_display_labels"],
                "strand_sequences_5to3": partition["strand_sequences_5to3"], "contacts": partition["contacts"],
                "contact_model": partition["contact_model"], "source": manifest["source"],
                "excluded_probe_state_count": len(excluded), "probe_state_count": int(mask.sum()),
                "expected_boundary_count": len(partition["boundaries"]),
                "probe_identification": "reviewed explicit indices" if overrides else "V2 strongest previous-block coupling in configured energy window",
                "kramers_kronig_method": "analytic infinite-domain exponential kernel", "hamiltonian_dtype": "float64",
                "validation": validation, "preparation_minutes": (time.perf_counter() - started) / 60}
    save_json(data / "metadata.json", metadata)
    print(f"Prepared all {n} states; {len(excluded)} excluded probe states. Ready for CUDA run.", flush=True)
    return metadata


def point_contract(metadata: dict, dtype: str) -> str:
    return json.dumps({"protocol": metadata["protocol"], "run_id": metadata["run_id"], "state_count": metadata["state_count"],
                       "base_pair_order": metadata["base_pair_order"], "parameters": metadata["parameters"],
                       "dtype": dtype}, sort_keys=True)


def load_point(path: Path, metadata: dict, dtype: str, index: int, energy: float) -> dict:
    with np.load(path, allow_pickle=False) as loaded:
        point = {key: loaded[key] for key in loaded.files}
    if str(point.get("contract", "")) != point_contract(metadata, dtype):
        raise ValueError(f"{path.name}: different run, parameters or precision; use a separate results directory.")
    if int(point["energy_index"]) != index or not np.isclose(float(point["energy_eV"]), energy, atol=1e-12, rtol=0):
        raise ValueError(f"{path.name}: wrong energy index/value.")
    if point["dos2d"].shape != (len(metadata["base_pair_order"]),) or point["contact_dos"].shape != (2,):
        raise ValueError(f"{path.name}: incorrect spatial dimensions.")
    if not np.isfinite(point["dos2d"]).all() or not np.isfinite(point["contact_dos"]).all() or not np.isfinite(point["total_dos"]):
        raise ValueError(f"{path.name}: nonfinite DoS.")
    if np.any(point["dos2d"] < 0) or np.any(point["contact_dos"] < 0) or not np.isclose(point["dos2d"].sum(), point["total_dos"], rtol=1e-10, atol=1e-10):
        raise ValueError(f"{path.name}: negative DoS or spatial sum mismatch.")
    return point


def green_observables(torch, matrix, bp_index: np.ndarray, bp_count: int, contacts: list,
                      rhs_chunk: int, check_residual: bool) -> dict:
    """One LU per energy; diagonal RHS blocks plus two low-rank contact projections."""
    lu, pivots, info = torch.linalg.lu_factor_ex(matrix, check_errors=False)
    if int(info.item()):
        raise RuntimeError(f"LU factorization failed: info={info.item()}.")
    is_single = matrix.dtype == torch.complex64
    residual = gaussian.scaled_residual(torch, matrix, lu, pivots) if check_residual else np.nan
    if check_residual and (not np.isfinite(residual) or residual > (5e-5 if is_single else 1e-10)):
        raise RuntimeError(f"Green function residual failed: {residual:.3e}.")
    diagonal = gaussian.inverse_diagonal_from_lu(torch, lu, pivots, matrix.shape[0], rhs_chunk)
    raw = -diagonal.imag / np.pi
    if not bool(torch.isfinite(raw).all().item()):
        raise RuntimeError("Nonfinite local DoS.")
    tolerance = (1e-5 if is_single else 1e-10) * max(float(raw.abs().max().item()), 1.0)
    raw_min = float(raw.min().item())
    negative_weight = float((-raw[raw < 0]).sum().item())
    if raw_min < -tolerance:
        raise RuntimeError(f"Negative LDOS beyond numerical tolerance: {raw_min:.3e}.")
    ldos = raw.clamp(min=0).to(torch.float64).cpu().numpy()
    projected = []
    for indices, factor in contacts:
        rhs = torch.zeros((matrix.shape[0], factor.shape[1]), dtype=matrix.dtype, device=matrix.device)
        rhs[indices] = factor.to(matrix.dtype)
        solved = torch.linalg.lu_solve(lu, pivots, rhs)
        value = float((-(factor.conj() * solved[indices]).sum().imag / np.pi).item())
        if not np.isfinite(value) or value < -tolerance:
            raise RuntimeError(f"Invalid contact-projected DoS: {value}.")
        projected.append(max(0.0, value))
    dos = np.bincount(bp_index, weights=ldos, minlength=bp_count)
    if not np.isclose(dos.sum(), ldos.sum(), atol=1e-10, rtol=1e-10):
        raise RuntimeError("bp trace does not conserve the full-space DoS.")
    return {"dos2d": dos, "total_dos": np.float64(ldos.sum()), "contact_dos": np.asarray(projected),
            "raw_ldos_min": np.float64(raw_min), "negative_ldos_count": np.int64((raw < 0).sum().item()),
            "negative_ldos_weight": np.float64(negative_weight),
            "negative_ldos_fraction": np.float64(negative_weight / max(ldos.sum(), 1e-30)),
            "scaled_residual": np.float64(residual)}


def run_gpu(args: argparse.Namespace) -> None:
    torch = gaussian.import_torch()
    metadata = read_json(args.data / "metadata.json")
    manifest = read_json(args.data / "manifest.json")
    if metadata["protocol"] not in (PROTOCOL, ORCA_PROTOCOL) or metadata["protocol"] != manifest["protocol"] or metadata["run_id"] != manifest["run_id"] or metadata["parameters"] != manifest["parameters"]:
        raise ValueError("Prepared material and package parameters disagree.")
    if args.rhs_chunk < 1:
        raise ValueError("RHS chunk size must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a CUDA build of PyTorch on the compute node.")
    torch.set_float32_matmul_precision("highest")
    real_dtype, complex_dtype = gaussian.torch_dtypes(torch, args.dtype)
    energies = np.load(args.data / "energies_eV.npy")
    if not np.allclose(energies, energy_grid(metadata["parameters"]), atol=1e-12, rtol=0):
        raise ValueError("Energy grid disagrees with package parameters.")
    selected = gaussian.selected_energy_indices(args, len(energies))
    n = metadata["state_count"]
    bp_index = np.load(args.data / "base_pair_index.npy")
    count = len(metadata["base_pair_order"])
    if bp_index.shape != (n,) or np.any((bp_index < 0) | (bp_index >= count)):
        raise ValueError("Invalid state-to-bp mapping.")
    sigma = np.load(args.data / "sigma_b_eV.npy", mmap_mode="r")
    h = np.load(args.data / "hamiltonian_eV.npy", mmap_mode="r")
    if h.shape != (n, n) or sigma.shape != (len(energies), n):
        raise ValueError("Prepared matrix dimensions disagree with metadata.")
    args.output.mkdir(parents=True, exist_ok=True)
    run_metadata = metadata | {"dtype": args.dtype}
    run_meta_path = args.output / "run_manifest.json"
    if run_meta_path.exists():
        saved = read_json(run_meta_path)
        if point_contract(saved, saved["dtype"]) != point_contract(metadata, args.dtype):
            raise ValueError("This results directory belongs to a different run or precision.")
    else:
        save_json(run_meta_path, run_metadata)
    geometry_name, partition_name = result_filenames(metadata)
    for source, destination in (("energies_eV.npy", "energies_eV.npy"), (partition_name, partition_name),
                                ("inputs/" + geometry_name, geometry_name), ("inputs/dna_selected.pdb", "dna_selected.pdb"),
                                ("inputs/atom_mapping.csv", "atom_mapping.csv"), ("spurious_states.csv", "spurious_states.csv")):
        if not (args.output / destination).exists():
            shutil.copy2(args.data / source, args.output / destination)
    points = args.output / "points"
    points.mkdir(exist_ok=True)
    pending = []
    for index in selected:
        path = points / f"energy_{index:04d}.npz"
        if args.resume and path.is_file():
            load_point(path, metadata, args.dtype, index, energies[index])
        else:
            pending.append(index)
    if not pending:
        print("All selected energy points are already complete and compatible.", flush=True)
        return
    hamiltonian = torch.tensor(h, dtype=real_dtype, device=device)
    contact_factors, contact_projectors = [], []
    for side in ("left", "right"):
        indices = torch.tensor(np.load(args.data / f"contact_{side}_indices.npy"), dtype=torch.int64, device=device)
        factor = torch.tensor(np.load(args.data / f"contact_{side}_factor.npy"), dtype=real_dtype, device=device)
        contact_factors.append((indices, factor))
        contact_projectors.append((indices, factor @ factor.T))
    print(f"{n} states on {device}; {len(pending)} pending energies; {args.dtype}.", flush=True)
    representative = {pending[0], 0, len(energies) // 6, len(energies) // 2, len(energies) - 1}
    for position, index in enumerate(pending, 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        matrix = hamiltonian.to(complex_dtype)
        matrix.neg_()
        shift = energies[index] + 1j * metadata["parameters"]["eta_eV"] - sigma[index]
        matrix.diagonal().add_(torch.tensor(shift, dtype=complex_dtype, device=device))
        for indices, projector in contact_projectors:
            matrix[indices[:, None], indices[None, :]] += 0.5j * metadata["parameters"]["gamma_contact_eV"] * projector
        observed = green_observables(torch, matrix, bp_index, count, contact_factors, args.rhs_chunk, index in representative)
        del matrix
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        gaussian.atomic_savez(points / f"energy_{index:04d}.npz", **observed,
                              contract=np.asarray(point_contract(metadata, args.dtype)), energy_index=np.int64(index),
                              energy_eV=np.float64(energies[index]), elapsed_seconds=np.float64(elapsed),
                              peak_allocated_GiB=np.float64(peak), rhs_chunk=np.int64(args.rhs_chunk))
        print(f"{position}/{len(pending)} E={energies[index]:.5f} eV index={index} time={elapsed:.2f}s peak={peak:.2f} GiB", flush=True)


def merge_results(results: Path) -> dict:
    metadata = read_json(results / "run_manifest.json")
    energies = np.load(results / "energies_eV.npy")
    missing = [index for index in range(len(energies)) if not (results / "points" / f"energy_{index:04d}.npz").is_file()]
    if missing:
        save_json(results / "verification_report.json", {"complete": False, "missing_count": len(missing), "missing_indices": missing})
        raise ValueError(f"Missing {len(missing)} energy points, beginning with {missing[:20]}; see verification_report.json.")
    points = [load_point(results / "points" / f"energy_{index:04d}.npz", metadata, metadata["dtype"], index, energy)
              for index, energy in enumerate(energies)]
    dos = np.asarray([point["dos2d"] for point in points])
    total = np.asarray([point["total_dos"] for point in points])
    contact = np.asarray([point["contact_dos"] for point in points])
    timings = np.asarray([point["elapsed_seconds"] for point in points])
    residuals = np.asarray([point["scaled_residual"] for point in points])
    common = {"run_id": np.asarray(metadata["run_id"]), "dtype": np.asarray(metadata["dtype"]),
              "base_pair_order": np.asarray(metadata["base_pair_order"]),
              "base_pair_display_labels": np.asarray(metadata["base_pair_display_labels"])}
    gaussian.atomic_savez(results / "dos2d_results.npz", **common, energy_eV=energies, dos2d=dos,
                          total_dos=total, contact_dos=contact, elapsed_seconds=timings, scaled_residual=residuals)
    centers, widths, averaged = v2.bin_average_dos(energies, np.column_stack([dos, total, contact]), metadata["parameters"]["dos_bin_width_eV"])
    count = dos.shape[1]
    gaussian.atomic_savez(results / "dos2d_binned.npz", **common, energy_eV=centers, bin_width_eV=widths,
                          dos2d=averaged[:, :count], total_dos=averaged[:, count], contact_dos=averaged[:, count + 1:],
                          integrated_states_per_bin=averaged[:, :count] * widths[:, None])
    columns = [f"bp_{int(pair):03d}" for pair in metadata["base_pair_order"]]
    write_csv(results / "dos2d.csv", ["energy_eV", *columns, "total_dos_eV-1", "left_contact_dos_eV-1", "right_contact_dos_eV-1"],
              np.column_stack([energies, dos, total, contact]))
    write_csv(results / "dos2d_binned.csv", ["energy_eV", "bin_width_eV", *columns, "total_dos_eV-1", "left_contact_dos_eV-1", "right_contact_dos_eV-1"],
              np.column_stack([centers, widths, averaged]))
    write_csv(results / "base_pairs.csv", ["column", "position_1based", "bases"],
              zip(columns, metadata["base_pair_order"], metadata["base_pair_display_labels"]))
    finite_residuals = residuals[np.isfinite(residuals)]
    summary = {"run_id": metadata["run_id"], "state_count": metadata["state_count"], "energy_count": len(energies),
               "base_pair_count": count, "dtype": metadata["dtype"], "all_finite": True,
               "total_elapsed_hours": float(timings.sum() / 3600), "mean_seconds_per_energy": float(timings.mean()),
               "max_representative_scaled_residual": float(finite_residuals.max()) if finite_residuals.size else None,
               "max_peak_allocated_GiB": max(float(point["peak_allocated_GiB"]) for point in points),
               "minimum_raw_ldos": min(float(point["raw_ldos_min"]) for point in points),
               "max_negative_ldos_fraction": max(float(point["negative_ldos_fraction"]) for point in points),
               "dos_bin_width_eV": metadata["parameters"]["dos_bin_width_eV"], "binned_energy_count": len(centers)}
    save_json(results / "run_summary.json", summary)
    print(f"Merged {len(energies)} energies, {count} bp positions.", flush=True)
    return verify_results(results)


def verify_results(results: Path, package: Path | None = None) -> dict:
    metadata = read_json(results / "run_manifest.json")
    if metadata.get("protocol") not in (PROTOCOL, ORCA_PROTOCOL):
        raise ValueError("This is not a result from the general DNA V2 package.")
    if package is not None:
        manifest = read_json(package / "manifest.json")
        for key in ("protocol", "run_id", "state_count", "parameters"):
            if manifest[key] != metadata[key]:
                raise ValueError(f"Returned results do not match the selected package: {key}.")
    expected = energy_grid(metadata["parameters"])
    with np.load(results / "dos2d_results.npz", allow_pickle=False) as raw, np.load(results / "dos2d_binned.npz", allow_pickle=False) as binned:
        for data in (raw, binned):
            if str(data["run_id"]) != metadata["run_id"] or str(data["dtype"]) != metadata["dtype"]:
                raise ValueError("Merged result identity/precision mismatch.")
            if data["base_pair_order"].astype(str).tolist() != metadata["base_pair_order"]:
                raise ValueError("bp position order differs from the package.")
            if not np.isfinite(data["dos2d"]).all() or not np.isfinite(data["total_dos"]).all() or not np.isfinite(data["contact_dos"]).all():
                raise ValueError("Nonfinite returned data.")
            if np.any(data["dos2d"] < 0) or np.any(data["contact_dos"] < 0):
                raise ValueError("Negative returned DoS.")
            if not np.allclose(data["dos2d"].sum(axis=1), data["total_dos"], atol=1e-10, rtol=1e-10):
                raise ValueError("Returned bp DoS does not conserve total DoS.")
        if raw["dos2d"].shape != (len(expected), len(metadata["base_pair_order"])):
            raise ValueError("Incomplete energy grid or incorrect bp count.")
        if not np.allclose(raw["energy_eV"], expected, rtol=0, atol=1e-12):
            raise ValueError("Wrong returned energy grid.")
        centers, widths, averaged = v2.bin_average_dos(expected, np.column_stack([raw["dos2d"], raw["total_dos"], raw["contact_dos"]]),
                                                     metadata["parameters"]["dos_bin_width_eV"])
        combined = np.column_stack([binned["dos2d"], binned["total_dos"], binned["contact_dos"]])
        if not np.allclose(centers, binned["energy_eV"], atol=1e-12, rtol=0) or not np.allclose(widths, binned["bin_width_eV"], atol=1e-12, rtol=0) or not np.allclose(averaged, combined, atol=1e-11, rtol=1e-11):
            raise ValueError("Returned bins disagree with integral-conserving V2 binning.")
    geometry_name, partition_name = result_filenames(metadata)
    partition = read_json(results / partition_name)
    coverage = np.zeros(metadata["state_count"], dtype=np.int8)
    for entry in partition["nucleotides"]:
        coverage[v2.partition_indices(entry)] += 1
    if np.any(coverage != 1):
        raise ValueError("Returned partition does not cover all states exactly once.")
    for name in (geometry_name, "dna_selected.pdb", "atom_mapping.csv", "run_summary.json"):
        if not (results / name).is_file():
            raise FileNotFoundError(f"Return the complete results folder, including {name}.")
    summary = read_json(results / "run_summary.json")
    if summary["run_id"] != metadata["run_id"] or summary["dtype"] != metadata["dtype"]:
        raise ValueError("Run summary belongs to different results.")
    residual = summary.get("max_representative_scaled_residual")
    if residual is None or not np.isfinite(residual) or residual > (5e-5 if metadata["dtype"] == "complex64" else 1e-10):
        raise ValueError("Returned solver residual check is missing or failed.")
    report = {"complete": True, "run_id": metadata["run_id"], "state_count": metadata["state_count"],
              "energy_count": len(expected), "base_pair_count": len(metadata["base_pair_order"]),
              "dtype": metadata["dtype"], "max_scaled_residual": residual,
              "package_matched": package is not None, "bin_integral_conserved": True}
    save_json(results / "verification_report.json", report)
    print(f"Verified full result: {len(expected)} energies; {metadata['state_count']} states.", flush=True)
    return report


def self_test(device: str = "cpu") -> None:
    torch = gaussian.import_torch()
    rng = np.random.default_rng(31)
    n = 9
    h = rng.normal(size=(n, n))
    h = (h + h.T) / 2
    sigma = v2.analytic_sigma_b(np.asarray([0.1]), np.diag(h), np.ones(n, dtype=bool))[0]
    left = np.linalg.qr(rng.normal(size=(3, 3)))[0][:, :2]
    right = np.linalg.qr(rng.normal(size=(3, 3)))[0][:, :1]
    bp = np.repeat(np.arange(3), 3)
    reference = (0.1 + 0.012j) * np.eye(n) - h - np.diag(sigma)
    reference[:3, :3] += 0.3j * left @ left.T
    reference[-3:, -3:] += 0.3j * right @ right.T
    inverse = np.linalg.inv(reference)
    expected_dos = np.bincount(bp, weights=-inverse.diagonal().imag / np.pi)
    expected_contact = np.asarray([-np.trace(left.T @ inverse[:3, :3] @ left).imag / np.pi,
                                   -np.trace(right.T @ inverse[-3:, -3:] @ right).imag / np.pi])
    for dtype, tolerance in (("complex64", 2e-5), ("complex128", 1e-11)):
        real_type, complex_type = gaussian.torch_dtypes(torch, dtype)
        contacts = [(torch.tensor(indices, device=device), torch.tensor(factor, dtype=real_type, device=device))
                    for indices, factor in (([0, 1, 2], left), ([6, 7, 8], right))]
        observed = green_observables(torch, torch.tensor(reference, dtype=complex_type, device=device), bp, 3, contacts, 2, True)
        np.testing.assert_allclose(observed["dos2d"], expected_dos, atol=tolerance, rtol=tolerance)
        np.testing.assert_allclose(observed["contact_dos"], expected_contact, atol=tolerance, rtol=tolerance)
    gaussian.self_test()
    print("PASS: full diagonal, non-diagonal contact projection, c64/c128 and shared V2 checks.")


def gpu_guide(manifest: dict) -> str:
    is_orca = manifest["protocol"] == ORCA_PROTOCOL
    engine = "ORCA" if is_orca else "Gaussian"
    geometry_name, partition_name = result_filenames(manifest)
    source_description = "ORCA S / 完整 MO / 轨道能量" if is_orca else "Gaussian F/S"
    raw_file = "dna.json" if is_orca else "dna.mat"
    h0_formula = "Q=S^(1/2)C、H0=Q epsilon Q.T" if is_orca else "H0=X.T F X"
    validation_note = ("ORCA 验证完整实闭壳层 C.T S C=I；Q=S^(1/2)C，等价于对重建 F 进行对称 Löwdin。H-Matrix 不作为 Fock。"
                       if is_orca else "输入中的 Gaussian `ORTHOGONAL BASIS` 只作验证，不能替代自行构建的对称 Löwdin。")
    n, p = manifest["state_count"], manifest["parameters"]
    count = len(energy_grid(p))
    return f"""# DNA 全空间 DoS GPU 执行指南

本包协议：`{manifest['protocol']}`；run_id：`{manifest['run_id']}`。
体系：{manifest['atom_count']} 原子、{manifest['base_pair_count']} 个位置、**{n} 态**。
本包来自 {source_description}。Windows 来源路径只作溯源；运行只用包内相对路径。

## 1. 复制与环境

复制整个包到超算可读写目录。本包不需要 Gaussian、ORCA、Obsidian、VMD 或 Windows。
原始 {raw_file} 不需再次上传。保留本机原始计算文件和 {engine} 收敛日志。

使用 Python 3.10+、NumPy 2.x、SciPy 和与 NVIDIA 驱动匹配的 CUDA 版 PyTorch。
若超算提供现成 PyTorch 环境，优先使用。没有时按 https://pytorch.org/get-started/locally/ 选择 CUDA 安装命令。
**不要把本机测试环境的 CPU-only PyTorch 复制到超算。**

```bash
cd /your/path/to/package
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python dna_dos_v2.py self-test --device cuda
```

最后一条只计算 9 阶合成模型，验证 GPU 上的 c64/c128 对角与接触投影求解。
在调度器分配的节点内运行。脚本不自动提交 Slurm/LSF，也不假设队列、账户或 GPU 型号。

## 2. 一次性 float64 数学预处理

```bash
export OMP_NUM_THREADS=8  # 改成调度器实际分配的 CPU 数
bash prepare.sh
```

此步骤在 CPU 上用 float64 计算 S^(-1/2)、{h0_formula}、nucleotide 块对角化和 H=U.T H0 U。
全空间不做 active-window 截断。单张 float64 方阵约 {n*n*8/2**30:.2f} GiB；需要多张矩阵和分解工作空间。
本次规模建议使用内存充足的节点（例如 64 GiB 或更多），实际峰值由 BLAS/LAPACK 后端决定。
`_prepare_work` 保存完成的中间阶段；中断后再次运行相同命令。
`metadata.json` 仅在全部预处理与数值检查通过后生成。

完成后确认：state_count={n}；bp 数={manifest['base_pair_count']}；
Löwdin 随机向量残差、局域化不变量及 contact projector 验证均通过。
{validation_note}

## 3. 检查 probe 候选

`probe_candidates.csv` 列出能窗内全部候选和距能窗最近的6个态，并报告各自的跨边界耦合；
窗外近邻只用于人工核查，不会自动被选中。`spurious_states.csv` 记录实际选中的局域编号。
候选数取决于实际分区，本包预计 {manifest['source']['expected_boundary_candidates']} 个边界，不能照搬 GGCC 的 30 个编号。
默认遵循 V2 能窗内最大前块耦合规则（{p['spurious_min_eV']} 至 {p['spurious_max_eV']} eV）。
排除仅作用于 probe；所有 Hamiltonian 态仍保留。

若某个 nucleotide 在该窗口无候选，程序保留 H 并停止，不启动全能窗扫描。
由科研人员或 Codex 根据候选能量、边界与跨块耦合核查；不要为凑足数量任意挑态。
确有必要人工确认时，将每个边界的 nucleotide 名和局域一基索引写成 JSON 字典，然后：

```bash
python dna_dos_v2.py prepare --data . --probe-overrides reviewed_probe_indices.json
```

该 JSON 必须覆盖所有边界，不能包含链首无前置边界的 nucleotide。
重复使用相同的人工选择会复用已完成的预处理，不会重新分解矩阵。
已完成预处理或开始扫描后，改变 probe/contact/展宽需另建计算包，不能混用旧点文件。

## 4. 单点显存与速度测试

```bash
bash run_gpu.sh --indices 0 --rhs-chunk 256
```

本包默认 `complex64`。H、复矩阵和 LU 三项约 {n*n*20/2**30:.2f} GiB，尚未包含 CUDA 工作空间。
查看输出的耗时、peak GiB、残差。RHS 阶段显存不足可改为 128 或 64；LU 阶段不足须用更大显存。
不要用降低 eta 或截断态数来解决显存问题。

## 5. 完整计算与续算

```bash
bash run_gpu.sh --rhs-chunk 256 --resume
bash merge.sh
```

范围 {p['energy_min_eV']} 至 {p['energy_max_eV']} eV，步长 {p['energy_step_eV']} eV，共 {count} 点。
eta={p['eta_eV']} eV；GammaL/R={p['gamma_contact_eV']} eV；GammaB={p['gamma_b_eV']} eV；lambda={p['lambda_eV']} eV。
probe 实部使用指数核的无限域解析 KK。
每点保存到 results_c64/points/energy_XXXX.npz。续算会核验 run_id、能量、参数和精度后跳过已有点。
`--no-resume` 会覆盖选定能量点，仅在明确需要重算这些点时使用。

能量点可分配给不同 GPU/作业（start 包含，stop 不包含），每个作业都需足够显存容纳完整方阵：

```bash
# 在分别获得 GPU 的两个作业中执行；无需把单个矩阵拆到两张卡。
bash run_gpu.sh --start 0 --stop {count//2} --resume
bash run_gpu.sh --start {count//2} --stop {count} --resume
# 两个作业完成后再执行一次：
bash merge.sh
```

complex128 仅作少量代表点对照时，使用独立目录：

```bash
python dna_dos_v2.py run --data . --output results_c128 --dtype complex128 --device cuda --indices 0 {count//2} --rhs-chunk 128
```

这里仅算两个点，不能 merge 成完整二维图。CUDA 数值误差须通过真实单点实测判断。

## 6. 回传内容

将整个 **results_c64** 目录复制回本机。核心文件包括：
- dos2d_results.npz / dos2d.csv：原始逐能量 bp DoS 与左右 contact-projected DoS；
- dos2d_binned.npz / dos2d_binned.csv：{p['dos_bin_width_eV']} eV 积分守恒分箱；
- run_manifest.json、run_summary.json、verification_report.json；
- {partition_name}、{geometry_name}、dna_selected.pdb、atom_mapping.csv、base_pairs.csv；
- points：建议保留用于审计和补算；已完整 merge 后本机绘图不要求逐点文件。

不必回传 Hamiltonian、F/S 或 _prepare_work。不要只传一份无元数据的 CSV。

## 7. 本机工作台

打开“DoS 与结果”，选择本机保留的计算包及回传 results_c64，点击“核验回传”，再“生成 Figure 3a–d”。
本机核验与绘图不需要 CUDA。gateway 来自接触谱候选或手动给定值，不使用 GGCC 的固定能量。
温度默认 {p['temperature_K']} K，可在绘图时修改，不需要重算 Green function。
Figure c 沿用 V2 的积分态数 / 能量 bin 宽度 / Bondi 原子球体积之和；另存不归一化的积分态数。
Figure d 全部原子继承所属 bp 的一个颜色，提供位置与序列轴。
显示平滑只影响 a/b，不参与 c/d 定量。
"""


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--source", type=Path, required=True)
    inspect.add_argument("--engine", choices=("gaussian", "orca"), default="gaussian")
    inspect.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--inspection", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    for key, value in DEFAULTS.items():
        export.add_argument("--" + key.replace("_", "-"), dest=key, type=float, default=value)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--probe-overrides", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    run.add_argument("--rhs-chunk", type=int, default=256)
    run.add_argument("--start", type=int, default=0)
    run.add_argument("--stop", type=int)
    run.add_argument("--indices", type=int, nargs="+")
    run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    merge = commands.add_parser("merge")
    merge.add_argument("--results", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--results", type=Path, required=True)
    verify.add_argument("--package", type=Path)
    test = commands.add_parser("self-test")
    test.add_argument("--device", default="cpu")
    plot = commands.add_parser("plot")
    plot.add_argument("--results", type=Path, required=True)
    plot.add_argument("--output", type=Path, required=True)
    plot.add_argument("--gateway", type=float)
    plot.add_argument("--gateway-min", type=float)
    plot.add_argument("--gateway-max", type=float)
    plot.add_argument("--temperature", type=float)
    plot.add_argument("--display-sigma", type=float, default=0.005)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "inspect":
        if args.engine == "orca":
            from orca_dos_v2 import inspect_source as inspect_orca
            inspect_orca(args.source, args.output)
        else:
            inspect_source(args.source, args.output)
    elif args.command == "export":
        export_package(args.inspection, args.output, {key: getattr(args, key) for key in DEFAULTS})
    elif args.command == "prepare":
        prepare_material(args.data, args.probe_overrides)
    elif args.command == "run":
        run_gpu(args)
    elif args.command == "merge":
        merge_results(args.results)
    elif args.command == "verify":
        if not (args.results / "dos2d_results.npz").is_file():
            merge_results(args.results)
        verify_results(args.results, args.package)
    elif args.command == "plot":
        from plot_dna_dos_v2 import run_plot
        run_plot(args.results, args.output, args.gateway, args.temperature, args.display_sigma,
                 (args.gateway_min, args.gateway_max))
    else:
        self_test(args.device)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error
