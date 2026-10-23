"""ORCA 6.0 JSON input adapter for the shared full-space DNA V2 solver."""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import uuid

import numpy as np
from numpy.lib.format import open_memmap

import dna_dos_v2 as workflow


SHELL_SIZE = {shell: 2 * angular + 1 for angular, shell in enumerate("spdfghi")}


def source_files(source: Path) -> dict[str, Path]:
    def unique(preferred, suffix):
        if (source / preferred).is_file():
            return source / preferred
        matches = [p for p in source.iterdir() if p.suffix.lower() == suffix and
                   not p.name.endswith((".property.json", "_preparation.json"))]
        if len(matches) != 1:
            raise ValueError(f"Expected {preferred} or exactly one {suffix} file in {source}.")
        return matches[0]

    files = {"json": unique("dna.json", ".json"), "geometry": unique("dna.inp", ".inp"),
             "pdb": unique("dna_selected.pdb", ".pdb"), "mapping": source / "atom_mapping.csv"}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return files


def json_header(path: Path) -> tuple[dict, list]:
    import ijson
    from ijson.common import ObjectBuilder

    wanted = {"Molecule." + key: key for key in
              ("Charge", "CoordinateUnits", "HFTyp", "Multiplicity")}
    wanted["Molecule.MolecularOrbitals.EnergyUnit"] = "EnergyUnit"
    header, atoms, builder, atoms_done = {}, [], None, False
    with path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle, use_float=True):
            if prefix == "Molecule.Atoms.item" and event == "start_map":
                builder = ObjectBuilder()
            if builder is not None:
                builder.event(event, value)
                if prefix == "Molecule.Atoms.item" and event == "end_map":
                    atoms.append(builder.value)
                    builder = None
            if prefix in wanted and event in ("number", "string"):
                header[wanted[prefix]] = value
            if prefix == "Molecule.Atoms" and event == "end_array":
                atoms_done = True
            if len(header) == len(wanted) and atoms_done:
                break
    if len(header) != len(wanted) or not atoms:
        raise ValueError("ORCA JSON lacks atoms, molecular metadata or MO EnergyUnit; export with orca.json.conf.")
    return header, atoms


def inspect_source(source: Path, output: Path) -> dict:
    files = source_files(source)
    states = workflow.file_states(files)
    report_path = output / "inspection.json"
    if report_path.is_file() and (output / "partitions.json").is_file():
        previous = workflow.read_json(report_path)
        if previous.get("protocol") == workflow.ORCA_PROTOCOL and previous.get("source_files") == states:
            return previous
    metadata, atoms = json_header(files["json"])
    if metadata["HFTyp"].upper() not in ("RHF", "RKS") or metadata["Multiplicity"] != 1:
        raise ValueError("Only real, closed-shell RHF/RKS ORCA orbitals are supported.")
    if metadata["EnergyUnit"].lower() not in ("eh", "hartree"):
        raise ValueError(f"Unsupported ORCA orbital energy unit: {metadata['EnergyUnit']}.")
    unit = metadata["CoordinateUnits"].lower()
    if unit not in ("angs", "angstrom", "bohr", "a.u."):
        raise ValueError(f"Unsupported ORCA coordinate unit: {unit}.")
    scale = workflow.gaussian.BOHR_TO_ANGSTROM if unit in ("bohr", "a.u.") else 1.
    if [atom["Idx"] for atom in atoms] != list(range(len(atoms))):
        raise ValueError("ORCA atom indices must be consecutive and in input order.")
    counts = []
    for atom in atoms:
        if atom["ElementLabel"].title() != workflow.ELEMENTS.get(atom["ElementNumber"]):
            raise ValueError("Unsupported element or inconsistent ORCA atomic identity.")
        try:
            counts.append(sum(SHELL_SIZE[shell["Shell"].lower()] for shell in atom["Basis"]))
        except KeyError as error:
            raise ValueError("Unsupported ORCA basis shell; export full spherical basis metadata.") from error
        if counts[-1] < 1:
            raise ValueError("Every atom must have at least one AO.")
    geometry = workflow.v2.load_geometry(files["geometry"])
    text = files["geometry"].read_text(encoding="utf-8")
    match = re.search(r"(?im)^\s*\*\s*xyz\s+(-?\d+)\s+(\d+)\s*$", text)
    if not match or (int(match[1]), int(match[2])) != (metadata["Charge"], metadata["Multiplicity"]):
        raise ValueError("ORCA input and JSON charge/multiplicity differ; inline * xyz geometry is required.")
    n = sum(counts)
    header = {"n_basis": n, "n_atoms": len(atoms),
              "atomic_numbers": np.asarray([a["ElementNumber"] for a in atoms]),
              "coordinates_A": np.asarray([a["Coords"] for a in atoms]) * scale,
              "ao_to_atom_1based": np.repeat(np.arange(1, len(atoms) + 1), counts)}
    partition = workflow.make_partition(header, files["mapping"], files["geometry"], files["pdb"],
                                        atom_index_key="orca_atom_1based", atoms=geometry)
    electrons = int(header["atomic_numbers"].sum() - metadata["Charge"])
    if electrons < 0 or electrons % 2:
        raise ValueError("Inconsistent closed-shell ORCA electron count.")
    if workflow.file_states(files) != states:
        raise ValueError("ORCA input changed during inspection; wait for export to finish.")
    output.mkdir(parents=True, exist_ok=True)
    workflow.save_json(output / "partitions.json", partition)
    report = {"protocol": workflow.ORCA_PROTOCOL, "engine": "orca", "source": str(source.resolve()),
              "source_files": states, "state_count": n, "atom_count": len(atoms),
              "charge": int(metadata["Charge"]), "multiplicity": 1, "electron_count": electrons,
              "nucleotide_count": len(partition["nucleotides"]), "base_pair_count": len(partition["base_pair_order"]),
              "contacts": partition["contacts"], "expected_boundary_candidates": len(partition["boundaries"]),
              "strand_sequences_5to3": partition["strand_sequences_5to3"], "ao_coverage_exactly_once": True,
              "geometry_json_max_abs_A": partition["coordinate_max_abs_A"], "basis_representation": "ORCA spherical",
              "matrix_payload_validation": "Full MO/S dimensions and finite values checked at export; C.T S C checked at prepare.",
              "scf_convergence": "Not independently verified; retain and review the ORCA normal-termination log.",
              "float64_square_GiB": n * n * 8 / 2**30,
              "c64_H_A_LU_GiB_excluding_workspace": n * n * 20 / 2**30}
    workflow.save_json(report_path, report)
    print(f"ORCA mapping passed: {n} states, {report['base_pair_count']} positions.", flush=True)
    return report


def extract_arrays(path: Path, inputs: Path, report: dict) -> None:
    import ijson

    n = report["state_count"]
    coefficients = open_memmap(inputs / "mo_coefficients.npy", mode="w+", dtype="float64", shape=(n, n), fortran_order=True)
    eps, occupations, count = np.empty(n), np.empty(n), 0
    with path.open("rb") as handle:
        for count, mo in enumerate(ijson.items(handle, "Molecule.MolecularOrbitals.MOs.item", use_float=True), 1):
            column = np.asarray(mo["MOCoefficients"], dtype=float)
            if count > n or column.shape != (n,) or not np.isfinite(column).all():
                raise ValueError("ORCA requires complete square finite MO coefficients; no rank truncation or spin blocks.")
            coefficients[:, count - 1] = column
            eps[count - 1], occupations[count - 1] = mo["OrbitalEnergy"], mo["Occupancy"]
            if count % 500 == 0:
                print(f"Extracted MO {count}/{n}", flush=True)
    if count != n or not np.isfinite(eps).all() or not np.isfinite(occupations).all():
        raise ValueError("ORCA MO data is incomplete or nonfinite.")
    if not np.all(np.isclose(occupations, 0., atol=1e-8) | np.isclose(occupations, 2., atol=1e-8)) or not np.isclose(occupations.sum(), report["electron_count"], atol=1e-6):
        raise ValueError("ORCA MO occupations disagree with the closed-shell electron count.")
    coefficients.flush()
    del coefficients
    np.save(inputs / "orbital_energies_hartree.npy", eps)
    np.save(inputs / "occupations.npy", occupations)
    overlap = open_memmap(inputs / "overlap.npy", mode="w+", dtype="float64", shape=(n, n))
    count = 0
    with path.open("rb") as handle:
        for count, row in enumerate(ijson.items(handle, "Molecule.S-Matrix.item", use_float=True), 1):
            values = np.asarray(row, dtype=float)
            if count > n or values.shape != (n,) or not np.isfinite(values).all():
                raise ValueError("ORCA S-Matrix must be a finite square AO overlap matrix.")
            overlap[count - 1] = values
    if count != n:
        raise ValueError("ORCA JSON lacks the complete S-Matrix; request 1elIntegrals: [S], not H.")
    overlap.flush()
    if workflow.gaussian.block_symmetry_error(overlap) > 1e-10:
        raise ValueError("ORCA overlap is not symmetric.")


def export_package(inspection: Path, output: Path, parameters: dict) -> dict:
    report = workflow.read_json(inspection / "inspection.json")
    files = {key: Path(row["path"]) for key, row in report["source_files"].items()}
    if workflow.file_states(files) != report["source_files"]:
        raise ValueError("ORCA source changed after inspection; inspect again.")
    energies = workflow.energy_grid(parameters)
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    manifest = {"protocol": workflow.ORCA_PROTOCOL, "engine": "orca", "run_id": str(uuid.uuid4()), "phase": "exporting",
                "state_count": report["state_count"], "atom_count": report["atom_count"],
                "base_pair_count": report["base_pair_count"], "parameters": parameters, "source": report,
                "units": {"orbital_energies": "hartree", "overlap": "dimensionless", "geometry": "angstrom"}}
    workflow.save_json(output / "manifest.json", manifest)
    extract_arrays(files["json"], inputs, report)
    shutil.copy2(inspection / "partitions.json", output / "partitions.json")
    for role, name in (("geometry", "dna.inp"), ("pdb", "dna_selected.pdb"), ("mapping", "atom_mapping.csv")):
        shutil.copy2(files[role], inputs / name)
    np.save(output / "energies_eV.npy", energies)
    workflow.write_package_runtime(output)
    if workflow.file_states(files) != report["source_files"]:
        raise ValueError("ORCA source changed while exporting; package remains incomplete.")
    manifest["phase"] = "exported"
    (output / "GPU_GUIDE.md").write_text(workflow.gpu_guide(manifest), encoding="utf-8")
    workflow.save_json(output / "manifest.json", manifest)
    print(f"ORCA GPU package exported: {output}", flush=True)
    return manifest
