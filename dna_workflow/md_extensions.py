"""Box comparison and representative-frame relaxation helpers for DNA jobs.

These commands prepare inputs only.  They never call ``gmx mdrun``.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

import md_pipeline as md
from dna_workflow import ENVIRONMENT, pdb_text, read_pdb, restore_identity, save_json, select_dna_atoms, validate_dna


BOX_TYPES = ("cubic", "dodecahedron", "octahedron", "triclinic")


def _require_new_directory(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True)
    return path


def _run(gmx: str, parts: list[str], cwd: Path, log: Path) -> None:
    argv = [gmx, *map(str, parts)]
    completed = subprocess.run(argv, cwd=cwd, text=True, encoding="utf-8", errors="replace",
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text("$ " + " ".join(argv) + "\n" + completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, argv, output=completed.stdout)


def _gro_summary(path: Path) -> tuple[int, int, int, list[list[float]], float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        total = int(lines[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid GRO atom count: {path}") from exc
    if len(lines) != total + 3:
        raise ValueError(f"Incomplete GRO file: {path}")
    water_atoms = 0
    water_molecules = 0
    previous_water = None
    for line in lines[2:2 + total]:
        residue = line[5:10].strip().upper()
        if residue not in {"SOL", "HOH", "WAT", "TIP3"}:
            previous_water = None
            continue
        water_atoms += 1
        key = (line[:5], residue)
        if key != previous_water:
            water_molecules += 1
            previous_water = key
    values = [float(value) for value in lines[-1].split()]
    if len(values) == 3:
        vectors = [[values[0], 0.0, 0.0], [0.0, values[1], 0.0], [0.0, 0.0, values[2]]]
    elif len(values) == 9:
        vectors = [[values[0], values[3], values[4]], [values[5], values[1], values[6]],
                   [values[7], values[8], values[2]]]
    else:
        raise ValueError(f"Invalid GRO box vectors in {path}")
    volume = abs(sum(vectors[0][i] * (vectors[1][(i + 1) % 3] * vectors[2][(i + 2) % 3]
                                      - vectors[1][(i + 2) % 3] * vectors[2][(i + 1) % 3])
                     for i in range(3)))
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError(f"Non-positive GRO box volume in {path}")
    return total, water_molecules, water_atoms, vectors, volume


def compare_boxes(args) -> dict:
    """Run editconf/solvate for each supported GROMACS box shape, before salt addition."""
    job = args.job.resolve()
    topology = job / "topology.pdb"
    if not topology.is_file():
        raise FileNotFoundError(f"Missing topology.pdb: {topology}")
    if not math.isfinite(args.padding) or args.padding <= 0:
        raise ValueError("padding must be a positive number in nm")
    output = _require_new_directory(args.output)
    dna = [atom for atom in read_pdb(topology) if atom.residue not in ENVIRONMENT]
    if len(dna) != len(read_pdb(topology)):
        raise ValueError("topology.pdb contains environment residues; compare from the DNA-only topology reference")
    rows = []
    try:
        for box_type in BOX_TYPES:
            folder = output / box_type
            folder.mkdir()
            boxed = folder / "boxed.gro"
            solvated = folder / "solvated.gro"
            _run(args.gmx, ["editconf", "-f", str(topology), "-o", str(boxed), "-c", "-d", str(args.padding),
                            "-bt", box_type], job, folder / "editconf.log")
            # No -p: this keeps the source job's topol.top untouched and this is deliberately pre-salt.
            _run(args.gmx, ["solvate", "-cp", str(boxed), "-cs", "spc216.gro", "-o", str(solvated)],
                 job, folder / "solvate.log")
            total, waters, water_atoms, vectors, volume = _gro_summary(solvated)
            rows.append({"box_type": box_type, "V_nm3": volume, "DNA_atoms": len(dna),
                         "water_molecules": waters, "water_atoms": water_atoms, "total_atoms": total,
                         "box_vectors": vectors, "salt_added": False})
    except Exception:
        save_json(output / "report.json", {"operation": "compare_boxes", "status": "failed", "job": str(job),
                                            "topology_pdb": str(topology), "padding_nm": args.padding,
                                            "salt_added": False, "comparisons": rows})
        raise
    report = {"operation": "compare_boxes", "status": "complete", "job": str(job),
              "topology_pdb": str(topology), "padding_nm": args.padding, "gmx": args.gmx,
              "salt_added": False, "salt_note": "Pre-salt solvation only; no genion step was run.",
              "comparisons": rows,
              "validation": "Each row is from actual GROMACS editconf then solvate using the same topology.pdb; source topol.top was not supplied or modified."}
    save_json(output / "report.json", report)
    with (output / "box_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["box_type", "V_nm3", "DNA_atoms", "water_molecules",
                                                     "water_atoms", "total_atoms", "box_vectors", "salt_added"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "box_vectors": json.dumps(row["box_vectors"])})
    print(json.dumps(report, ensure_ascii=False))
    return report


def _dna_topology_text(source: Path) -> str:
    """Keep original includes/conditionals, replacing only environment-dependent molecules."""
    text = source.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^\s*\[\s*molecules\s*\]\s*\n(?P<body>.*)\Z", text)
    if not match:
        raise ValueError("Source topol.top has no [ molecules ] section")
    dna_itps = sorted(source.parent.glob("topol_DNA_chain*.itp"))
    if not dna_itps:
        raise ValueError("Source topol.top has no DNA chain topology includes")
    moltypes = set()
    for itp in dna_itps:
        body = itp.read_text(encoding="utf-8")
        molecule = re.search(r"(?ms)^\s*\[\s*moleculetype\s*\]\s*\n\s*(?:;[^\n]*\n\s*)*(\S+)", body)
        if not molecule:
            raise ValueError(f"Cannot read [ moleculetype ] from {itp.name}")
        moltypes.add(molecule.group(1))
    dna_lines = []
    for raw in match.group("body").splitlines():
        fields = raw.split(";", 1)[0].split()
        if fields and fields[0] in moltypes:
            dna_lines.append(raw)
    if not dna_lines:
        raise ValueError("Source [ molecules ] contains no entries matching DNA chain [ moleculetype ] names")
    return text[:match.start()] + "[ molecules ]\n" + "\n".join(dna_lines) + "\n"


def _unrestrained_em_mdp(source: Path) -> tuple[str, list[str]]:
    """Use the parent EM settings while removing position-restraint preprocessor flags."""
    removed = []
    lines = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        content, comment = raw.split(";", 1) if ";" in raw else (raw, "")
        if "=" not in content or content.split("=", 1)[0].strip().lower().replace("_", "-") != "define":
            lines.append(raw)
            continue
        key, value = content.split("=", 1)
        kept = [flag for flag in value.split() if not flag.upper().startswith("-DPOSRES")]
        removed.extend(flag for flag in value.split() if flag.upper().startswith("-DPOSRES"))
        if kept:
            lines.append(f"{key}= {' '.join(kept)}" + (f" ; {comment}" if comment else ""))
        else:
            lines.append("; define removed for unrestrained representative EM" + (f": {comment}" if comment else ""))
    return "\n".join(lines) + "\n", removed


def _copy_relax_inputs(parent: Path, child: Path, cfg: dict) -> None:
    ff = parent / cfg["forcefield"]
    if not ff.is_dir():
        raise FileNotFoundError(f"Missing source force field: {ff}")
    shutil.copytree(ff, child / ff.name)
    for item in parent.glob("*DNA_chain*.itp"):
        shutil.copy2(item, child / item.name)
    for item in parent.glob("posre_DNA_chain*.itp"):
        shutil.copy2(item, child / item.name)
    (child / "topol.top").write_text(_dna_topology_text(parent / "topol.top"), encoding="utf-8")


def _command_for_job(job: Path, gmx: str):
    def command(parts, label, input_text=None, capture=False, echo=True):
        argv = [gmx, *map(str, parts)]
        logs = job / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / "commands.log").open("a", encoding="utf-8") as stream:
            stream.write(" ".join(argv) + "\n")
        return md.stream_process(argv, job, logs / f"{label}.log", input_text, capture, echo)
    return command


def prepare_relax(args) -> dict:
    """Create an independent, unrestrained-EM child job from one representative all-atom DNA frame."""
    parent = args.job.resolve()
    frame = args.frame.resolve()
    if not frame.is_file():
        raise FileNotFoundError(f"Missing representative frame: {frame}")
    if args.box_type not in BOX_TYPES or not math.isfinite(args.padding) or args.padding <= 0:
        raise ValueError("Use a supported box type and positive padding")
    if not math.isfinite(args.salt) or args.salt < 0:
        raise ValueError("salt must be non-negative")
    cfg = json.loads((parent / "workflow.json").read_text(encoding="utf-8"))
    reference = restore_identity(read_pdb(parent / "topology.pdb"), cfg["strands_5to3"])
    validate_dna(reference, cfg["strands_5to3"], parent / cfg["forcefield"], hydrogens=True)
    selected = select_dna_atoms(read_pdb(frame), reference)
    selected = [replace(a, serial=i + 1, xyz=b.xyz) for i, (a, b) in enumerate(zip(reference, selected))]
    validate_dna(selected, cfg["strands_5to3"], parent / cfg["forcefield"], hydrogens=True)
    child = _require_new_directory(args.output)
    try:
        _copy_relax_inputs(parent, child, cfg)
        shutil.copy2(frame, child / "source_frame.pdb")
        (child / "topology.pdb").write_text(pdb_text(selected), encoding="ascii")
        (child / "selected_source_frame.pdb").write_text(pdb_text(selected), encoding="ascii")
        child_cfg = {**cfg, "role": "representative_relaxation", "source_job": str(parent), "source_frame": str(frame),
                     "source_frame_copy": "source_frame.pdb", "selected_source_frame": "selected_source_frame.pdb",
                     "topology_reference": "topology.pdb", "box_type": args.box_type, "padding_nm": args.padding,
                     "salt_M": args.salt, "parent_provenance": {"source_job": str(parent), "source_frame": str(frame)},
                     "model": "Representative all-atom MD frame re-solvated and ionized for unrestrained EM; EM can change conformation and does not guarantee Gaussian SCF improvement."}
        save_json(child / "workflow.json", child_cfg)
        em_mdp, removed_posres = _unrestrained_em_mdp(parent / "em.mdp")
        (child / "em.mdp").write_text(em_mdp, encoding="ascii")
        command = _command_for_job(child, args.gmx)
        command(["editconf", "-f", "topology.pdb", "-o", "boxed.gro", "-c", "-d", str(args.padding), "-bt", args.box_type], "box")
        command(["solvate", "-cp", "boxed.gro", "-cs", "spc216.gro", "-o", "solvated.gro", "-p", "topol.top"], "solvate")
        (child / "ions.mdp").write_text("integrator = steep\nnsteps = 0\ncutoff-scheme = Verlet\n"
                                         "coulombtype = Cut-off\nrcoulomb = 1.2\nrvdw = 1.2\npbc = xyz\n", encoding="ascii")
        command(["grompp", "-f", "ions.mdp", "-c", "solvated.gro", "-p", "topol.top", "-o", "ions.tpr", "-po", "ions.processed.mdp"], "ions-grompp")
        command(["genion", "-s", "ions.tpr", "-o", "solv_ions.gro", "-p", "topol.top", "-pname", "SOD", "-nname", "CLA",
                 "-neutral", "-conc", str(args.salt), "-seed", str(cfg["seed"])], "genion", "SOL\n")
        save_json(child / "solvate.complete.json", {"water": "CHARMM TIP3P", "additional_salt_M": args.salt,
                                                     "neutralize": True, "box_type": args.box_type, "padding_nm": args.padding})
        md.run(SimpleNamespace(job=child, gmx=args.gmx, stage="prepare_em", ntmpi=None, ntomp=None, maxh=None,
                               noninteractive_terminals=False, terminal_timeout=60.0))
    except Exception:
        raise
    report = {"operation": "prepare_relax", "child_job": str(child), "role": "representative_relaxation",
              "source_job": str(parent), "source_frame": str(frame), "selected_dna_atoms": len(selected),
              "box_type": args.box_type, "padding_nm": args.padding, "salt_M": args.salt,
              "source_frame_copy": str(child / "source_frame.pdb"), "selected_source_frame": str(child / "selected_source_frame.pdb"),
              "em_tpr": str(child / "em.tpr"), "mdrun_executed": False, "unrestrained_em": True,
              "removed_posres_defines": removed_posres,
              "note": "Run em.mdrun on the server, return em.tpr/em.gro/em.log, then call verify_relax."}
    save_json(child / "prepare_relax.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return report


def _gro_dna_atoms(path: Path, reference) -> list:
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[1])
    if len(lines) != count + 3:
        raise ValueError("Incomplete em.gro")
    if count < len(reference):
        raise ValueError("em.gro has fewer atoms than DNA topology")
    result = []
    for atom, line in zip(reference, lines[2:2 + len(reference)]):
        if (line[5:10].strip(), line[10:15].strip()) != (atom.residue, atom.name):
            raise ValueError("em.gro does not start with the original DNA atom order; refusing environment/Gaussian mix-up")
        xyz = tuple(10.0 * float(line[i:i + 8]) for i in (20, 28, 36))
        result.append(replace(atom, xyz=xyz))
    return result


def _raw_rmsd(a, b) -> float:
    if len(a) != len(b):
        raise ValueError("RMSD atom count mismatch")
    return math.sqrt(sum(sum((x - y) ** 2 for x, y in zip(left.xyz, right.xyz)) for left, right in zip(a, b)) / len(a))


def _whole_em_dna(child: Path, reference, strand_count: int, command) -> list:
    """Restore periodic images using EM topology, without altering returned coordinates."""
    _gro_dna_atoms(child / "em.gro", reference)  # Check atom order before selecting by index.
    with tempfile.TemporaryDirectory(prefix="dna_em_pbc_") as temporary:
        root = Path(temporary)
        index, whole = root / "dna.ndx", root / "whole.gro"
        ids = list(range(1, len(reference) + 1))
        index.write_text("[ DNA ]\n" + "\n".join(" ".join(map(str, ids[i:i+15])) for i in range(0, len(ids), 15)) + "\n", encoding="ascii")
        command(["trjconv", "-s", "em.tpr", "-f", "em.gro", "-n", str(index),
                 "-o", str(whole), "-pbc", "whole"], "relax-dna-whole", "0\n")
        if strand_count > 1:
            tpr, clustered = root / "dna.tpr", root / "clustered.gro"
            command(["convert-tpr", "-s", "em.tpr", "-n", str(index), "-o", str(tpr)], "relax-dna-tpr", "0\n")
            command(["trjconv", "-s", str(tpr), "-f", str(whole), "-n", str(index),
                     "-o", str(clustered), "-pbc", "cluster"], "relax-dna-cluster", "0\n0\n")
            whole = clustered
        return _gro_dna_atoms(whole, reference)


def verify_relax(args) -> dict:
    """Verify returned EM, extract DNA only, restore chain identities, and report EM displacement."""
    child = args.job.resolve()
    cfg = json.loads((child / "workflow.json").read_text(encoding="utf-8"))
    if cfg.get("role") != "representative_relaxation":
        raise ValueError("verify_relax requires a representative_relaxation child job")
    command = _command_for_job(child, args.gmx)
    em = md.verify_return(child, "em", command)
    reference = restore_identity(read_pdb(child / "topology.pdb"), cfg["strands_5to3"])
    relaxed = _whole_em_dna(child, reference, len(cfg["strands_5to3"]), command)
    check = validate_dna(relaxed, cfg["strands_5to3"], child / cfg["forcefield"], hydrogens=True)
    (child / "relaxed_dna.pdb").write_text(pdb_text(relaxed), encoding="ascii")
    rmsd_raw = _raw_rmsd(reference, relaxed)
    report = {"operation": "verify_relax", "role": "representative_relaxation", "input_frame": cfg["source_frame"],
              "relaxed_dna_pdb": str(child / "relaxed_dna.pdb"), "atom_count": len(relaxed), "charge": check["charge"],
              "Fmax_kJ_mol_nm": em["Fmax_kJ_mol_nm"], "rmsd_raw_A": rmsd_raw, "rmsd_fit": False,
              "export_files": md.file_state(child, ["em.tpr", "em.gro", "em.log", "em.mdp", "relaxed_dna.pdb", "topology.pdb"]),
              "rmsd_definition": "all DNA atoms after common topology order; unfit Cartesian RMSD includes box-centering translation and rotation, so it is not a pure deformation metric; coordinates nm->A",
              "chain_identity": "restored from child topology.pdb", "environment_exported": False,
              "pbc_reconstruction": ["GROMACS em.tpr-based trjconv -pbc whole",
                                     *(["DNA-only trjconv -pbc cluster"] if len(cfg["strands_5to3"]) > 1 else [])],
              "covalent_validation": check["validation"],
              "caveat": "Unrestrained EM may alter the representative conformation; it does not guarantee improved Gaussian SCF convergence."}
    save_json(child / "relaxation_report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    p = sub.add_parser("compare_boxes", help="Compare pre-salt GROMACS solvent boxes without changing the source job")
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gmx", required=True)
    p.add_argument("--padding", type=float, default=1.5)
    p.set_defaults(func=compare_boxes)
    p = sub.add_parser("prepare_relax", help="Re-solvate and prepare unrestrained EM for one representative all-atom frame")
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--frame", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gmx", required=True)
    p.add_argument("--padding", type=float, default=1.5)
    p.add_argument("--salt", type=float, default=0.15)
    p.add_argument("--box-type", choices=BOX_TYPES, default="dodecahedron")
    p.set_defaults(func=prepare_relax)
    p = sub.add_parser("verify_relax", help="Verify returned EM and export DNA-only relaxed_dna.pdb")
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--gmx", required=True)
    p.set_defaults(func=verify_relax)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
