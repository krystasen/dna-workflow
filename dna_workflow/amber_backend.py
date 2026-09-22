"""AmberTools OL21/OL24 + OPC topology and solvation backend.

Run this script with the workflow Python.  ParmEd is deliberately imported only
by the configured Amber Python because that is the environment that owns the
Amber/ParmEd installation.  This backend prepares GROMACS inputs only; it never
runs ``mdrun``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


AMBER_ENV = "/home/chenz/.local/share/dna-amber/env"
DEFAULT_TLEAP = AMBER_ENV + "/bin/tleap"
DEFAULT_PYTHON = AMBER_ENV + "/bin/python"
AMBER_MODELS = {"ol21_opc": "OL21", "ol24_opc": "OL24"}
DNA_BASES = {"A", "C", "G", "T"}
ION_NAMES = {"NA", "NA+", "SOD", "K", "K+", "POT", "CL", "CL-", "CLA"}
ELEMENTS = {0: "", 1: "H", 6: "C", 7: "N", 8: "O", 15: "P"}


class AmberBackendError(RuntimeError):
    """A supported Amber workflow could not be prepared."""


def _normalise_residue(name: str) -> str | None:
    match = re.fullmatch(r"D([ACGT])(?:[35])?", name.strip().upper())
    return "D" + match.group(1) if match else None


def _normalise_cfg(cfg: dict[str, Any]) -> tuple[str, list[str]]:
    model = cfg.get("md_model")
    if model == "charmm36_tip3p":
        raise AmberBackendError("charmm36_tip3p must use the existing GROMACS backend, not amber_backend.py.")
    if model not in AMBER_MODELS:
        raise AmberBackendError("md_model must be ol21_opc or ol24_opc for the Amber backend.")
    strands = cfg.get("strands_5to3")
    if not isinstance(strands, list) or not strands or any(not isinstance(s, str) or not s or set(s.upper()) - DNA_BASES for s in strands):
        raise AmberBackendError("workflow.json requires canonical non-empty strands_5to3 for Amber DNA topology.")
    return model, [s.upper() for s in strands]


def _validate_box(cfg: dict[str, Any]) -> tuple[str, float]:
    box_type = cfg.get("box_type")
    if box_type == "dodecahedron":
        raise AmberBackendError("AmberTools backend does not support dodecahedron; explicitly select cubic, triclinic, or octahedron.")
    if box_type not in {"cubic", "triclinic", "octahedron"}:
        raise AmberBackendError("Amber box_type must be cubic, triclinic, or octahedron.")
    try:
        padding = float(cfg["padding_nm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AmberBackendError("workflow.json requires numeric padding_nm.") from exc
    if not math.isfinite(padding) or padding <= 0:
        raise AmberBackendError("padding_nm must be finite and positive.")
    return box_type, padding


def _salt_settings(cfg: dict[str, Any]) -> tuple[str, float]:
    cation = cfg.get("salt_cation", "Na+")
    if cation not in {"Na+", "K+"}:
        raise AmberBackendError("salt_cation must be Na+ or K+ for OPC.")
    try:
        salt_m = float(cfg["salt_M"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AmberBackendError("workflow.json requires numeric salt_M.") from exc
    if not math.isfinite(salt_m) or salt_m < 0:
        raise AmberBackendError("salt_M must be finite and non-negative.")
    return cation, salt_m


def _tool_path(path: str | Path) -> str:
    """Make a workspace path usable by a WSL executable without requiring wslpath."""
    value = str(path)
    if os.name == "nt" and re.match(r"^[A-Za-z]:[\\/]", value):
        return "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    return value.replace("\\", "/")


def _tool_argv(executable: str, args: list[str]) -> list[str]:
    # Workflow config supplies WSL Amber paths while the GUI starts this script
    # with its Windows Python.  Native paths continue to run directly.
    return (["wsl.exe", "--exec", executable, *args]
            if os.name == "nt" and executable.startswith("/") else [executable, *args])


def _run(argv: list[str], cwd: Path, log: Path, extra_env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(exist_ok=True)
    try:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        result = subprocess.run(argv, cwd=cwd, env=env, text=True, encoding="utf-8", errors="replace",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError as exc:
        raise AmberBackendError(f"Required executable was not found: {argv[0]}") from exc
    log.write_text("$ " + " ".join(argv) + "\n" + (result.stdout or ""), encoding="utf-8")
    if result.returncode:
        raise AmberBackendError(f"Command failed ({result.returncode}); inspect {log.name}.")


def _leap_quote(path: Path) -> str:
    return '"' + _tool_path(path).replace('"', '\\"') + '"'


def _run_tleap(job: Path, tleap: str, label: str, text: str) -> None:
    script = job / "logs" / f"{label}.in"
    script.parent.mkdir(exist_ok=True)
    script.write_text(text, encoding="ascii")
    # Amber's shell wrapper uses unquoted ``$*`` and breaks WSL paths containing
    # spaces. Call the teLeap binary directly and provide its include paths.
    amberhome = str(Path(tleap).parent.parent).replace("\\", "/")
    executable = str(Path(tleap).with_name("teLeap")) if Path(tleap).name == "tleap" else tleap
    args = ["-I" + amberhome + "/dat/leap/prep", "-I" + amberhome + "/dat/leap/lib",
            "-I" + amberhome + "/dat/leap/parm", "-I" + amberhome + "/dat/leap/cmd",
            "-f", _tool_path(script)]
    _run(_tool_argv(executable, args), job, job / "logs" / f"{label}.log", {"AMBERHOME": amberhome})


def _prepare_input_pdb(source: Path, destination: Path, strands: list[str]) -> dict[str, Any]:
    """Preserve PDB order, force one TER per declared strand, and map C5M -> C7."""
    raw = source.read_text(encoding="utf-8").splitlines()
    atoms: list[tuple[str, str, int, str]] = []
    segments: list[list[tuple[str, str, int, str]]] = []
    current: list[tuple[str, str, int, str]] = []
    last_chain: str | None = None
    output: list[str] = []
    changed_c5m = 0
    for line in raw:
        record = line[:6].strip()
        if record not in {"ATOM", "HETATM"}:
            if record == "TER" and current:
                segments.append(current)
                current = []
                last_chain = None
            continue
        line = line.ljust(80)
        residue = _normalise_residue(line[17:20])
        chain = line[21].strip()
        try:
            resid = int(line[22:26])
        except ValueError as exc:
            raise AmberBackendError(f"{source.name}: invalid PDB residue number.") from exc
        if residue is None:
            raise AmberBackendError(f"{source.name}: unsupported residue {line[17:20].strip()!r}; canonical D[ACGT] only.")
        if last_chain is not None and chain != last_chain and current:
            segments.append(current)
            current = []
        last_chain = chain
        name = line[12:16].strip()
        if name == "C5M":
            name = "C7"
            line = line[:12] + " C7 " + line[16:]
            changed_c5m += 1
        item = (line, residue, resid, chain)
        atoms.append(item)
        current.append(item)
    if current:
        segments.append(current)
    if not atoms:
        raise AmberBackendError(f"{source.name}: no ATOM/HETATM records.")
    if len(segments) != len(strands):
        raise AmberBackendError(f"{source.name}: found {len(segments)} DNA strand segment(s), expected {len(strands)}; use separate chains or TER records.")
    observed: list[str] = []
    for segment in segments:
        residues: list[tuple[str, int]] = []
        for _, residue, resid, _ in segment:
            if not residues or residues[-1] != (residue, resid):
                residues.append((residue, resid))
        observed.append("".join(name[1] for name, _ in residues))
    if observed != strands:
        raise AmberBackendError(f"{source.name}: residue sequences {observed} differ from workflow strands_5to3 {strands}.")
    for number, segment in enumerate(segments):
        if number:
            output.append("TER")
        output.extend(line for line, _, _, _ in segment)
    destination.write_text("\n".join(output + ["TER", "END", ""]), encoding="ascii")
    return {"input_atoms": len(atoms), "strands": len(segments), "C5M_to_C7": changed_c5m}


def _model_source(model: str) -> str:
    return "leaprc.DNA." + AMBER_MODELS[model]


def _verify_ol24(job: Path, tleap: str) -> None:
    try:
        _run_tleap(job, tleap, "ol24-availability", "source leaprc.DNA.OL24\nquit\n")
    except AmberBackendError as exc:
        raise AmberBackendError("ol24_opc requested, but official AmberTools load file leaprc.DNA.OL24 is unavailable; install an AmberTools release that provides it. Inspect ol24-availability.log for the original tLeap output.") from exc


def _run_parmed(job: Path, amber_python: str, label: str, worker: str, *args: str) -> None:
    argv = [_tool_path(Path(__file__).resolve()), worker, *args]
    _run(_tool_argv(amber_python, argv), job, job / "logs" / f"{label}.log")


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AmberBackendError("Missing or invalid dna_manifest.json; run topology first.") from exc


def prepare_topology(job: Path, cfg: dict[str, Any], tleap: str, python: str) -> dict[str, Any]:
    """Build Amber DNA topology and a dry GROMACS topology without running MD."""
    job = Path(job).resolve()
    model, strands = _normalise_cfg(cfg)
    if model == "ol24_opc":
        _verify_ol24(job, tleap)
    source = job / "dna_input.pdb"
    if not source.is_file():
        raise AmberBackendError("dna_input.pdb is required before topology.")
    manifest = job / "dna_manifest.json"
    if manifest.exists():
        raise AmberBackendError("Topology manifest already exists; use a new job rather than overwriting Amber topology.")
    input_report = _prepare_input_pdb(source, job / "amber_input.pdb", strands)
    leap = (f"source {_model_source(model)}\nsource leaprc.water.opc\n"
            f"dna = loadPdb {_leap_quote(job / 'amber_input.pdb')}\ncheck dna\n"
            f"saveOff dna {_leap_quote(job / 'dna.off')}\n"
            f"saveAmberParm dna {_leap_quote(job / 'dna.prmtop')} {_leap_quote(job / 'dna.inpcrd')}\nquit\n")
    _run_tleap(job, tleap, "amber-topology", leap)
    _run_parmed(job, python, "parmed-dna", "_parmed_export_dna",
                "--prmtop", _tool_path(job / "dna.prmtop"), "--inpcrd", _tool_path(job / "dna.inpcrd"),
                "--topology-pdb", _tool_path(job / "topology.pdb"), "--topol", _tool_path(job / "topol.top"),
                "--gro", _tool_path(job / "topology.gro"),
                "--manifest", _tool_path(manifest), "--model", model, "--strands", json.dumps(strands))
    report = _read_manifest(manifest)
    (job / "topology.complete.json").write_text(json.dumps({
        "stage": "topology", "files": ["dna.off", "dna.prmtop", "dna.inpcrd", "topology.pdb", "topol.top", "topology.gro", "dna_manifest.json"],
        "validation": "AmberTools topology and ParmEd GROMACS export completed; no MD run."}, indent=2) + "\n", encoding="utf-8")
    return report


def _solvation_text(job: Path, model: str, box_type: str, padding_nm: float, cation: str | None = None,
                    salt_pairs: int = 0, counterion: str | None = None, output_prefix: str = "solvated") -> str:
    padding_a = padding_nm * 10.0
    box = (f"solvateBox dna OPCBOX {padding_a:g} iso" if box_type == "cubic" else
           f"solvateBox dna OPCBOX {padding_a:g}" if box_type == "triclinic" else
           f"solvateOct dna OPCBOX {padding_a:g}")
    ions = ""
    if counterion:
        ions += f"addIonsRand dna {counterion} 0\n"
    if salt_pairs:
        assert cation is not None
        ions += f"addIonsRand dna {cation} {salt_pairs}\naddIonsRand dna Cl- {salt_pairs}\n"
    return (f"source {_model_source(model)}\nsource leaprc.water.opc\n"
            f"dna = loadMol2 {_leap_quote(job / 'dna_current.mol2')}\n"
            f"{box}\n{ions}check dna\n"
            f"saveAmberParm dna {_leap_quote(job / (output_prefix + '.prmtop'))} {_leap_quote(job / (output_prefix + '.inpcrd'))}\nquit\n")


def _ion_plan(charge: float, cation: str) -> tuple[str | None, int]:
    rounded = round(charge)
    if not math.isclose(charge, rounded, abs_tol=1e-3):
        raise AmberBackendError(f"DNA topology charge {charge:g} is not an integer; refusing to guess neutralizing ions.")
    return (cation if rounded < 0 else "Cl-" if rounded > 0 else None), abs(rounded)


def solvate(job: Path, cfg: dict[str, Any], tleap: str, python: str) -> dict[str, Any]:
    """Solvate the existing Amber DNA topology without rebuilding it from a PDB."""
    job = Path(job).resolve()
    model, _ = _normalise_cfg(cfg)
    box_type, padding_nm = _validate_box(cfg)
    cation, salt_m = _salt_settings(cfg)
    manifest_path = job / "dna_manifest.json"
    manifest = _read_manifest(manifest_path)
    for required in ("dna.prmtop", "dna.inpcrd"):
        if not (job / required).is_file():
            raise AmberBackendError(f"{required} is required before solvate; run topology first.")
    if (job / "solv_ions.gro").exists():
        raise AmberBackendError("solv_ions.gro already exists; use a new job for re-solvation instead of overwriting a prepared system.")
    _run_parmed(job, python, "parmed-current-dna", "_parmed_export_mol2",
                "--prmtop", _tool_path(job / "dna.prmtop"), "--inpcrd", _tool_path(job / "dna.inpcrd"),
                "--manifest", _tool_path(manifest_path), "--output", _tool_path(job / "dna_current.mol2"))
    _run_tleap(job, tleap, "amber-solvation-box", _solvation_text(job, model, box_type, padding_nm, output_prefix="solvation_box"))
    _run_parmed(job, python, "parmed-box-volume", "_parmed_box_info",
                "--prmtop", _tool_path(job / "solvation_box.prmtop"), "--inpcrd", _tool_path(job / "solvation_box.inpcrd"),
                "--output", _tool_path(job / "solvation_box.json"))
    try:
        box_info = json.loads((job / "solvation_box.json").read_text(encoding="utf-8"))
        volume_nm3 = float(box_info["volume_nm3"])
        charge = float(manifest["charge"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise AmberBackendError("Could not read Amber solvation-box volume or DNA charge.") from exc
    salt_pairs = math.floor(salt_m * volume_nm3 * 0.602214076 + 0.5)
    counterion, neutralizing_ions = _ion_plan(charge, cation)
    _run_tleap(job, tleap, "amber-solvate", _solvation_text(job, model, box_type, padding_nm, cation, salt_pairs,
                                                              counterion, "solvated"))
    _run_parmed(job, python, "parmed-solvated", "_parmed_export_solvated",
                "--prmtop", _tool_path(job / "solvated.prmtop"), "--inpcrd", _tool_path(job / "solvated.inpcrd"),
                "--topol", _tool_path(job / "topol.top"), "--gro", _tool_path(job / "solv_ions.gro"),
                "--manifest", _tool_path(manifest_path),
                "--output", _tool_path(job / "opc_check.json"))
    try:
        opc = json.loads((job / "opc_check.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AmberBackendError("Could not read OPC four-site check from ParmEd.") from exc
    expected_ions = {cation: salt_pairs + (neutralizing_ions if counterion == cation else 0),
                     "Cl-": salt_pairs + (neutralizing_ions if counterion == "Cl-" else 0)}
    observed_ions = {name: int(opc["ion_counts"].get(name, 0)) for name in expected_ions}
    if observed_ions != expected_ions:
        raise AmberBackendError(f"Final ion counts {observed_ions} differ from expected {expected_ions}.")
    report = {
        "box_request": box_type,
        "actual_cell": "cubic" if box_type == "cubic" else "orthorhombic" if box_type == "triclinic" else "truncated_octahedron",
        "padding_nm": padding_nm, "water_model": "OPC", "water_sites": 4,
        "box_volume_nm3_before_ions": volume_nm3, "salt_M_requested": salt_m, "salt_pairs_added": salt_pairs,
        "neutralizing_ion": counterion, "neutralizing_ions_added": neutralizing_ions,
        "salt_cation": cation, "salt_anion": "Cl-", "coordinate_source": "dna.inpcrd",
        "final_files": ["solvated.prmtop", "solvated.inpcrd", "topol.top", "solv_ions.gro"],
        "opc_check": opc,
    }
    (job / "amber_solvation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (job / "solvate.complete.json").write_text(json.dumps({
        "stage": "solvate", "files": ["dna_current.mol2", "solvated.prmtop", "solvated.inpcrd", "topol.top", "solv_ions.gro", "amber_solvation.json"],
        "validation": "AmberTools OPC solvation, ParmEd export, DNA-prefix and four-site-water checks completed; no mdrun executed."}, indent=2) + "\n", encoding="utf-8")
    return report


def _parmed_import():
    try:
        import parmed as pmd
        from parmed.gromacs import GromacsGroFile, GromacsTopologyFile
    except ImportError as exc:
        raise RuntimeError("ParmEd is missing from this Amber Python; install it in cfg amber_python environment.") from exc
    return pmd, GromacsGroFile, GromacsTopologyFile


def _atom_element(atom: Any) -> str:
    atomic_number = int(getattr(atom, "atomic_number", 0) or 0)
    if atomic_number in ELEMENTS:
        return ELEMENTS[atomic_number]
    return re.sub(r"[^A-Za-z]", "", atom.name).upper()[:1]


def _write_pdb(structure: Any, path: Path) -> None:
    lines: list[str] = []
    last_chain: str | None = None
    for index, atom in enumerate(structure.atoms, 1):
        residue = atom.residue
        chain = (getattr(residue, "chain", "") or " ")[:1]
        if last_chain is not None and chain != last_chain:
            lines.append("TER")
        name = atom.name.strip()
        field = f" {name:<3}" if len(name) < 4 else name[:4]
        element = _atom_element(atom)
        lines.append(f"ATOM  {index:5d} {field:4s} {residue.name:>3s} {chain}{int(residue.number):4d}    "
                     f"{atom.xx:8.3f}{atom.xy:8.3f}{atom.xz:8.3f}  1.00  0.00          {element:>2s}")
        last_chain = chain
    path.write_text("\n".join(lines + ["TER", "END", ""]), encoding="ascii")


def _write_posre(path: Path, atom_count: int) -> None:
    lines = ["[ position_restraints ]", "; ai  funct  fx  fy  fz"]
    lines.extend(f"{index:6d}     1  1000  1000  1000" for index in range(1, atom_count + 1))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _moleculetype_blocks(text: str) -> list[dict[str, Any]]:
    headers = list(re.finditer(r"^\s*\[\s*moleculetype\s*\]\s*$", text, re.MULTILINE | re.IGNORECASE))
    system = re.search(r"^\s*\[\s*system\s*\]", text, re.MULTILINE | re.IGNORECASE)
    if not headers or system is None:
        raise RuntimeError("ParmEd topology lacks moleculetype or system sections.")
    blocks = []
    for index, header in enumerate(headers):
        start = header.start()
        end = headers[index + 1].start() if index + 1 < len(headers) else system.start()
        name = next((line.split()[0] for line in text[header.end():end].splitlines()
                     if line.strip() and not line.lstrip().startswith(";")), None)
        if not name:
            raise RuntimeError("ParmEd moleculetype lacks a name.")
        atom_section = re.search(r"^\s*\[\s*atoms\s*\]\s*$", text[start:end], re.MULTILINE | re.IGNORECASE)
        if atom_section is None:
            raise RuntimeError("ParmEd moleculetype lacks an atoms section.")
        atom_start = start + atom_section.end()
        next_section = re.search(r"^\s*\[\s*[^]]+\s*\]", text[atom_start:end], re.MULTILINE)
        atom_end = atom_start + next_section.start() if next_section else end
        atoms = [line.split() for line in text[atom_start:atom_end].splitlines()
                 if line.strip() and not line.lstrip().startswith((";", "#"))]
        if any(len(fields) < 5 for fields in atoms):
            raise RuntimeError("Unexpected ParmEd [ atoms ] line.")
        blocks.append({"start": start, "end": end, "name": name, "residues": [fields[3] for fields in atoms], "atom_count": len(atoms)})
    return blocks


def _molecule_counts(text: str) -> dict[str, int]:
    section = re.search(r"^\s*\[\s*molecules\s*\]\s*$", text, re.MULTILINE | re.IGNORECASE)
    if section is None:
        raise RuntimeError("ParmEd topology lacks a molecules section.")
    end = re.search(r"^\s*\[\s*[^]]+\s*\]", text[section.end():], re.MULTILINE)
    body = text[section.end(): section.end() + end.start()] if end else text[section.end():]
    counts: dict[str, int] = {}
    for line in body.splitlines():
        if not line.strip() or line.lstrip().startswith(";"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise RuntimeError("Unexpected ParmEd [ molecules ] line.")
        counts[fields[0]] = int(fields[1])
    return counts


def _assert_opc_gromacs_sections(topol: Path) -> None:
    text = topol.read_text(encoding="utf-8")
    waters = [block for block in _moleculetype_blocks(text)
              if block["residues"] and all(residue.upper() in {"WAT", "OPC"} for residue in block["residues"])]
    if len(waters) != 1:
        raise RuntimeError("ParmEd topology must contain exactly one OPC water moleculetype.")
    water_text = text[waters[0]["start"]:waters[0]["end"]]
    for section in ("settles", "virtual_sites3"):
        if not re.search(rf"^\s*\[\s*{section}\s*\]\s*$", water_text, re.MULTILINE | re.IGNORECASE):
            raise RuntimeError(f"OPC GROMACS topology lacks required [{section}] section.")


def _add_dna_posres(topol: Path, manifest: dict[str, Any]) -> list[str]:
    text = topol.read_text(encoding="utf-8")
    blocks = _moleculetype_blocks(text)
    counts = _molecule_counts(text)
    dna_blocks = [block for block in blocks if block["residues"] and all(_normalise_residue(residue) for residue in block["residues"])]
    if not dna_blocks:
        raise RuntimeError("ParmEd topology contains no DNA moleculetype.")
    if sum(block["atom_count"] * counts.get(block["name"], 0) for block in dna_blocks) != len(manifest["atoms"]):
        raise RuntimeError("DNA moleculetype instances do not account for all dna_manifest atoms.")
    files = []
    parts = []
    cursor = 0
    for block in blocks:
        end = block["end"]
        parts.append(text[cursor:end])
        if block in dna_blocks:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", block["name"])
            name = f"posre_DNA_{safe_name}.itp"
            _write_posre(topol.parent / name, block["atom_count"])
            files.append(name)
            parts.append(f"\n#ifdef POSRES\n#include \"{name}\"\n#endif\n")
        cursor = end
    parts.append(text[cursor:])
    topol.write_text("".join(parts), encoding="utf-8")
    return files


def _bond_pairs(structure: Any) -> set[tuple[int, int]]:
    return {tuple(sorted((bond.atom1.idx + 1, bond.atom2.idx + 1))) for bond in structure.bonds}


def _assign_dna_identity(structure: Any, strands: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[tuple[int, int]]]:
    expected = [(chr(65 + chain_index), residue_index + 1, "D" + base)
                for chain_index, sequence in enumerate(strands) for residue_index, base in enumerate(sequence)]
    residues = list(structure.residues)
    if len(residues) != len(expected):
        raise RuntimeError(f"Amber topology has {len(residues)} residues; expected {len(expected)} DNA residues.")
    bonds = _bond_pairs(structure)
    atoms: list[dict[str, Any]] = []
    strand_rows: list[dict[str, Any]] = []
    offset = 0
    for residue, (chain, resid, wanted) in zip(residues, expected):
        actual = _normalise_residue(residue.name)
        if actual != wanted:
            raise RuntimeError(f"Amber residue {residue.name!r} does not match expected {wanted}.")
        residue.name, residue.chain, residue.number = wanted, chain, resid
        for atom in residue.atoms:
            atoms.append({"index": atom.idx + 1, "name": atom.name, "residue": wanted, "chain": chain,
                          "resid": resid, "element": _atom_element(atom)})
        offset += len(residue.atoms)
    for chain_index, sequence in enumerate(strands):
        first = residues[sum(len(s) for s in strands[:chain_index])]
        last = residues[sum(len(s) for s in strands[:chain_index + 1]) - 1]
        ids = [atom.idx + 1 for residue in residues[sum(len(s) for s in strands[:chain_index]):sum(len(s) for s in strands[:chain_index + 1])] for atom in residue.atoms]
        terminal_hydrogens: dict[str, dict[str, Any]] = {}
        for label, residue, oxygen in (("5OH", first, "O5'"), ("3OH", last, "O3'")):
            oxygen_atom = next((atom for atom in residue.atoms if atom.name.strip().upper() == oxygen), None)
            if oxygen_atom is None:
                raise RuntimeError(f"Amber topology lacks terminal {oxygen} on chain {chr(65 + chain_index)}.")
            hydrogen = next((atom for atom in residue.atoms
                             if tuple(sorted((oxygen_atom.idx + 1, atom.idx + 1))) in bonds and _atom_element(atom) == "H"), None)
            if hydrogen is None:
                raise RuntimeError(f"Amber topology lacks a hydrogen bonded to terminal {oxygen} on chain {chr(65 + chain_index)}.")
            terminal_hydrogens[label] = {"index": hydrogen.idx + 1, "name": hydrogen.name}
        strand_rows.append({"chain": chr(65 + chain_index), "sequence_5to3": sequence, "atom_indices": ids,
                            "residue_indices": list(range(sum(len(s) for s in strands[:chain_index]) + 1,
                                                          sum(len(s) for s in strands[:chain_index + 1]) + 1)),
                            "terminal_hydrogens": terminal_hydrogens})
    for previous, current in zip(residues, residues[1:]):
        if previous.chain != current.chain:
            continue
        o3 = next((atom for atom in previous.atoms if atom.name.strip().upper() == "O3'"), None)
        p = next((atom for atom in current.atoms if atom.name.strip().upper() == "P"), None)
        if o3 is None or p is None or tuple(sorted((o3.idx + 1, p.idx + 1))) not in bonds:
            raise RuntimeError(f"Amber topology lacks the O3'-P bond before {current.chain}:{current.number}.")
    return atoms, strand_rows, bonds


def _parmed_export_dna(args: argparse.Namespace) -> None:
    pmd, gro_writer, top_writer = _parmed_import()
    structure = pmd.load_file(args.prmtop, args.inpcrd)
    strands = json.loads(args.strands)
    atoms, strand_rows, bonds = _assign_dna_identity(structure, strands)
    _write_pdb(structure, Path(args.topology_pdb))
    top = top_writer.from_structure(structure)
    top.write(args.topol)
    gro_writer.write(top, args.gro)
    atom_to_chain = {item["index"]: item["chain"] for item in atoms}
    interstrand = [list(pair) for pair in sorted(bonds) if atom_to_chain[pair[0]] != atom_to_chain[pair[1]]]
    if interstrand:
        raise RuntimeError("Amber topology contains inter-strand DNA bonds; refusing a connected duplex.")
    charge = sum(float(atom.charge) for atom in structure.atoms)
    rounded_charge = round(charge)
    if not math.isclose(charge, rounded_charge, abs_tol=1e-3):
        raise RuntimeError(f"Amber DNA charge {charge:g} is not an integer.")
    manifest = {"atoms": [{key: atom[key] for key in ("name", "residue", "chain", "resid", "element")} for atom in atoms],
                "bonds": [[left - 1, right - 1] for left, right in sorted(bonds)],
                "charge": rounded_charge, "md_model": args.model}
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _add_dna_posres(Path(args.topol), manifest)


def _parmed_export_mol2(args: argparse.Namespace) -> None:
    pmd, _, _ = _parmed_import()
    structure = pmd.load_file(args.prmtop, args.inpcrd)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    expected = manifest["atoms"]
    if len(structure.atoms) != len(expected):
        raise RuntimeError("dna.prmtop atom count differs from dna_manifest.json.")
    for atom, reference in zip(structure.atoms, expected):
        if atom.name != reference["name"] or _normalise_residue(atom.residue.name) != reference["residue"]:
            raise RuntimeError("dna.prmtop identity/order differs from dna_manifest.json; refusing to rebuild DNA for solvation.")
    structure.save(args.output, overwrite=True)


def _box_volume_nm3(structure: Any) -> float:
    box = getattr(structure, "box", None)
    if box is None or len(box) < 6:
        raise RuntimeError("Amber solvation box is missing periodic dimensions.")
    a, b, c, alpha, beta, gamma = map(float, box[:6])
    ca, cb, cg = math.cos(math.radians(alpha)), math.cos(math.radians(beta)), math.cos(math.radians(gamma))
    volume_a3 = a * b * c * math.sqrt(1 + 2 * ca * cb * cg - ca * ca - cb * cb - cg * cg)
    if not math.isfinite(volume_a3) or volume_a3 <= 0:
        raise RuntimeError("Amber solvation box volume is invalid.")
    return volume_a3 / 1000.0


def _parmed_box_info(args: argparse.Namespace) -> None:
    pmd, _, _ = _parmed_import()
    structure = pmd.load_file(args.prmtop, args.inpcrd)
    Path(args.output).write_text(json.dumps({"volume_nm3": _box_volume_nm3(structure)}, indent=2) + "\n", encoding="utf-8")


def _validate_opc_and_prefix(structure: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = manifest["atoms"]
    if len(structure.atoms) <= len(expected):
        raise RuntimeError("Solvated Amber topology has no solvent/ions.")
    for atom, reference in zip(structure.atoms, expected):
        if atom.name != reference["name"] or _normalise_residue(atom.residue.name) != reference["residue"]:
            raise RuntimeError("Solvation changed the Amber DNA atom order or identity.")
        atom.residue.name, atom.residue.chain, atom.residue.number = reference["residue"], reference["chain"], reference["resid"]
    expected_bonds = {tuple(sorted((left + 1, right + 1))) for left, right in manifest["bonds"]}
    found_bonds = {pair for pair in _bond_pairs(structure) if max(pair) <= len(expected)}
    if found_bonds != expected_bonds:
        raise RuntimeError("Solvation changed DNA bonding from dna_atom_bond_manifest.json.")
    water = [residue for residue in structure.residues
             if _normalise_residue(residue.name) is None and residue.name.strip().upper() not in ION_NAMES]
    if not water or any(len(residue.atoms) != 4 for residue in water):
        raise RuntimeError("OPC check failed: expected four-site water residues after solvation.")
    for residue in water:
        elements = [_atom_element(atom) for atom in residue.atoms]
        if elements.count("O") != 1 or elements.count("H") != 2 or not any(not element for element in elements):
            raise RuntimeError("OPC check failed: each water must contain O, two H, and one extra point.")
    ion_counts: dict[str, int] = {}
    for residue in structure.residues:
        name = residue.name.strip()
        if name in {"Na+", "K+", "Cl-"}:
            ion_counts[name] = ion_counts.get(name, 0) + 1
    total_charge = sum(float(atom.charge) for atom in structure.atoms)
    if not math.isclose(total_charge, 0.0, abs_tol=1e-4):
        raise RuntimeError(f"Final solvated system charge is {total_charge:g} e, not neutral.")
    return {"water_molecules": len(water), "sites_per_water": 4, "final_atom_count": len(structure.atoms),
            "ion_counts": ion_counts, "total_charge_e": total_charge}


def _parmed_export_solvated(args: argparse.Namespace) -> None:
    pmd, gro_writer, top_writer = _parmed_import()
    structure = pmd.load_file(args.prmtop, args.inpcrd)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    opc = _validate_opc_and_prefix(structure, manifest)
    top = top_writer.from_structure(structure)
    top.write(args.topol)
    gro_writer.write(top, args.gro)
    _assert_opc_gromacs_sections(Path(args.topol))
    posre = _add_dna_posres(Path(args.topol), manifest)
    Path(args.output).write_text(json.dumps({"dna_atom_prefix_preserved": True, "posre_files": posre, **opc}, indent=2) + "\n", encoding="utf-8")


def _worker_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("worker", choices={"_parmed_export_dna", "_parmed_export_mol2", "_parmed_box_info", "_parmed_export_solvated"})
    parser.add_argument("--prmtop", required=True)
    parser.add_argument("--inpcrd", required=True)
    parser.add_argument("--topology-pdb")
    parser.add_argument("--topol")
    parser.add_argument("--gro")
    parser.add_argument("--manifest")
    parser.add_argument("--model")
    parser.add_argument("--strands")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.worker == "_parmed_export_dna":
            _parmed_export_dna(args)
        elif args.worker == "_parmed_export_mol2":
            _parmed_export_mol2(args)
        elif args.worker == "_parmed_box_info":
            _parmed_box_info(args)
        else:
            _parmed_export_solvated(args)
    except Exception as exc:
        print(f"Amber/ParmEd worker error: {exc}", file=sys.stderr)
        return 2
    return 0


def _cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("topology", "solvate"))
    parser.add_argument("--job", type=Path, default=Path.cwd())
    parser.add_argument("--tleap", help="Override workflow.json amber_tleap")
    parser.add_argument("--python", dest="amber_python", help="Override workflow.json amber_python")
    args = parser.parse_args(argv)
    job = args.job.resolve()
    try:
        cfg = json.loads((job / "workflow.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        parser.error(f"cannot read {job / 'workflow.json'}: {exc}")
    tleap = args.tleap or cfg.get("amber_tleap") or DEFAULT_TLEAP
    amber_python = args.amber_python or cfg.get("amber_python") or DEFAULT_PYTHON
    try:
        if args.stage == "topology":
            prepare_topology(job, cfg, tleap, amber_python)
        else:
            solvate(job, cfg, tleap, amber_python)
    except AmberBackendError as exc:
        print(f"Amber backend: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0].startswith("_parmed_"):
        return _worker_main(argv)
    return _cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
