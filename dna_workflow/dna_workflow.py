"""Sequence -> MD job; selected all-atom frame -> Gaussian or ORCA job.

Linear DNA with canonical or catalogued OL21 modifications and OH termini is supported.
Open Babel is required only by `build`; other commands use the standard library.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from importlib.metadata import version
import json
import math
from pathlib import Path
import re
import shutil


HERE = Path(__file__).resolve().parent
BASES = {"DA", "DC", "DG", "DT"}
from modified_bases import (RESIDUE_PARENTS, apply_modifications, expected_residues, expected_dna_charge,
                            library_root, snapshot_library, validate_modifications)
ENVIRONMENT = {"SOL", "HOH", "WAT", "OPC", "TIP3", "SOD", "CLA", "POT", "NA", "CL", "K", "Na+", "K+", "Cl-"}
MD_MODELS = ("charmm36_tip3p", "ol21_opc", "ol24_opc")
AMBER_ENV = "/home/chenz/.local/share/dna-amber/env"
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "Cl": 17}


@dataclass
class Atom:
    serial: int
    name: str
    residue: str
    chain: str
    resid: int
    xyz: tuple[float, float, float]
    element: str


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sequence(text: str) -> str:
    value = "".join(text.upper().split())
    if len(value) < 2 or set(value) - set("ACGT"):
        raise ValueError("Use at least two canonical A/C/G/T nucleotides, written 5' -> 3'.")
    return value


def complement(text: str) -> str:
    return text.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def read_pdb(path: Path) -> list[Atom]:
    atoms = []
    models = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MODEL"):
            models += 1
            if models > 1:
                raise ValueError("Export exactly one real trajectory frame, not a multi-model PDB.")
        if line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        if line[16:17].strip() or line[26:27].strip():
            raise ValueError("Resolve alternate locations/insertion codes before conversion.")
        name = line[12:16].strip()
        element = line[76:78].strip().upper() or name.lstrip("123")[0].upper()
        element = 'Cl' if element == 'CL' else element
        xyz = tuple(float(line[a:b]) for a, b in [(30, 38), (38, 46), (46, 54)])
        if not all(math.isfinite(x) for x in xyz):
            raise ValueError("Non-finite PDB coordinates.")
        atoms.append(Atom(int(line[6:11]), name, line[17:20].strip(), line[21:22],
                          int(line[22:26]), xyz, element))
    if not atoms:
        raise ValueError(f"No atoms in {path}")
    return atoms


def pdb_text(atoms: list[Atom]) -> str:
    lines = []
    last_chain = None
    if len(atoms) > 99999:
        raise ValueError("This PDB exporter supports at most 99999 atoms.")
    for i, atom in enumerate(atoms, 1):
        if last_chain is not None and last_chain != atom.chain:
            lines.append("TER")
        x, y, z = atom.xyz
        if any(len(f"{v:8.3f}") > 8 for v in atom.xyz):
            raise ValueError("Coordinates exceed PDB field width.")
        name = f" {atom.name:<3}" if len(atom.name) < 4 else atom.name
        lines.append(f"ATOM  {i:5d} {name:4s} {atom.residue:>3s} {atom.chain}{atom.resid:4d}    "
                     f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom.element:>2s}  ")
        last_chain = atom.chain
    return "\n".join(lines + ["TER", "END", ""])


def residues(atoms: list[Atom]) -> list[list[Atom]]:
    groups = []
    seen = set()
    for atom in atoms:
        key = (atom.chain, atom.resid)
        if not groups or key != (groups[-1][0].chain, groups[-1][0].resid):
            if key in seen:
                raise ValueError(f"Non-contiguous or duplicate residue {key}")
            seen.add(key)
            groups.append([])
        groups[-1].append(atom)
    return groups


def rtp_data(ff: Path) -> dict:
    """Read actual CHARMM residue atom names and covalent bonds, not a second database."""
    data = {}
    residue = section = None
    for raw in (ff / "na.rtp").read_text(encoding="utf-8").splitlines():
        line = raw.split(";", 1)[0].strip()
        match = re.fullmatch(r"\[\s*(.*?)\s*\]", line)
        if match:
            label = match[1]
            if label in BASES:
                residue = label
                data[label] = {"atoms": set(), "bonds": []}
                section = None
            elif label in {"atoms", "bonds", "angles", "dihedrals", "impropers", "cmap", "exclusions"}:
                section = label
            else:
                residue = None
            continue
        if residue and line:
            fields = line.split()
            if section == "atoms":
                data[residue]["atoms"].add(fields[0])
            elif section == "bonds":
                data[residue]["bonds"].append(tuple(fields[:2]))
    if set(data) != BASES:
        raise ValueError("Force field must contain DA/DC/DG/DT in na.rtp.")
    for filename, patch in [("na.n.tdb", "5TER"), ("na.c.tdb", "3TER")]:
        if not re.search(r"\[\s*" + patch + r"\s*\]", (ff / filename).read_text()):
            raise ValueError(f"Missing CHARMM terminal patch {patch}")
    return data


def validate_dna(atoms: list[Atom], strands: list[str], ff: Path, hydrogens=False) -> dict:
    data = rtp_data(ff)
    groups = residues(atoms)
    expected = [(chr(65 + c), i + 1, "D" + base, i == 0, i == len(s) - 1)
                for c, s in enumerate(strands) for i, base in enumerate(s)]
    if len(groups) != len(expected):
        raise ValueError("Residue count differs from sequence manifest.")
    bond_lengths = []
    for group, (chain, resid, name, first, last) in zip(groups, expected):
        atom0 = group[0]
        if (atom0.chain, atom0.resid, atom0.residue) != (chain, resid, name):
            raise ValueError(f"Sequence/order mismatch at {chain}:{resid}:{name}")
        names = {a.name for a in group}
        wanted = set(data[name]["atoms"])
        if first:
            wanted -= {"P", "O1P", "O2P"}
            wanted.add("H5T")
        if last:
            wanted.add("H3T")
        if not hydrogens:
            wanted = {n for n in wanted if not n.startswith("H")}
        if names != wanted or len(names) != len(group):
            raise ValueError(f"{chain}:{resid} atom mismatch: missing={wanted-names}, extra={names-wanted}")
        if any(a.element != a.name[0] for a in group):
            raise ValueError(f"Unexpected element at {chain}:{resid}")
        byname = {a.name: a for a in group}
        bonds = list(data[name]["bonds"])
        if first:
            bonds.append(("O5'", "H5T"))
        if last:
            bonds.append(("O3'", "H3T"))
        for a, b in bonds:
            if a not in byname or b not in byname:
                continue
            d = math.dist(byname[a].xyz, byname[b].xyz)
            low, high = (0.65, 1.40) if "H" in (byname[a].element, byname[b].element) else (1.05, 1.95)
            if not low <= d <= high:
                raise ValueError(f"Bad covalent distance {chain}:{resid} {a}-{b}: {d:.3f} A; check PBC/structure.")
    for previous, current in zip(groups, groups[1:]):
        if previous[0].chain == current[0].chain:
            p = next(a for a in current if a.name == "P")
            o = next(a for a in previous if a.name == "O3'")
            distance = math.dist(o.xyz, p.xyz)
            if not 1.30 < distance < 1.95:
                raise ValueError(f"Broken O3'-P connection at {current[0].chain}:{current[0].resid}: {distance:.3f} A")
            bond_lengths.append(distance)
    return {"atom_count": len(atoms), "nucleotides": len(groups), "strands": strands,
            "charge": -sum(len(s) - 1 for s in strands), "cross_residue_bonds": len(bond_lengths),
            "OP_distance_A": [min(bond_lengths), max(bond_lengths)] if bond_lengths else [],
            "validation": "CHARMM residue atom sets and bond geometry checked; not a pdb2gmx execution"}


def validate_job_dna(atoms: list[Atom], job: Path, cfg: dict) -> dict:
    """Validate CHARMM via RTP, AMBER via the actual exported covalent topology."""
    if cfg.get("md_model", MD_MODELS[0]) == MD_MODELS[0]:
        return validate_dna(atoms, cfg["strands_5to3"], job / cfg["forcefield"], hydrogens=True)
    manifest = json.loads((job / "dna_manifest.json").read_text(encoding="utf-8"))
    if manifest["md_model"] != cfg["md_model"]:
        raise ValueError("DNA topology model differs from job configuration")
    fields = ("name", "residue", "chain", "resid", "element")
    if [tuple(getattr(a, k) for k in fields) for a in atoms] != [tuple(a[k] for k in fields) for a in manifest["atoms"]]:
        raise ValueError("DNA atom identity/order differs from Amber topology manifest")
    expected = expected_residues(cfg["strands_5to3"], cfg.get("modifications"))
    groups = residues(atoms)
    if [(g[0].chain, g[0].resid, g[0].residue) for g in groups] != expected:
        raise ValueError("Amber residue sequence does not match the input strands")
    charge = expected_dna_charge(cfg["strands_5to3"], cfg.get("modifications"), library_root(job))
    if not math.isclose(manifest["charge"], charge, abs_tol=1e-5):
        raise ValueError("Amber DNA charge differs from the configured DNA/modifications")
    cross_bonds = []
    for i, j in manifest["bonds"]:
        a, b = atoms[i], atoms[j]
        if a.chain != b.chain:
            raise ValueError("Unexpected covalent bond between DNA strands")
        distance = math.dist(a.xyz, b.xyz)
        low, high = (0.65, 1.40) if "H" in (a.element, b.element) else (1.05, 1.95)
        if not low <= distance <= high:
            raise ValueError(f"Bad Amber DNA bond {i+1}-{j+1}: {distance:.3f} A; check PBC")
        if a.resid != b.resid:
            cross_bonds.append(distance)
    if len(cross_bonds) != sum(len(s)-1 for s in cfg["strands_5to3"]):
        raise ValueError("Amber backbone connectivity differs from input strands")
    return {"atom_count": len(atoms), "nucleotides": len(groups), "strands": cfg["strands_5to3"],
            "charge": charge, "cross_residue_bonds": len(cross_bonds),
            "validation": "Amber atom identities, sequence, charge and exported covalent bond geometry checked"}


def build_atoms(seq: str, duplex: bool) -> list[Atom]:
    from openbabel import openbabel as ob
    conversion = ob.OBConversion()
    conversion.SetInFormat("fasta")
    if not duplex:
        conversion.AddOption("1", ob.OBConversion.INOPTIONS)
    molecule = ob.OBMol()
    if not conversion.ReadString(molecule, ">DNA\n" + seq + "\n"):
        raise ValueError("Open Babel FASTA construction failed.")
    # Preserve FASTA residue names: OBAtom.GetResidue otherwise triggers re-perception in this wheel.
    molecule.SetChainsPerceived()
    atoms = []
    for res in ob.OBResidueIter(molecule):
        for atom in ob.OBResidueAtomIter(res):
            name = res.GetAtomID(atom).strip()
            if atom.GetAtomicNum() == 1 or (res.GetNum() == 1 and name in {"OXT", "P", "O1P", "O2P"}):
                continue
            atoms.append(Atom(len(atoms) + 1, name, res.GetName(), res.GetChain(), res.GetNum(),
                              (atom.GetX(), atom.GetY(), atom.GetZ()), ob.GetSymbol(atom.GetAtomicNum())))
    return atoms


def mdp_text(stage: str, cfg: dict) -> str:
    common = {"cutoff-scheme": "Verlet", "nstlist": 20, "coulombtype": "PME", "rcoulomb": 1.2,
              "vdwtype": "Cut-off", "vdw-modifier": "Force-switch", "rvdw-switch": 1.0,
              "rvdw": 1.2, "DispCorr": "no", "pbc": "xyz", "pme_order": 4, "fourierspacing": 0.12}
    if cfg.get("md_model", MD_MODELS[0]) != MD_MODELS[0]:
        # Separate AMBER cutoff convention; do not inherit CHARMM force switching.
        common.update({"vdw-modifier": "Potential-shift", "DispCorr": "EnerPres"})
        common.pop("rvdw-switch")
    if stage == "em":
        settings = {"integrator": "steep", "nsteps": 50000, "emtol": 1000, "emstep": 0.01}
    else:
        duration = cfg["md_ns"] * 1000 if stage == "md" else cfg[stage + "_ps"]
        settings = {"integrator": "md", "dt": 0.002, "nsteps": round(duration / 0.002),
                    "constraints": "h-bonds", "constraint_algorithm": "lincs", "lincs_order": 4,
                    "lincs_iter": 1, "continuation": "no" if stage == "nvt" else "yes",
                    "tcoupl": "V-rescale", "tc-grps": "System", "tau_t": 1.0, "ref_t": cfg["temperature_K"],
                    "gen_vel": "yes" if stage == "nvt" else "no", "nstxout": 0, "nstvout": 0,
                    "nstfout": 0, "nstenergy": 500, "nstlog": 500,
                    "nstxout-compressed": round(cfg["xtc_ps"] / 0.002), "compressed-x-grps": "System",
                    "pcoupl": "no" if stage == "nvt" else "C-rescale"}
        if stage != "md":
            settings["define"] = "-DPOSRES"
        if stage == "nvt":
            settings.update(gen_temp=cfg["temperature_K"], gen_seed=cfg["seed"])
        else:
            settings.update(pcoupltype="isotropic", tau_p=5.0, ref_p=1.0, compressibility="4.5e-5")
        if stage == "npt":
            settings["refcoord_scaling"] = "com"
    settings.update(common)
    return "; Generated DNA workflow; duration is a target, not proof of equilibration.\n" + "".join(
        f"{key:<24} = {value}\n" for key, value in settings.items())


def build(args) -> None:
    seq = sequence(args.sequence)
    strands = [seq]
    if args.duplex or args.partner:
        partner = sequence(args.partner) if args.partner else complement(seq)
        if partner != complement(seq):
            raise ValueError("Initial version supports matched antiparallel duplexes only; both inputs are 5' -> 3'.")
        strands.append(partner)
    model = getattr(args, "md_model", MD_MODELS[0])
    if model not in MD_MODELS:
        raise ValueError("Unsupported DNA/water model")
    modifications = validate_modifications(json.loads(getattr(args, "modifications", "[]")), strands, model)
    amber = model != MD_MODELS[0]
    ff = args.forcefield.resolve() if args.forcefield else None
    if not amber and (ff is None or not ff.name.endswith(".ff")):
        raise ValueError("Pass the CHARMM GROMACS force-field directory ending in .ff.")
    if amber and args.box_type == "dodecahedron":
        raise ValueError("AmberTools branch supports cubic/triclinic/octahedron, not dodecahedron. Select the box explicitly.")
    atoms = build_atoms(seq, len(strands) == 2)
    if modifications:
        atoms = apply_modifications(atoms, strands, modifications)
    report = ({"charge": expected_dna_charge(strands, modifications), "atom_count": len(atoms),
               "validation": "Ideal heavy-atom seed only; AmberTools topology validation pending"}
              if amber else validate_dna(atoms, strands, ff))
    for value in [args.temperature, args.nvt_ps, args.npt_ps, args.md_ns, args.xtc_ps, args.padding]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Temperature, durations, output interval and padding must be positive.")
    if args.padding < 1.2 or not math.isfinite(args.salt) or args.salt < 0:
        raise ValueError("Use at least 1.2 nm padding and non-negative salt concentration.")
    for duration in [args.nvt_ps, args.npt_ps, args.md_ns * 1000, args.xtc_ps]:
        if not math.isclose(duration / 0.002, round(duration / 0.002), abs_tol=1e-6):
            raise ValueError("Time parameters must be multiples of 0.002 ps.")
    cfg = {"strands_5to3": strands, "termini": "5OH-3OH", "forcefield": model if amber else ff.name,
           "md_model": model, "water_model": "OPC" if amber else "CHARMM TIP3P",
           "temperature_K": args.temperature, "nvt_ps": args.nvt_ps, "npt_ps": args.npt_ps,
           "md_ns": args.md_ns, "xtc_ps": args.xtc_ps, "seed": args.seed, "salt_M": args.salt,
           "padding_nm": args.padding, "box_type": args.box_type, "formal_dna_charge": report["charge"],
           "openbabel_wheel_version": version("openbabel-wheel"),
           "topology_reference": "topology.pdb", "model": "Open Babel ideal helical start; not an equilibrated conformation"}
    if modifications:
        cfg.update(modifications=modifications, modification_parameters="RESP + initial AMBER/GAFF2 analogy; not long-time validated")
    if amber:
        cfg.update(amber_tleap=args.amber_tleap, amber_python=args.amber_python, salt_cation=args.salt_cation,
                   parameter_validation="pending; topology conversion is not force-field validation")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot_library(output, modifications)
    (output / "dna_input.pdb").write_text(pdb_text(atoms), encoding="ascii")
    (output / "sequence.fasta").write_text("".join(f">strand_{chr(65+i)} 5to3\n{s}\n" for i,s in enumerate(strands)), encoding="ascii")
    save_json(output / "workflow.json", cfg)
    save_json(output / "structure_check.json", report)
    if not amber:
        shutil.copytree(ff, output / ff.name)
    for name in ["dna_workflow.py", "md_pipeline.py", "amber_backend.py", "modified_bases.py", "vmd_handoff.tcl"]:
        shutil.copyfile(HERE / name, output / name)
    for stage in ["em", "nvt", "npt", "md"]:
        (output / (stage + ".mdp")).write_text(mdp_text(stage, cfg), encoding="ascii")
    print(output)
    print(f"{len(strands)} strand(s); {len(atoms)} heavy atoms; DNA charge {report['charge']}; {report['validation']}")


def restore_identity(atoms: list[Atom], strands: list[str], modifications: list[dict] | None = None) -> list[Atom]:
    """pdb2gmx may globally renumber residues; preserve order, restore manifest chain IDs."""
    groups = residues(atoms)
    expected = expected_residues(strands, modifications)
    if len(groups) != len(expected):
        raise ValueError("Topology residue count differs from input strands.")
    result = []
    for group, (chain, resid, name) in zip(groups, expected):
        if any(a.residue != name for a in group):
            raise ValueError("Topology sequence differs from input sequence.")
        result.extend(replace(a, chain=chain, resid=resid) for a in group)
    return result


def select_dna_atoms(raw: list[Atom], reference: list[Atom]) -> list[Atom]:
    """Select canonical DNA from a frame without guessing changed chemistry or atom order."""
    allowed = BASES | {a.residue for a in reference if a.residue in RESIDUE_PARENTS}
    unknown = {a.residue for a in raw} - allowed - ENVIRONMENT
    if unknown:
        raise ValueError(f"Unsupported residues {unknown}; do not silently discard modified DNA or ligands.")
    selected = [a for a in raw if a.residue in allowed]
    if [(a.name, a.residue) for a in selected] != [(a.name, a.residue) for a in reference]:
        raise ValueError("Frame DNA atom order/completeness differs from topology.pdb. Export all DNA atoms including H in original order.")
    return selected


def quantum_input(args, engine="gaussian") -> None:
    job = args.job.resolve()
    cfg = json.loads((job / "workflow.json").read_text(encoding="utf-8"))
    is_relaxation = cfg.get("role") == "representative_relaxation"
    if is_relaxation:
        expected = (job / "relaxed_dna.pdb").resolve()
        if args.frame.resolve() != expected:
            raise ValueError("Representative-relaxation Gaussian input must be this child job's relaxed_dna.pdb.")
        from md_pipeline import verified_return, file_state
        if not verified_return(job, "em"):
            raise RuntimeError("Representative-relaxation Gaussian export requires verified EM return; run verify_relax first.")
        try:
            exported = json.loads((job / "relaxation_report.json").read_text(encoding="utf-8"))
            current = file_state(job, ["em.tpr", "em.gro", "em.log", "em.mdp", "relaxed_dna.pdb", "topology.pdb"])
        except (OSError, ValueError) as exc:
            raise RuntimeError("Missing validated EM-to-PDB export; run verify_relax first.") from exc
        if exported.get("export_files") != current:
            raise RuntimeError("EM results or relaxed PDB changed since export; rerun verify_relax before Gaussian.")
    reference = restore_identity(read_pdb(job / "topology.pdb"), cfg["strands_5to3"], cfg.get("modifications"))
    validate_job_dna(reference, job, cfg)
    raw = read_pdb(args.frame)
    selected = select_dna_atoms(raw, reference)
    cleaned = [replace(a, serial=i+1, xyz=b.xyz) for i,(a,b) in enumerate(zip(reference, selected))]
    report = validate_job_dna(cleaned, job, cfg)
    charge = report["charge"]
    electrons = sum(Z[a.element] for a in cleaned) - charge
    if electrons % 2:
        raise ValueError("Canonical closed-shell DNA has an odd electron count; check hydrogen/terminal completeness.")
    if args.nproc < 1 or not re.fullmatch(r"[1-9][0-9]*(?:MB|GB)", args.memory.upper()):
        raise ValueError("Provide positive nproc and memory such as 32GB (below allocated memory).")
    if args.time_ps is not None and (not math.isfinite(args.time_ps) or args.time_ps < 0):
        raise ValueError("time-ps must be non-negative.")
    route = "#P B3LYP/6-31G(d,p) 6D 10F SCRF=(IEFPCM,Solvent=Water) NoSymm Output=Matrix"
    if args.dispersion != "none":
        route += " EmpiricalDispersion=" + args.dispersion
    if engine == "orca":
        route = "! SP B3LYP 6-31G(d,p) TightSCF RIJCOSX def2/J"
        route += {"none": "", "GD3": " D3ZERO", "GD3BJ": " D3BJ"}[args.dispersion]
        if args.solvent == "cpcm":
            route += " CPCM(Water)"
        total_mb = int(re.match(r"\d+", args.memory).group()) * (1024 if args.memory.upper().endswith("GB") else 1)
        maxcore = int(0.75 * total_mb / args.nproc)
        if maxcore < 1:
            raise ValueError("Total memory is too small for the requested ORCA process count.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "dna_selected.pdb").write_text(pdb_text(cleaned), encoding="ascii")
    coordinates = "\n".join(f"{a.element:<2} {a.xyz[0]:16.8f} {a.xyz[1]:16.8f} {a.xyz[2]:16.8f}" for a in cleaned)
    title = ("EM-relaxed representative structure; unoptimized Gaussian single point; source was a user-selected confirmed PDB"
             if is_relaxation else "Selected raw MD frame; unoptimized Gaussian single point")
    if engine == "gaussian":
        text = (f"%chk=dna.chk\n%nprocshared={args.nproc}\n%mem={args.memory.upper()}\n{route}\n\n"
                f"{title}; time_ps={args.time_ps}\n\n{charge} 1\n"
                f"{coordinates}\n\ndna.mat\n\n")
        (output / "dna.gjf").write_text(text, encoding="ascii")
    else:
        solvent = '%cpcm\n  smd true\n  SMDsolvent "water"\nend\n' if args.solvent == "smd" else ""
        text = (f"{route}\n%pal nprocs {args.nproc} end\n%maxcore {maxcore}\n"
                f"%scf MaxIter 500 end\n%method COSX_IntSymmetry Full end\n{solvent}"
                f"* xyz {charge} 1\n{coordinates}\n*\n")
        (output / "dna.inp").write_text(text, encoding="ascii")
        save_json(output / "orca.json.conf", {"MOCoefficients": True, "Basisset": True,
                                               "1elIntegrals": ["S"], "JSONFormats": ["json"]})
    with (output / "atom_mapping.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([f"{engine}_atom_1based", "source_pdb_serial", "chain", "nucleotide_1based", "resname", "atom_name", "bp_1based"])
        for i,(a,b) in enumerate(zip(cleaned, selected),1):
            bp = a.resid if a.chain == "A" else len(cfg["strands_5to3"][0]) + 1 - a.resid
            writer.writerow([i,b.serial,a.chain,a.resid,a.residue,a.name,bp if len(cfg["strands_5to3"]) == 2 else ""])
    provenance = ({"route": f"raw_frame_to_relax_child_to_verified_em_to_{engine}",
                   "source_job": cfg["source_job"], "source_frame": cfg["source_frame"],
                   "relaxed_frame": str(args.frame.resolve()), "em_verified": True}
                  if is_relaxation else {"route": "direct_raw_md_frame", "source_frame": str(args.frame.resolve())})
    save_json(output / f"{engine}_preparation.json", {**report, "source_frame": str(args.frame.resolve()),
              "time_ps": args.time_ps, "removed_environment_atoms": len(raw)-len(selected),
              "electron_count": electrons, "multiplicity": 1, "route": route,
              "coordinate_units": "angstrom", "coordinates_changed": False,
              "input_role": "representative_relaxation" if is_relaxation else "raw_md_frame", "provenance": provenance,
              "ao_mapping": f"Not available until {'ORCA' if engine == 'orca' else 'Gaussian'} exports basis metadata; CSV is atom mapping only.",
              "validation": f"Geometry, DNA composition, atom order and charge checked; {'ORCA' if engine == 'orca' else 'Gaussian'} not executed"})
    if engine == "orca":
        (output / "README.md").write_text(
            "# ORCA 6.0.1 DNA 单点\n\n"
            "运行 `orca dna.inp > dna.out`。正常结束后，在同目录保留 orca.json.conf，运行 "
            "`orca_2json dna.gbw -json`。\n\n"
            "回传 dna.json、dna.out，以及本目录原有的 dna.inp、dna_selected.pdb 和 atom_mapping.csv。"
            "工作台中选择 ORCA 后检查矩阵并生成 GPU 包。\n\n"
            f"memory={args.memory.upper()} 为总内存预算，%maxcore={maxcore} MB/进程，已预留约25%开销。"
            "ORCA 默认球谐5D，Gaussian 现有路径为笛卡尔6D；二者态数不必相同。\n",
            encoding="utf-8")
    print(output / ("dna.gjf" if engine == "gaussian" else "dna.inp"))


def gaussian(args) -> None:
    quantum_input(args)


def orca(args) -> None:
    quantum_input(args, "orca")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--sequence", required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--single", action="store_true")
    mode.add_argument("--duplex", action="store_true")
    mode.add_argument("--partner", help="Second strand written 5' -> 3'; must be reverse complement")
    p.add_argument("--forcefield", type=Path)
    p.add_argument("--md-model", choices=MD_MODELS, default=MD_MODELS[0])
    p.add_argument("--modifications", default="[]", help="JSON list of chain/position/base modification sites")
    p.add_argument("--amber-tleap", default=AMBER_ENV + "/bin/tleap")
    p.add_argument("--amber-python", default=AMBER_ENV + "/bin/python")
    p.add_argument("--salt-cation", choices=["Na+", "K+"], default="Na+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--temperature", type=float, default=310.0)
    p.add_argument("--salt", type=float, default=0.15)
    p.add_argument("--padding", type=float, default=1.5)
    p.add_argument("--box-type", choices=["cubic", "dodecahedron", "octahedron", "triclinic"], default="dodecahedron")
    p.add_argument("--nvt-ps", type=float, default=100.0)
    p.add_argument("--npt-ps", type=float, default=1000.0)
    p.add_argument("--md-ns", type=float, default=100.0)
    p.add_argument("--xtc-ps", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20260905)
    p.set_defaults(func=build)
    p = sub.add_parser("gaussian")
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--frame", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--time-ps", type=float)
    p.add_argument("--nproc", type=int, default=16)
    p.add_argument("--memory", default="32GB")
    p.add_argument("--dispersion", choices=["none", "GD3", "GD3BJ"], default="none")
    p.set_defaults(func=gaussian)
    p = sub.add_parser("orca")
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--frame", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--time-ps", type=float)
    p.add_argument("--nproc", type=int, default=16)
    p.add_argument("--memory", default="32GB", help="total memory budget; maxcore reserves 25%% overhead")
    p.add_argument("--dispersion", choices=["none", "GD3", "GD3BJ"], default="none")
    p.add_argument("--solvent", choices=["cpcm", "smd"], default="cpcm")
    p.set_defaults(func=orca)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
