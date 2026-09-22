"""Small executable checks. --md-gro adds a real all-atom MD frame conversion test.

Mocked GROMACS calls test orchestration only, NOT physical topology/MD execution.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import dna_workflow as wf
import md_pipeline as md


def fails(fn, exception=ValueError):
    try:
        fn()
    except exception:
        return
    raise AssertionError(f"Expected {exception.__name__}")


def real_frame_test(gro: Path, ff: Path, root: Path):
    # GRO is used only as a local regression fixture, not a new production input interface.
    lines = gro.read_text().splitlines()
    atoms = []
    chain = "A"
    previous_resid = None
    for line in lines[2:2+int(lines[1])]:
        resid = int(line[:5])
        name = line[10:15].strip()
        residue = line[5:10].strip()
        if residue not in wf.BASES:
            continue
        if previous_resid is not None and resid < previous_resid:
            chain = chr(ord(chain)+1)
        previous_resid = resid
        atoms.append(wf.Atom(int(line[15:20]), name, residue, chain, resid,
                            tuple(10*float(line[a:b]) for a,b in [(20,28),(28,36),(36,44)]), name[0]))
    strands = []
    for group in wf.residues(atoms):
        if len(strands) <= ord(group[0].chain)-65:
            strands.append("")
        strands[-1] += group[0].residue[-1]
    report = wf.validate_dna(atoms, strands, ff, hydrogens=True)
    job = root / "real_frame"
    job.mkdir()
    shutil.copytree(ff, job / ff.name)
    wf.save_json(job / "workflow.json", {"strands_5to3": strands, "forcefield": ff.name})
    (job / "topology.pdb").write_text(wf.pdb_text(atoms), encoding="ascii")
    # Known environment atom verifies filtering without moving any DNA coordinate.
    water = wf.Atom(len(atoms)+1, "OW", "SOL", "W", 1, (100.,100.,100.), "O")
    frame = job / "selected.pdb"
    frame.write_text(wf.pdb_text(atoms + [water]), encoding="ascii")
    args = SimpleNamespace(job=job, frame=frame, output=job / "gaussian", nproc=16,
                           memory="32GB", time_ps=100000., dispersion="none")
    wf.gaussian(args)
    result = json.loads((args.output / "gaussian_preparation.json").read_text())
    assert result["removed_environment_atoms"] == 1 and result["charge"] == report["charge"]
    gjf = (args.output / "dna.gjf").read_text()
    assert "Output=Matrix" in gjf and "NoSymm" in gjf and gjf.endswith("dna.mat\n\n")
    assert "EmpiricalDispersion" not in gjf
    assert [a.xyz for a in wf.read_pdb(args.output / "dna_selected.pdb")] == [a.xyz for a in wf.read_pdb(job / "topology.pdb")]
    # Missing a hydrogen must block Gaussian export rather than silently change chemistry.
    frame.write_text(wf.pdb_text([a for a in atoms if a.element != "H"]), encoding="ascii")
    fails(lambda: wf.gaussian(replace_namespace(args, output=job / "should_not_exist")))
    assert not (job / "should_not_exist").exists()
    print(f"PASS real MD frame -> Gaussian: {len(atoms)} DNA atoms, {report['charge']} charge; environment removed; missing-H rejected")


def replace_namespace(ns, **changes):
    return SimpleNamespace(**(vars(ns) | changes))


def orchestration_test(root: Path):
    job = root / "mock_md"
    job.mkdir()
    wf.save_json(job / "workflow.json", {})
    for name in ["topol.top", "npt.gro", "npt.cpt", "npt.complete.json"]:
        (job / name).write_text("fixture")
    (job / "md.mdp").write_text("nsteps = 100\n")
    calls = []
    steps = [40]
    def fake(argv, cwd, log_path, input_text=None, capture=False, echo=True):
        calls.append(argv)
        if "grompp" in argv:
            (job / "md.tpr").write_text("mock")
        if "mdrun" in argv:
            for name in ["md.gro", "md.cpt", "md.log"]:
                (job / name).write_text("mock")
        out = f"step = {steps[0]}\n" if "dump" in argv else "mock output\n"
        return out if capture else ""
    args = SimpleNamespace(job=job, gmx="gmx", stage="md", ntmpi=1, ntomp=2, maxh=None,
                           noninteractive_terminals=False, terminal_timeout=60)
    with patch.object(md, "stream_process", fake):
        fails(lambda: md.run(args), RuntimeError)
        assert not (job / "md.complete.json").exists()
        steps[0] = 100
        md.run(args)
        assert any("-cpi" in c and "-append" in c for c in calls)
        assert any("-ntmpi" in c and c[c.index("-ntmpi") + 1] == "1" for c in calls)
        assert (job / "md.complete.json").exists()
        (job / "md.mdp").write_text("nsteps = 200\n")
        fails(lambda: md.run(args), RuntimeError)
    print("PASS orchestration MOCK ONLY: incomplete checkpoint blocks downstream; resume uses checkpoint; changed MDP rejected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forcefield", type=Path, required=True)
    parser.add_argument("--md-gro", type=Path)
    args = parser.parse_args()
    ff = args.forcefield.resolve()
    with tempfile.TemporaryDirectory(prefix="dna_workflow_test_") as temporary:
        root = Path(temporary)
        for seq, duplex in [("ACGTAC", False), ("ACGTAC", True), ("AT", True), ("TTTT", True)]:
            strands = [seq, wf.complement(seq)] if duplex else [seq]
            atoms = wf.build_atoms(seq, duplex)
            report = wf.validate_dna(atoms, strands, ff)
            assert report["charge"] == -(len(seq)-1)*len(strands)
            file = root / "roundtrip.pdb"
            file.write_text(wf.pdb_text(atoms), encoding="ascii")
            wf.validate_dna(wf.read_pdb(file), strands, ff)
            broken = list(atoms)
            broken[-1] = replace(broken[-1], name="BAD")
            fails(lambda: wf.validate_dna(broken, strands, ff))
        for bad in ["", "A", "ACNT", "ACGU"]:
            fails(lambda: wf.sequence(bad))
        assert wf.complement("ACGTAC") == "GTACGT"
        cfg = {"temperature_K":310, "nvt_ps":100, "npt_ps":1000, "md_ns":100, "xtc_ps":10, "seed":42}
        for stage in ["em", "nvt", "npt", "md"]:
            settings = dict(line.split("=",1) for line in wf.mdp_text(stage,cfg).splitlines() if "=" in line)
            settings = {k.strip():v.strip() for k,v in settings.items()}
            assert settings["vdw-modifier"] == "Force-switch"
            if stage == "md":
                assert int(settings["nsteps"]) == 50000000 and int(settings["nstxout-compressed"]) == 5000
                assert "define" not in settings and settings["gen_vel"] == "no"
        print("PASS ss/ds non-palindromic sequence, CHARMM heavy atoms, termini, O-P geometry, PDB roundtrip and MDP time units")
        orchestration_test(root)
        if args.md_gro:
            real_frame_test(args.md_gro, ff, root)
    print("PASS: all requested checks; no real GROMACS or Gaussian run was performed")


if __name__ == "__main__":
    main()
