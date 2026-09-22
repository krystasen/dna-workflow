"""Fast checks for md_extensions; no GROMACS dynamics and no user data are run."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import md_extensions as ext
from dna_workflow import Atom, gaussian, save_json, select_dna_atoms


def gro_line(resid, residue, name, serial, xyz=(0.0, 0.0, 0.0)):
    return f"{resid:5d}{residue:<5}{name:>5}{serial:5d}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}\n"


def fails(fn, error):
    try:
        fn()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def main():
    with tempfile.TemporaryDirectory(prefix="md_extensions_test_") as temporary:
        root = Path(temporary)
        gro = root / "box.gro"
        gro.write_text("test\n6\n" + "".join(gro_line(1, "SOL", name, i) for i, name in enumerate(("OW", "HW1", "HW2", "OW", "HW1", "HW2"), 1))
                       + "2 3 4 0.1 0.2 0.3 0.4 0.5 0.6\n", encoding="ascii")
        total, waters, water_atoms, vectors, volume = ext._gro_summary(gro)
        assert (total, waters, water_atoms) == (6, 1, 6)
        assert vectors == [[2.0, 0.1, 0.2], [0.3, 3.0, 0.4], [0.5, 0.6, 4.0]]
        assert abs(volume - 23.156) < 1e-9

        topol = root / "topol.top"
        (root / "topol_DNA_chain_A.itp").write_text("[ moleculetype ]\nDNA_real 3\n", encoding="ascii")
        topol.write_text('#include "ff.itp"\n#include "topol_DNA_chain_A.itp"\n#ifdef POSRES_WATER\n#include "tip3p.itp"\n#endif\n[ molecules ]\nDNA_real 1\nSOL 5\nSOD 2\n', encoding="utf-8")
        rebuilt = ext._dna_topology_text(topol)
        assert "DNA_real 1" in rebuilt and "SOL 5" not in rebuilt and "SOD 2" not in rebuilt and "#ifdef POSRES_WATER" in rebuilt

        dna = [Atom(1, "P", "DA", "A", 1, (0, 0, 0), "P"), Atom(2, "H5T", "DA", "A", 1, (1, 0, 0), "H")]
        raw = [dna[0], Atom(3, "OW", "SOL", "", 1, (9, 9, 9), "O"), dna[1]]
        assert select_dna_atoms(raw, dna) == dna

        child = root / "child"
        child.mkdir()
        save_json(child / "workflow.json", {"role": "representative_relaxation"})
        fails(lambda: ext.verify_relax(SimpleNamespace(job=child, gmx="gmx")), FileNotFoundError)
        fails(lambda: gaussian(SimpleNamespace(job=child, frame=child / "relaxed_dna.pdb", output=root / "gaussian",
                                               nproc=1, memory="1GB", time_ps=None, dispersion="none")), RuntimeError)

        # The real 27-bp DNA exceeds GROMACS' 4096-character index-line buffer.
        many = [Atom(i, "P", "DA", "A", 1, (0, 0, 0), "P") for i in range(1, 1714)]
        raw_gro = "test\n1713\n" + "".join(gro_line(1, "DA", "P", i) for i in range(1, 1714)) + "2 2 2\n"
        (child / "em.gro").write_text(raw_gro, encoding="ascii")
        def whole_command(parts, label, input_text):
            assert parts[0] == "trjconv" and parts[-2:] == ["-pbc", "whole"]
            ndx = Path(parts[parts.index("-n") + 1]).read_text()
            assert max(map(len, ndx.splitlines())) < 100
            assert list(map(int, ndx.splitlines()[1:][0].split())) == list(range(1, 16))
            assert list(map(int, " ".join(ndx.splitlines()[1:]).split())) == list(range(1, 1714))
            Path(parts[parts.index("-o") + 1]).write_text(raw_gro, encoding="ascii")
        assert len(ext._whole_em_dna(child, many, 1, whole_command)) == 1713
        assert (child / "em.gro").read_text() == raw_gro

        parent = root / "parent"; parent.mkdir()
        frame = root / "representative.pdb"; frame.write_text("END\n", encoding="ascii")
        (parent / "em.mdp").write_text("integrator = steep\nnsteps = 1\n", encoding="ascii")
        save_json(parent / "workflow.json", {"strands_5to3": ["A"], "forcefield": "ff", "seed": 7})
        calls = []
        def command_factory(job, gmx):
            def command(parts, *unused, **kwargs):
                calls.append(parts)
                return ""
            return command
        def fake_run(args):
            assert args.stage == "prepare_em"
        with patch.object(ext, "read_pdb", side_effect=[dna, raw]), patch.object(ext, "restore_identity", side_effect=lambda atoms, strands: atoms), \
             patch.object(ext, "validate_dna", return_value={"atom_count": 2}), patch.object(ext, "select_dna_atoms", return_value=dna), \
             patch.object(ext, "_copy_relax_inputs"), patch.object(ext, "_command_for_job", side_effect=command_factory), patch.object(ext.md, "run", side_effect=fake_run):
            ext.prepare_relax(SimpleNamespace(job=parent, frame=frame, output=root / "prepared", gmx="gmx", padding=1.5, salt=0.15, box_type="dodecahedron"))
        assert [parts[0] for parts in calls] == ["editconf", "solvate", "grompp", "genion"]
        assert all(parts[0] != "mdrun" for parts in calls)
    print("PASS md_extensions: frame DNA selection, environment-free topology, triclinic volume, EM-return gate, prepare-only no mdrun")


if __name__ == "__main__":
    main()
