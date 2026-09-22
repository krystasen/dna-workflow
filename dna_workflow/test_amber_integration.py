"""Branch/identity regressions; synthetic fixtures are not physical validation."""
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import dna_workflow as wf
import gui_server as gui


class AmberIntegrationTest(unittest.TestCase):
    def test_environment_reports_missing_amber_dependency(self):
        app = gui.App()
        responses = [SimpleNamespace(returncode=0, stdout="Ubuntu-24.04\n".encode("utf-16-le"), stderr=b""),
                     SimpleNamespace(returncode=0, stdout="WSL2"),
                     SimpleNamespace(returncode=0, stdout="GROMACS 2025.4"),
                     SimpleNamespace(returncode=1, stdout="No module named parmed")]
        with patch.object(gui.subprocess, "run", side_effect=responses):
            app.environment_check({"md_model": "ol21_opc"})
        self.assertEqual(app.environment["status"], "unavailable")
        self.assertIn("parmed", app.environment["error"])

    def test_mdp_profiles_are_separate(self):
        cfg = dict(temperature_K=310, nvt_ps=100, npt_ps=1000, md_ns=100, xtc_ps=10, seed=42)
        self.assertIn("Force-switch", wf.mdp_text("md", cfg))
        for model in ("ol21_opc", "ol24_opc"):
            text = wf.mdp_text("md", cfg | {"md_model": model})
            self.assertNotIn("Force-switch", text)
            self.assertNotIn("rvdw-switch", text)
            self.assertIn("EnerPres", text)
            self.assertNotIn("-DPOSRES", text)
            self.assertIn("-DPOSRES", wf.mdp_text("npt", cfg | {"md_model": model}))

    def test_amber_identity_and_bonds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Deliberately minimal synthetic identity/bond fixture.
            atoms = [wf.Atom(1, "O3'", "DA", "A", 1, (0., 0., 0.), "O"),
                     wf.Atom(2, "P", "DC", "A", 2, (1.6, 0., 0.), "P")]
            cfg = {"md_model": "ol21_opc", "strands_5to3": ["AC"]}
            manifest = {"md_model": "ol21_opc", "charge": -1, "atoms": [asdict(a) for a in atoms], "bonds": [[0, 1]]}
            wf.save_json(root / "dna_manifest.json", manifest)
            self.assertEqual(wf.validate_job_dna(atoms, root, cfg)["charge"], -1)
            with self.assertRaisesRegex(ValueError, "identity"):
                wf.validate_job_dna([atoms[0], replace(atoms[1], name="BAD")], root, cfg)
            with self.assertRaisesRegex(ValueError, "bond"):
                wf.validate_job_dna([atoms[0], replace(atoms[1], xyz=(9., 0., 0.))], root, cfg)
            with self.assertRaisesRegex(ValueError, "model"):
                wf.validate_job_dna(atoms, root, cfg | {"md_model": "ol24_opc"})

    def test_gui_explicit_box_and_amber_arguments(self):
        app = gui.App()
        with tempfile.TemporaryDirectory() as tmp:
            payload = dict(sequence="ACGT", mode="duplex", output=str(Path(tmp)/"job"), md_model="ol21_opc",
                           temperature=310, salt=.15, nvt_ps=100, npt_ps=1000, md_ns=100, xtc_ps=10,
                           padding=1.5, seed=42, box_type="dodecahedron")
            with self.assertRaisesRegex(gui.APIError, "AmberTools"):
                app.create_job(payload)
            calls = []
            with patch.object(app, "_start", side_effect=lambda name, work: work("test")), \
                 patch.object(app, "_run", side_effect=lambda _, argv, *a: calls.append(argv)), \
                 patch.object(app, "open_job"):
                app.create_job(payload | {"box_type": "octahedron"})
            self.assertEqual(len(calls), 1)
            self.assertIn("--md-model", calls[0])
            self.assertIn("--amber-tleap", calls[0])
            self.assertNotIn("--forcefield", calls[0])
            self.assertNotIn("mdrun", calls[0])

    def test_gui_rejects_missing_charmm_forcefield_before_background_build(self):
        app = gui.App()
        with tempfile.TemporaryDirectory() as tmp:
            payload = dict(sequence="ACGT", mode="duplex", output=str(Path(tmp)/"job"),
                           md_model="charmm36_tip3p", forcefield=str(Path(tmp)/"missing.ff"),
                           temperature=310, salt=.15, nvt_ps=100, npt_ps=1000, md_ns=100, xtc_ps=10,
                           padding=1.5, seed=42, box_type="dodecahedron")
            with self.assertRaisesRegex(gui.APIError, "缺少 na.rtp"):
                app.create_job(payload)


if __name__ == "__main__":
    unittest.main()
