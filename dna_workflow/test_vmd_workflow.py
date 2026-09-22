from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest
import tkinter
from unittest.mock import patch

import vmd_workflow


class VmdWorkflowTests(unittest.TestCase):
    def test_independent_gromos_separates_groups_and_accounts_for_remainder(self):
        tcl = tkinter.Tcl()
        tcl.call("source", str(vmd_workflow.HERE / "vmd_cluster.tcl"))
        # Two close pairs separated by 8 A, plus an isolated frame.
        distances = tcl.call("dict", "create")
        for i in range(5):
            for j in range(i + 1, 5):
                distances = tcl.call("dict", "set", "test_distances", (i, j),
                                     0.5 if (i, j) in {(0, 1), (2, 3)} else 8.0)
        result = tcl.call("::vmd_cluster::gromos_groups", distances, 5, 1.0, 2)
        self.assertEqual([tuple(map(int, tcl.splitlist(g))) for g in tcl.splitlist(result)],
                         [(0, 1), (2, 3), (4,)])
        one = tcl.call("::vmd_cluster::gromos_groups", distances, 5, 9.0, 2)
        self.assertEqual([tuple(map(int, tcl.splitlist(g))) for g in tcl.splitlist(one)],
                         [(0, 1, 2, 3, 4), ()])

    def test_cluster_script_keeps_dynamic_values_as_tcl_literals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(selection='not hydrogen; puts bad', distfunc="fitrmsd", cutoff=2.5,
                                      num=5, first=0, last=-1, step=10, algorithm="gromos")
            script = vmd_workflow._cluster_script(root, root / "dna.pdb", root / "dna_whole.xtc", root / "out [x]", args)
        self.assertIn(r"not\ hydrogen\;\ puts\ bad", script)
        self.assertIn(r"out\ \[x\]", script)
        self.assertNotIn("source not hydrogen", script)

    def test_success_requires_tcl_marker_even_when_vmd_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = subprocess.CompletedProcess(["vmd"], 0, "ordinary VMD output\n", "")
            with patch.object(vmd_workflow.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "success marker"):
                    vmd_workflow._run_vmd(Path("vmd.exe"), output / "run.tcl", output, output, 1)
            self.assertTrue((output / "vmd.stdout.log").is_file())
            self.assertTrue((output / "workflow.log").is_file())

    def test_console_declares_namespace_before_its_context_variables(self) -> None:
        script = vmd_workflow._console_script(Path("C:/job/dna.pdb"), Path("C:/job/dna_whole.xtc"),
                                              Path("C:/out"), Path("C:/user_script.tcl"))
        self.assertLess(script.index("namespace eval ::vmd_workflow"), script.index("set ::vmd_workflow::job"))

    def test_cutoff_must_be_finite(self) -> None:
        self.assertEqual(vmd_workflow.positive_float("2.5"), 2.5)
        for value in ("nan", "inf", "-inf"):
            with self.assertRaises(argparse.ArgumentTypeError):
                vmd_workflow.positive_float(value)

    def test_tcl_medoid_uses_the_requested_cluster_metric(self) -> None:
        cluster_tcl = (Path(vmd_workflow.HERE) / "vmd_cluster.tcl").read_text(encoding="utf-8")
        self.assertIn("switch -- $distfunc", cluster_tcl)
        self.assertIn('set fixed_selection "index [join $selected_indices { }]"', cluster_tcl)
        self.assertIn("::vmd_cluster::medoid $molid $fixed_selection $distfunc $members", cluster_tcl)

    def test_console_copies_the_submitted_script_into_its_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = root / "job"
            job.mkdir()
            for name in ("dna.pdb", "dna_whole.xtc"):
                (job / name).write_text("fixture", encoding="utf-8")
            original = root / "original_user_script.tcl"
            original.write_text("puts {local user Tcl}\n", encoding="utf-8")
            vmd = root / "vmd.exe"
            vmd.write_text("fixture", encoding="utf-8")
            output = root / "console_output"
            args = argparse.Namespace(job=str(job), output=str(output), script_file=str(original), vmd=str(vmd), timeout=1)
            with patch.object(vmd_workflow, "_run_vmd"):
                vmd_workflow.run_console(args)
            self.assertEqual((output / "submitted.tcl").read_text(encoding="utf-8"), "puts {local user Tcl}\n")
            generated = (output / "run_console.tcl").read_text(encoding="utf-8")
            self.assertIn("submitted.tcl", generated)
            self.assertNotIn("original_user_script.tcl", generated)

    def test_success_needs_stdout_and_output_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "SUCCESS").write_text("action=cluster\n", encoding="utf-8")
            result = subprocess.CompletedProcess(["vmd"], 0, vmd_workflow.SUCCESS_TEXT + "\n", "")
            with patch.object(vmd_workflow.subprocess, "run", return_value=result):
                vmd_workflow._run_vmd(Path("vmd.exe"), output / "run.tcl", output, output, 1)
            self.assertEqual((output / "vmd_startup.tcl").read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
