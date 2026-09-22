"""Small deterministic end-to-end checks; no real DNA calculation is rerun."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

import numpy as np
import torch

import dna_dos_v2 as w
import plot_dna_dos_v2 as plot


def synthetic_gaussian_source(root: Path, single=False):
    """A labelled algebra/IO fixture, not a physical DNA electronic structure."""
    root.mkdir()
    rows, atoms = [], []
    parts = [
        [("O5'", "O", -3., 0.), ("C5'", "C", -4.4, 0.), ("C3'", "C", -1.4, 0.), ("O3'", "O", 0., 0.), ("H3T", "H", -0.1, 1.)],
        [("P", "P", 1.6, 0.), ("O5'", "O", 3., 0.), ("C5'", "C", 3., 1.4), ("O3'", "O", 5.5, 0.), ("H3T", "H", 6.46, 0.)],
    ]
    for strand, chain in enumerate("A" if single else "AB"):
        for position, part in enumerate(parts, 1):
            base = ("GA", "TC")[strand][position - 1]
            for name, element, x, y in part:
                serial = len(atoms) + 1
                xyz = np.asarray([x if strand == 0 else 5.5 - x, y + strand * 15., 0.])
                atoms.append((element, xyz))
                rows.append([serial, serial, chain, position, "D" + base, name, position if strand == 0 else 3 - position])
    fields = ["gaussian_atom_1based", "source_pdb_serial", "chain", "nucleotide_1based", "resname", "atom_name", "bp_1based"]
    with (root / "atom_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    pdb = []
    for row, (element, xyz) in zip(rows, atoms):
        pdb.append(f"ATOM  {row[0]:5d} {row[5]:<4s} {row[4]:>3s} {row[2]}{row[3]:4d}    "
                   f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}\n")
    (root / "dna_selected.pdb").write_text("".join(pdb) + "END\n", encoding="utf-8")
    charge = sum({value: key for key, value in w.ELEMENTS.items()}[element] for element, _ in atoms) % 2
    (root / "dna.gjf").write_text(f"#P synthetic matrix fixture\n\nSYNTHETIC TEST ONLY\n\n{charge} 1\n" +
                                   "".join(f"{element} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}\n" for element, xyz in atoms) + "\n", encoding="utf-8")
    nat, n = len(atoms), len(atoms) * 4
    rng = np.random.default_rng(941)
    seed = rng.normal(size=(n, n)) / np.sqrt(n)
    overlap = seed.T @ seed + 1.5 * np.eye(n)
    eigenvalues, eigenvectors = np.linalg.eigh(overlap)
    x = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    sqrt_s = (eigenvectors * eigenvalues ** 0.5) @ eigenvectors.T
    local_energies = np.r_[np.linspace(-20., -7., 19), -5.88]
    local = np.diag(np.tile(local_energies, n // 20))
    local[19, 39] = local[39, 19] = 0.13
    if not single:
        local[59, 79] = local[79, 59] = 0.13
        local[39, 79] = local[79, 39] = 0.02
    unitary = np.zeros((n, n))
    for start in range(0, n, 20):
        unitary[start:start + 20, start:start + 20] = np.linalg.qr(rng.normal(size=(20, 20)))[0]
    h0 = unitary @ local @ unitary.T
    fock = sqrt_s @ h0 @ sqrt_s / w.gaussian.HARTREE_TO_EV
    z = np.asarray([{value: key for key, value in w.ELEMENTS.items()}[element] for element, _ in atoms], dtype="<i8")
    ao_atoms = np.repeat(np.arange(1, nat + 1, dtype="<i8"), 4)
    label = lambda value: value.encode().ljust(64, b" ")
    with (root / "dna.mat").open("wb") as handle:
        def record(value):
            handle.write(struct.pack("<i", len(value)))
            handle.write(value)
            handle.write(struct.pack("<i", len(value)))
        record(label("Gaussian matrix elements") + struct.pack("<qq", 2, 7) + label("SYNTHETIC TEST"))
        record(label("SYNTHETIC TEST ONLY") + struct.pack("<9q", nat, n, n, charge, 1, int(z.sum()) - charge, 8, 8, 0))
        for value in (z, z * 0, z * 0, np.asarray([xyz for _, xyz in atoms]) / w.gaussian.BOHR_TO_ANGSTROM,
                      np.r_[ao_atoms, np.ones(n, dtype="<i8")]):
            record(value.tobytes())
        arrays = [("OVERLAP", overlap[np.tril_indices(n)], [-n, n]),
                  ("ORTHOGONAL BASIS", x.ravel(order="F"), [n, n]),
                  ("ALPHA FOCK MATRIX", fock[np.tril_indices(n)], [-n, n]),
                  ("ALPHA ORBITAL ENERGIES", np.linalg.eigvalsh(h0) / w.gaussian.HARTREE_TO_EV, [n, 0])]
        for name, values, shape in arrays:
            record(label(name) + struct.pack("<9q", 0, 1, len(values), 128, *shape, 0, 0, 0))
            for start in range(0, len(values), 128):
                record(np.asarray(values[start:start + 128], dtype="<f8").tobytes())
        record(label("END") + struct.pack("<9q", *([0] * 9)))
    return h0


class DnaDosTest(unittest.TestCase):
    def test_full_portable_roundtrip_resume_and_rejection(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="dna_dos_v2_test_") as tmp:
            root = Path(tmp)
            h0 = synthetic_gaussian_source(root / "source with spaces")
            inspection = root / "inspection"
            report = w.inspect_source(root / "source with spaces", inspection)
            self.assertEqual((report["state_count"], report["base_pair_count"], report["expected_boundary_candidates"]), (80, 2, 2))
            before = (inspection / "record_index.npz").stat().st_mtime_ns
            w.inspect_source(root / "source with spaces", inspection)
            self.assertEqual(before, (inspection / "record_index.npz").stat().st_mtime_ns)
            parameters = w.DEFAULTS | {"energy_max_eV": -5.76, "temperature_K": 10.}
            package = root / "portable package"
            w.export_package(inspection, package, parameters)
            metadata = w.prepare_material(package)
            self.assertEqual(metadata["excluded_probe_state_count"], 2)
            np.testing.assert_allclose(np.linalg.eigvalsh(np.load(package / "hamiltonian_eV.npy")), np.linalg.eigvalsh(h0), atol=1e-10)
            results = package / "results_c64"
            arguments = argparse.Namespace(data=package, output=results, device="cpu", dtype="complex64",
                                           rhs_chunk=7, start=0, stop=None, indices=None, resume=True)
            w.run_gpu(arguments)
            point = results / "points/energy_0000.npz"
            timestamp = point.stat().st_mtime_ns
            w.run_gpu(arguments)
            self.assertEqual(timestamp, point.stat().st_mtime_ns)
            w.merge_results(results)
            verified = w.verify_results(results, package)
            self.assertTrue(verified["complete"])
            self.assertEqual(verified["energy_count"], 49)
            moved = root / "returned elsewhere"
            shutil.copytree(results, moved, ignore=shutil.ignore_patterns("points"))
            self.assertTrue(w.verify_results(moved, package)["complete"])
            data = plot.calculate(moved, gateway=-5.88, temperature=10.)
            self.assertEqual(data["cumulative"].shape, (2,))
            self.assertTrue(np.all(data["cumulative"] >= 0))
            figure = plot.draw_figure(data)
            dots = figure.axes[3].collections[0]
            order = np.argsort(data["rotated"][:, 2])
            expected_colours = np.log10(np.maximum(data["cumulative"], 10**-0.5))[data["atom_to_pair"]][order]
            np.testing.assert_allclose(dots.get_array(), expected_colours)
            self.assertEqual(dots.get_alpha(), 0.70)
            self.assertEqual(len(figure.axes[3].collections), 1)
            plot.plt.close(figure)
            other = root / "wrong package"
            other.mkdir()
            manifest = w.read_json(package / "manifest.json") | {"run_id": "different-run"}
            w.save_json(other / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "do not match"):
                w.verify_results(moved, other)
            with self.assertRaisesRegex(ValueError, "different run"):
                w.load_point(point, metadata, "complex128", 0, -6.)
            point.unlink()
            with self.assertRaisesRegex(ValueError, "Missing 1 energy"):
                w.merge_results(results)
            self.assertEqual(w.read_json(results / "verification_report.json")["missing_indices"], [0])

    def test_diagonal_and_contact_solve_against_inverse(self):
        torch.set_num_threads(1)
        w.self_test()

    def test_partial_bins_and_gateway_are_explicit(self):
        full_grid = np.linspace(-6., 0., 1201)
        constant = np.ones((len(full_grid), 2))
        selected, _ = plot.choose_gateway(full_grid, constant, constant, 0, 310., manual=-5.320)
        self.assertEqual(selected["energy_eV"], -5.320)
        self.assertEqual(selected["selection"], "user-specified energy")
        energies = np.linspace(-6., -5.76, 49)
        values = np.column_stack([np.ones(49), 2 * np.ones(49)])
        centers, widths, binned = w.v2.bin_average_dos(energies, values, 0.024)
        integrated, density = plot.window_integral(centers, widths, binned, np.ones(2), -5.88, 100., 0.024)
        expected = 6 * plot.KB_EV * 100
        np.testing.assert_allclose(integrated, [expected, 2 * expected], rtol=1e-12)
        np.testing.assert_allclose(density, integrated / 0.024)
        with self.assertRaisesRegex(ValueError, "outside"):
            plot.window_integral(centers, widths, binned, np.ones(2), -6., 100., 0.024)
        changed = w.DEFAULTS | {"energy_step_eV": 0.1}
        with self.assertRaisesRegex(ValueError, "half"):
            w.energy_grid(changed)

    def test_atom_map_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="dna_dos_mapping_") as tmp:
            root = Path(tmp)
            synthetic_gaussian_source(root / "source")
            mapping = root / "source/atom_mapping.csv"
            lines = mapping.read_text().splitlines()
            lines[-1] = lines[-2]
            mapping.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "exactly once"):
                w.inspect_source(root / "source", root / "inspection")

    def test_single_strand_uses_5prime_and_3prime_contacts(self):
        with tempfile.TemporaryDirectory(prefix="dna_dos_single_") as tmp:
            root = Path(tmp)
            synthetic_gaussian_source(root / "source", single=True)
            report = w.inspect_source(root / "source", root / "inspection")
            self.assertEqual(report['expected_boundary_candidates'], 1)
            self.assertEqual([item['atom_name'] for item in report['contacts']], ["O5'", "O3'"])
            self.assertEqual(report['strand_sequences_5to3'], ['GA'])


if __name__ == "__main__":
    unittest.main()
