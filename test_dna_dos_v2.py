"""Small deterministic end-to-end checks; no real DNA calculation is rerun."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

import dna_dos_v2 as w
import orca_dos_v2 as orca
import plot_dna_dos_v2 as plot


def synthetic_gaussian_source(root: Path, single=False, modified_residue=None, chlorine=False, modified_position=1):
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
                if chlorine and strand==0 and position==modified_position and name=="C5'":
                    name,element='CL5','Cl'
                serial = len(atoms) + 1
                xyz = np.asarray([x if strand == 0 else 5.5 - x, y + strand * 15., 0.])
                atoms.append((element, xyz))
                residue=modified_residue if modified_residue and strand==0 and position==modified_position else "D"+base
                rows.append([serial, serial, chain, position, residue, name,
                             "" if single else position if strand == 0 else 3 - position])
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


def synthetic_orca_source(root: Path, single=False, modified_residue=None):
    h0 = synthetic_gaussian_source(root, single=single, modified_residue=modified_residue, modified_position=2)
    header, matrices = w.gaussian.scan_mat(root / "dna.mat")
    n = header["n_basis"]
    packed = w.gaussian.extract_real_matrix(root / "dna.mat", matrices["OVERLAP"], root / "s.npy",
                                            (n * (n + 1) // 2,), False)
    overlap = np.zeros((n, n))
    overlap[np.tril_indices(n)] = packed
    overlap += np.tril(overlap, -1).T
    values, vectors = np.linalg.eigh(overlap)
    x = (vectors * values ** -.5) @ vectors.T
    eps, q = np.linalg.eigh(h0 / w.gaussian.HARTREE_TO_EV)
    c = x @ q
    geometry = w.gaussian.gaussian_geometry(root / "dna.gjf")
    charge = int(header["charge"])
    text = f"! B3LYP 6-31G(d,p)\n* xyz {charge} 1\n" + "".join(
        f"{element} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}\n" for element, xyz in geometry) + "*\n"
    (root / "dna.inp").write_text(text, encoding="utf-8")
    molecule = {"Atoms": [{"Idx": i, "ElementNumber": int(header["atomic_numbers"][i]),
                           "ElementLabel": element, "Coords": (xyz / w.gaussian.BOHR_TO_ANGSTROM).tolist(),
                           "Basis": [{"Shell": "s"}] * 4} for i, (element, xyz) in enumerate(geometry)],
                "Charge": charge, "CoordinateUnits": "Bohr", "HFTyp": "RHF", "Multiplicity": 1,
                "MolecularOrbitals": {"EnergyUnit": "Eh", "MOs": [
                    {"MOCoefficients": c[:, i].tolist(), "OrbitalEnergy": float(eps[i]),
                     "Occupancy": 2. if i < header["electrons"] // 2 else 0.} for i in range(n)]},
                "S-Matrix": overlap.tolist()}
    w.save_json(root / "dna.json", {"Molecule": molecule})
    # Keep the Gaussian fixture alongside it to compare both paths from identical data.
    (root / "orca_mapping.csv").write_text((root / "atom_mapping.csv").read_text().replace(
        "gaussian_atom_1based", "orca_atom_1based"), encoding="utf-8")
    return h0


class DnaDosTest(unittest.TestCase):
    def test_orca_and_gaussian_identical_hamiltonian_roundtrip(self):
        torch.set_num_threads(1)
        for single, modified in ((True, "XG6"), (False, None)):
            with self.subTest(single=single), tempfile.TemporaryDirectory(prefix="dna_orca_v2_") as tmp:
                root = Path(tmp)
                h0 = synthetic_orca_source(root / "source", single=single, modified_residue=modified)
                parameters = w.DEFAULTS | {"energy_max_eV": -5.76, "temperature_K": 10.}
                w.inspect_source(root / "source", root / "gaussian_inspection")
                w.export_package(root / "gaussian_inspection", root / "gaussian_package", parameters)
                nparts = w.read_json(root / "gaussian_inspection/partitions.json")
                (root / "source/atom_mapping.csv").write_bytes((root / "source/orca_mapping.csv").read_bytes())
                report = orca.inspect_source(root / "source", root / "orca_inspection")
                self.assertEqual(report["protocol"], w.ORCA_PROTOCOL)
                self.assertEqual(w.read_json(root / "orca_inspection/partitions.json"), nparts)
                package = root / "orca_package"
                w.export_package(root / "orca_inspection", package, parameters)
                # A fresh interpreter must use only portable files, not local source paths.
                subprocess.run([sys.executable, str(package / "dna_dos_v2.py"), "prepare", "--data", str(package)],
                               cwd=package, check=True, capture_output=True, text=True)
                w.prepare_material(root / "gaussian_package")
                np.testing.assert_allclose(np.load(package / "_prepare_work/h0_eV.npy"), h0, atol=1e-10)
                for dtype in ("complex64", "complex128"):
                    results = package / dtype
                    w.run_gpu(argparse.Namespace(data=package, output=results, device="cpu", dtype=dtype,
                                                rhs_chunk=7, start=0, stop=None, indices=None, resume=True))
                    self.assertTrue(w.merge_results(results)["complete"])
                    self.assertTrue(w.verify_results(results, package)["complete"])
                    data = plot.calculate(results, gateway=-5.88, temperature=10.)
                    self.assertTrue(np.isfinite(data["cumulative"]).all())
                    if dtype == "complex128":
                        gau_results = root / "gaussian_results"
                        w.run_gpu(argparse.Namespace(data=root / "gaussian_package", output=gau_results, device="cpu", dtype=dtype,
                                                    rhs_chunk=7, start=0, stop=None, indices=None, resume=True))
                        w.merge_results(gau_results)
                        with np.load(results / "dos2d_results.npz") as actual, np.load(gau_results / "dos2d_results.npz") as expected:
                            np.testing.assert_allclose(actual["dos2d"], expected["dos2d"], rtol=1e-9, atol=1e-10)
                            np.testing.assert_allclose(actual["contact_dos"], expected["contact_dos"], rtol=1e-9, atol=1e-10)
                        with self.assertRaisesRegex(ValueError, "protocol"):
                            w.verify_results(results, root / "gaussian_package")
                guide = (package / "GPU_GUIDE.md").read_text(encoding="utf-8")
                self.assertIn(w.ORCA_PROTOCOL, guide)
                self.assertIn("orca_partitions.json", guide)
                self.assertNotIn("dna.gjf", guide)

    def test_orca_rejects_incomplete_wrong_or_nonorthogonal_data(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="dna_orca_invalid_") as tmp:
            root = Path(tmp)
            synthetic_orca_source(root / "source", single=True)
            (root / "source/atom_mapping.csv").write_bytes((root / "source/orca_mapping.csv").read_bytes())
            path = root / "source/dna.json"
            original = w.read_json(path)
            def inspect_and_export(label):
                orca.inspect_source(root / "source", root / label)
                return w.export_package(root / label, root / (label + "_package"), w.DEFAULTS)
            import copy
            for label, change, message in (
                ("missing_mo", lambda m: m["MolecularOrbitals"]["MOs"].pop(), "incomplete"),
                ("missing_s", lambda m: m.pop("S-Matrix"), "S-Matrix"),
                ("fractional", lambda m: m["MolecularOrbitals"]["MOs"][0].update(Occupancy=1.), "occupations"),
                ("wrong_xyz", lambda m: m["Atoms"][0]["Coords"].__setitem__(0, 100.), "coordinates")):
                value = copy.deepcopy(original)
                change(value["Molecule"])
                w.save_json(path, value)
                with self.subTest(label=label), self.assertRaisesRegex(ValueError, message):
                    inspect_and_export(label)
            value = copy.deepcopy(original)
            # JSON object order is not an interface contract: atoms may follow MO data.
            value["Molecule"] = {key: value["Molecule"][key] for key in
                                 ("Charge", "CoordinateUnits", "HFTyp", "Multiplicity", "MolecularOrbitals", "Atoms", "S-Matrix")}
            w.save_json(path, value)
            self.assertEqual(orca.inspect_source(root / "source", root / "reordered")["atom_count"], 10)
            value = copy.deepcopy(original)
            value["Molecule"]["MolecularOrbitals"]["MOs"][0]["MOCoefficients"][0] += .2
            w.save_json(path, value)
            inspect_and_export("bad_c")
            with self.assertRaisesRegex(ValueError, "C.T S C"):
                w.prepare_material(root / "bad_c_package")

    def test_empty_probe_window_keeps_nearest_states_and_resumes_with_review(self):
        torch.set_num_threads(1)
        partition = {'nucleotides': [{'name': 's1_01_G', 'ao_ranges': [[1, 2]]},
                                    {'name': 's1_02_A', 'ao_ranges': [[3, 10]]}],
                     'boundaries': [{'previous': 's1_01_G', 'current': 's1_02_A'}]}
        onsite = np.asarray([0., 0., -2100., -179., -8., -7., -6.08, -5.5, -1., 5.])
        h = np.zeros((10, 10))
        h[2, 0], h[3, 0], h[6, 0] = 15., 13., 1.8
        rows = w.probe_candidates(partition, onsite, h, w.DEFAULTS)
        self.assertIn(5, [row['local_index_1based'] for row in rows])
        self.assertFalse(any(row['onsite_eV'] < -100. for row in rows))
        self.assertFalse(any(row['in_v2_window'] for row in rows))
        with tempfile.TemporaryDirectory(prefix='dna_dos_probe_resume_') as tmp:
            root = Path(tmp)
            synthetic_gaussian_source(root/'source', single=True)
            w.inspect_source(root/'source', root/'inspection')
            package = root/'package'
            w.export_package(root/'inspection', package, w.DEFAULTS | {'spurious_min_eV': -5.87})
            with self.assertRaisesRegex(ValueError, r'No dangling-bond candidate in s1_02_A.*Nearest states: local=20'):
                w.prepare_material(package)
            gc.collect()
            self.assertFalse((package/'metadata.json').exists())
            retained = [package/'hamiltonian_eV.npy', package/'_prepare_work/local_vectors.npz',
                        package/'_prepare_work/matrix_validation.json']
            before = [path.stat().st_mtime_ns for path in retained]
            review = root/'reviewed_probe_indices.json'
            w.save_json(review, {'s1_02_A': 20})
            metadata = w.prepare_material(package, review)
            self.assertEqual([path.stat().st_mtime_ns for path in retained], before)
            self.assertEqual(metadata['excluded_probe_state_count'], 1)
            self.assertEqual(metadata['probe_identification'], 'reviewed explicit indices')
            self.assertEqual(metadata['parameters']['spurious_min_eV'], -5.87)
            self.assertEqual(w.prepare_material(package, review), metadata)
            self.assertEqual([path.stat().st_mtime_ns for path in retained], before)
            w.save_json(review, {'s1_02_A': 19})
            with self.assertRaisesRegex(ValueError, 'Changed probe choices'):
                w.prepare_material(package, review)

    def test_registered_modified_residues_and_chlorine_are_preserved(self):
        torch.set_num_threads(1)
        self.assertEqual(plot.reference.VDW_A['Cl'],1.75)
        library = Path(__file__).resolve().parent / 'modified_base_library'
        catalog = w.read_json(library / 'catalog.json')
        registered = {}
        for identifier, row in catalog['entries'].items():
            entry = w.read_json(library / row['path'])
            for template in entry['templates']:
                registered[template['name']] = (entry['parent_base'], identifier == 'Cl5dC')
        self.assertEqual({name: parent for name, (parent, _) in registered.items()}, w.MODIFIED_BASES)
        parameters = w.DEFAULTS | {'energy_max_eV': -5.976, 'energy_step_eV': .004, 'temperature_K': 10.}
        for residue, (parent, chlorine) in registered.items():
            for single in (True, False):
                with self.subTest(residue=residue, single=single), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    # All codes occur at an O-P boundary in this algebra/IO fixture.
                    synthetic_gaussian_source(root/'source', single=single, modified_residue=residue,
                                              chlorine=chlorine, modified_position=2)
                    report = w.inspect_source(root/'source', root/'inspection')
                    partition = w.read_json(root/'inspection/partitions.json')
                    item = partition['nucleotides'][1]
                    self.assertEqual(item['residue_name'], residue)
                    self.assertEqual(item['base'], parent)
                    self.assertEqual(item['name'], f's1_02_{residue}')
                    self.assertEqual(report['state_count'], 40 if single else 80)
                    atoms = w.v2.load_geometry(root/'source/dna.gjf')
                    self.assertEqual('Cl' in [element for element, _ in atoms], chlorine)
                    package = root/'package'
                    w.export_package(root/'inspection', package, parameters)
                    self.assertEqual((package/'spurious30_v2.py').read_bytes(),
                                     (w.HERE/'spurious30_v2.py').read_bytes())
                    metadata = w.prepare_material(package)
                    self.assertEqual(metadata['excluded_probe_state_count'], 1 if single else 2)
                    prepared = w.read_json(package/'gaussian_partitions.json')['nucleotides'][1]
                    self.assertEqual(prepared['residue_name'], residue)
                    self.assertEqual(len(prepared['exclude_probe_local']), 1)
                    results = package/'results_c64'
                    w.run_gpu(argparse.Namespace(data=package, output=results, device='cpu', dtype='complex64',
                                                rhs_chunk=7, start=0, stop=None, indices=None, resume=True))
                    self.assertTrue(w.merge_results(results)['complete'])
                    data = plot.calculate(results, gateway=-5.988, temperature=10.)
                    self.assertTrue(np.all(np.isfinite(data['cumulative'])))
                    self.assertGreater(np.min(data['volumes']), 0.)

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
