import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("amber_backend", HERE / "amber_backend.py")
amber = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(amber)


def pdb_line(serial, name, residue, chain, resid, x):
    field = f" {name:<3}" if len(name) < 4 else name
    return f"ATOM  {serial:5d} {field:4s} {residue:>3s} {chain}{resid:4d}    {x:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           C"


class AmberBackendTest(unittest.TestCase):
    def test_prepare_input_maps_c5m_and_preserves_two_strands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dna_input.pdb"
            source.write_text("\n".join([
                pdb_line(1, "P", "DA", "A", 1, 0), pdb_line(2, "C5M", "DT", "A", 2, 1), "TER",
                pdb_line(3, "P", "DA", "B", 1, 2), pdb_line(4, "C5M", "DT", "B", 2, 3), "END", ""
            ]), encoding="ascii")
            report = amber._prepare_input_pdb(source, root / "amber_input.pdb", ["AT", "AT"])
            text = (root / "amber_input.pdb").read_text(encoding="ascii")
            self.assertEqual(report["C5M_to_C7"], 2)
            self.assertNotIn("C5M", text)
            self.assertEqual(text.count("TER\n"), 2)
            self.assertIn(" C7 ", text)

    def test_box_and_salt_script_are_explicit(self):
        job = Path("/tmp/amber-job")
        cubic = amber._solvation_text(job, "ol21_opc", "cubic", 1.5, "Na+", 3, "Na+")
        triclinic = amber._solvation_text(job, "ol21_opc", "triclinic", 1.5)
        octahedron = amber._solvation_text(job, "ol21_opc", "octahedron", 1.5)
        self.assertIn("loadMol2", cubic)
        self.assertIn("solvateBox dna OPCBOX 15 iso", cubic)
        self.assertIn("addIonsRand dna Na+ 0", cubic)
        self.assertIn("addIonsRand dna Na+ 3", cubic)
        self.assertIn("addIonsRand dna Cl- 3", cubic)
        self.assertIn("solvateBox dna OPCBOX 15\n", triclinic)
        self.assertIn("solvateOct dna OPCBOX 15", octahedron)

    def test_dodecahedron_and_non_amber_model_are_rejected(self):
        with self.assertRaisesRegex(amber.AmberBackendError, "dodecahedron"):
            amber._validate_box({"box_type": "dodecahedron", "padding_nm": 1.5})
        with self.assertRaisesRegex(amber.AmberBackendError, "GROMACS backend"):
            amber._normalise_cfg({"md_model": "charmm36_tip3p", "strands_5to3": ["AT"]})

    def test_posre_is_inserted_inside_each_dna_moleculetype(self):
        top = """[ moleculetype ]\nDNA_A 3\n[ atoms ]\n1 C 1 DA C 1 0 1\n2 H 1 DA H 1 0 1\n[ moleculetype ]\nDNA_B 3\n[ atoms ]\n1 C 1 DT C 1 0 1\n[ moleculetype ]\nWAT 3\n[ atoms ]\n1 OW 1 WAT OW 1 0 1\n2 HW 1 WAT HW1 1 0 1\n3 HW 1 WAT HW2 1 0 1\n[ system ]\ntest\n[ molecules ]\nDNA_A 1\nDNA_B 1\nWAT 2\n"""
        manifest = {"atoms": [{"chain": "A"}, {"chain": "A"}, {"chain": "B"}], "bonds": [], "charge": 0, "md_model": "ol21_opc"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topol.top"
            path.write_text(top, encoding="ascii")
            self.assertEqual(amber._add_dna_posres(path, manifest), ["posre_DNA_DNA_A.itp", "posre_DNA_DNA_B.itp"])
            result = path.read_text(encoding="utf-8")
            self.assertLess(result.index('"posre_DNA_DNA_A.itp"'), result.index("[ moleculetype ]", result.index("DNA_B")))
            self.assertLess(result.index('"posre_DNA_DNA_B.itp"'), result.index("[ moleculetype ]", result.index('"posre_DNA_DNA_B.itp"')))

    def test_palindromic_duplex_uses_one_local_posre_for_two_instances(self):
        top = """[ moleculetype ]\nDNA_GGCC 3\n[ atoms ]\n1 C 1 DG C 1 0 1\n2 C 2 DC C 1 0 1\n[ moleculetype ]\nWAT 3\n[ atoms ]\n1 OW 1 WAT OW 1 0 1\n2 HW 1 WAT HW1 1 0 1\n3 HW 1 WAT HW2 1 0 1\n[ system ]\ntest\n[ molecules ]\nDNA_GGCC 2\nWAT 4\n"""
        manifest = {"atoms": [{"chain": "A"}, {"chain": "A"}, {"chain": "B"}, {"chain": "B"}], "bonds": [], "charge": 0, "md_model": "ol21_opc"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topol.top"
            path.write_text(top, encoding="ascii")
            self.assertEqual(amber._add_dna_posres(path, manifest), ["posre_DNA_DNA_GGCC.itp"])
            posre = (path.parent / "posre_DNA_DNA_GGCC.itp").read_text(encoding="ascii")
            self.assertEqual(sum(line.lstrip()[:1].isdigit() for line in posre.splitlines()), 2)

    def test_neutral_charge_tolerance_accepts_decimal_roundoff(self):
        from types import SimpleNamespace as Obj
        dna_res = Obj(name="DA")
        dna = Obj(name="C", residue=dna_res, idx=0, charge=0., atomic_number=6)
        water = Obj(name="WAT")
        water.atoms = [Obj(residue=water, charge=q, atomic_number=z)
                       for z,q in [(8,0.), (1,.679142), (1,.679142), (0,-1.358284)]]
        ion = Obj(name="K+")
        ion.atoms = [Obj(residue=ion, charge=.000175712, atomic_number=19)]
        structure = Obj(atoms=[dna,*water.atoms,*ion.atoms], residues=[dna_res,water,ion], bonds=[])
        manifest = {"atoms":[{"name":"C", "residue":"DA", "chain":"A", "resid":1}], "bonds":[]}
        result = amber._validate_opc_and_prefix(structure, manifest)
        self.assertAlmostEqual(result["total_charge_e"], .000175712)
        ion.atoms[0].charge = 1.0
        with self.assertRaisesRegex(RuntimeError, "not neutral"):
            amber._validate_opc_and_prefix(structure, manifest)

    def test_opc_topology_requires_settles_and_virtual_sites3(self):
        top = """[ moleculetype ]\nWAT 3\n[ atoms ]\n1 OW 1 WAT OW 1 0 1\n2 HW 1 WAT HW1 1 0 1\n3 EP 1 WAT EP 1 0 1\n[ settles ]\n1 1 0.1 0.16\n[ virtual_sites3 ]\n3 1 2 1 1 0.1 0.1\n[ system ]\ntest\n[ molecules ]\nWAT 1\n"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topol.top"
            path.write_text(top, encoding="ascii")
            amber._assert_opc_gromacs_sections(path)


if __name__ == "__main__":
    unittest.main()
