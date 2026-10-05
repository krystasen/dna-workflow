"""Modification database, API validation and downstream chemical-identity tests."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dna_workflow as wf
import gui_server as gui
import md_pipeline as md
import modified_bases as mods


class ModifiedBaseTests(unittest.TestCase):
    def test_database_and_valid_sites(self):
        self.assertIn('O6MeG', [r['id'] for r in mods.available_bases()])
        site={'chain':'A','position':2,'base':'O6MeG'}
        self.assertEqual(mods.validate_modifications([site],['AGC'],'ol21_opc'),[site])
        for invalid, model in ((site,'charmm36_tip3p'),(site,'ol24_opc'),
                               (site|{'position':1},'ol21_opc'),(site|{'position':99},'ol21_opc'),
                               (site|{'position':True},'ol21_opc'),(site|{'chain':'B'},'ol21_opc'),
                               (site|{'base':'8oxoG'},'ol21_opc')):
            with self.subTest(invalid=invalid,model=model):
                with self.assertRaises(ValueError):
                    mods.validate_modifications([invalid],['AGC'],model)
        with self.assertRaises(ValueError):
            mods.validate_modifications([site,site],['AGC'],'ol21_opc')

    def test_build_and_snapshot_preserve_O6MeG(self):
        site={'chain':'A','position':2,'base':'O6MeG'}
        atoms=mods.apply_modifications(wf.build_atoms('AGC',False),['AGC'],[site])
        group=[a for a in atoms if a.resid==2]
        self.assertEqual({a.residue for a in group},{'OG6'})
        self.assertIn('CM',{a.name for a in group})
        self.assertNotIn('H1',{a.name for a in group})
        self.assertEqual(len(wf.residues(atoms)),3)
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp)
            mods.snapshot_library(job,[site])
            self.assertTrue((job/'modified_base_library/O6MeG/O6MeG.frcmod').is_file())
            self.assertIn('O6MeG.frcmod',mods.load_commands(job,[site]))
            self.assertEqual(len(mods.available_bases(mods.library_root(job))),1)

    def test_return_identity_never_coerces_modified_G(self):
        site={'chain':'A','position':2,'base':'O6MeG'}
        atoms=[wf.Atom(1,"O3'",'DA','A',1,(0.,0.,0.),'O'),
               wf.Atom(2,'P','OG6','A',2,(1.6,0.,0.),'P'),
               wf.Atom(3,'CM','OG6','A',2,(2.,0.,0.),'C'),
               wf.Atom(4,'P','DC','A',3,(3.2,0.,0.),'P')]
        frame=[replace(a,chain=' ',xyz=(1.,2.,3.)) for a in atoms]
        restored=md.restore_trajectory_identity(frame,atoms,['AGC'],[site])
        self.assertEqual(restored[1].residue,'OG6')
        self.assertEqual(wf.select_dna_atoms(restored,atoms),restored)
        with self.assertRaises(ValueError):
            wf.select_dna_atoms([replace(a,residue='DG') if a.residue=='OG6' else a for a in restored],atoms)

    def test_GUI_dispatches_explicit_sites(self):
        app=gui.App()
        site={'chain':'A','position':2,'base':'O6MeG'}
        with tempfile.TemporaryDirectory() as tmp:
            payload=dict(sequence='AGC',mode='single',output=str(Path(tmp)/'new'),md_model='ol21_opc',
                         modifications=[site],temperature=310,salt=.15,nvt_ps=100,npt_ps=1000,md_ns=100,
                         xtc_ps=10,padding=1.5,seed=42,box_type='octahedron')
            calls=[]
            with patch.object(app,'_start',side_effect=lambda name,work:work('test')),\
                 patch.object(app,'_run',side_effect=lambda _,argv,*a:calls.append(argv)),patch.object(app,'open_job'):
                app.create_job(payload)
            self.assertEqual(json.loads(calls[0][calls[0].index('--modifications')+1]),[site])
            with self.assertRaises(gui.APIError):
                app.create_job(payload|{'md_model':'ol24_opc'})
        html=(gui.GUI/'index.html').read_text(encoding='utf-8')
        self.assertIn('id="modification-position"',html)
        self.assertNotIn('O6‑MeG（暂不支持）',html)


if __name__=='__main__':
    unittest.main()
