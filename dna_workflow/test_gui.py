"""Minimal local API checks; no WSL, GROMACS, Gaussian, VMD, or production job is run."""
from __future__ import annotations

import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gui_server as gui
import dna_workflow as wf
import md_pipeline as md


class GuiApiTest(unittest.TestCase):
    def test_tcl_executor_removed_but_cluster_and_tk_preserved(self):
        html = (gui.GUI / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("vmd-console", html)
        self.assertNotIn("vmd_console", html)
        self.assertNotIn("VMD Tcl 执行器", html)
        for action in ("cluster", "open_vmd", "open_tk"):
            self.assertIn(f'data-action="{action}"', html)
            self.assertIn(action, gui.ALLOWED_ACTIONS)
        status, data = self.request("/api/action", {"action": "vmd_console"})
        self.assertEqual(status, 400)
        self.assertIn("Unsupported action", data["error"])
        self.assertIsNone(self.app.task)

    def test_box_comparison_removed_and_defaults_preserved(self):
        html = (gui.GUI / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("compare_boxes", html)
        self.assertNotIn("compare-box", html)
        self.assertIn('value="dodecahedron"', html)
        self.assertIn('value="gromos" selected', html)

    def test_trjconv_missing_chains_restore_by_atom_order(self):
        topology = [wf.Atom(i+1, "P", "D"+b, c, r, (0.,0.,0.), "P")
                    for i,(c,r,b) in enumerate([("A",1,"A"),("A",2,"C"),("B",1,"G"),("B",2,"T")])]
        from dataclasses import replace
        exported = [replace(a, chain=" ", xyz=(1.,2.,3.)) for a in topology]
        restored = md.restore_trajectory_identity(exported, topology, ["AC","GT"])
        self.assertEqual([a.chain for a in restored], ["A","A","B","B"])
        self.assertTrue(all(a.xyz == (1.,2.,3.) for a in restored))
        with self.assertRaises(ValueError):
            md.restore_trajectory_identity(exported[:-1], topology, ["AC","GT"])

    def setUp(self):
        self.app = gui.App()
        self.server = gui.make_server("127.0.0.1", 0, self.app)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, payload, origin=True):
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = f"http://127.0.0.1:{self.server.server_port}"
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        conn.request("POST", path, body, headers)
        response = conn.getresponse()
        data = json.loads(response.read().decode())
        conn.close()
        return response.status, data

    def test_o6meg_is_rejected_before_any_build(self):
        status, data = self.request("/api/job/create", {"sequence": "ACO6-MeGTA"})
        self.assertEqual(status, 400)
        self.assertIn("not supported", data["error"])
        self.assertIsNone(self.app.job)

    def test_action_requires_open_job(self):
        status, data = self.request("/api/action", {"action": "solvate"})
        self.assertEqual(status, 409)
        self.assertIn("Open an existing job", data["error"])

    def test_reopen_clears_old_failure_but_keeps_missing_return_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = Path(tmp)
            (job / "workflow.json").write_text('{}')
            for name in ('em.log', 'em.tpr', 'em.mdp'):
                (job / name).write_text('test')
            self.app.job = job
            self.app.task = dict(action='solvate', status='failed', error='old failure')
            self.app.open_job({'path': str(job)})
            self.assertIsNone(self.app.task)
            stage = self.app.state()['job']['md_stages']['em']
            self.assertFalse(stage['can_verify'])
            self.assertEqual(stage['status'], 'blocked')
            self.assertIn('em.gro', stage['reason'])
            self.app.task = dict(action='verify_em', status='running')
            with self.assertRaises(gui.APIError):
                self.app.open_job({'path': str(job)})
            self.assertEqual(self.app.task['status'], 'running')

    def test_wsl_argv_keeps_space_paths_as_single_arguments(self):
        argv = gui.wsl_argv("Ubuntu-24.04", "/opt/gromacs/2025.4/bin/gmx",
                            "/mnt/d/data/obsidian/05 Code/dna_workflow/md_pipeline.py",
                            "/mnt/d/data/DNA DoS/short ds", "prepare_md", [])
        self.assertEqual(argv[6], "/mnt/d/data/obsidian/05 Code/dna_workflow/md_pipeline.py")
        self.assertEqual(argv[9], "/mnt/d/data/DNA DoS/short ds")

    def test_thread_mpi_uses_one_rank_and_gmx_mpi_does_not(self):
        job = Path(r"D:\data\DNA DoS\short ds")
        with patch.object(self.app, "_wsl_path", side_effect=["/mnt/d/code/md_pipeline.py", "/mnt/d/data/DNA DoS/short ds"]):
            thread_mpi = self.app._wsl_command(job, "nvt", {"ntomp": 2})
        with patch.object(self.app, "_wsl_path", side_effect=["/mnt/d/code/md_pipeline.py", "/mnt/d/data/DNA DoS/short ds"]):
            mpi = self.app._wsl_command(job, "nvt", {"gmx": "/opt/gromacs/bin/gmx_mpi", "ntomp": 2})
        self.assertEqual(thread_mpi[-4:], ["--ntmpi", "1", "--ntomp", "2"])
        self.assertNotIn("--ntmpi", mpi)

    def test_post_requires_local_origin(self):
        status, data = self.request("/api/job/open", {"path": "D:/ignored"}, origin=False)
        self.assertEqual(status, 403)
        self.assertIn("local GUI", data["error"])

    def test_second_server_cannot_reuse_the_live_port(self):
        with self.assertRaises(OSError):
            gui.make_server('127.0.0.1', self.server.server_port)

    def test_gui_never_accepts_mdrun_actions(self):
        for action in ("em", "nvt", "npt", "md", "production", "compare_boxes", "dos_run", "dos_prepare"):
            status, _ = self.request("/api/action", {"action": action})
            self.assertEqual(status, 400)

    def test_analysis_rejects_existing_output_and_missing_trajectory(self):
        with tempfile.TemporaryDirectory(prefix='dna_analysis_') as tmp:
            job=Path(tmp)
            (job/'workflow.json').write_text('{}')
            self.app.open_job({'path':str(job)})
            status, data=self.request('/api/action', {'action':'prepare_relax','output':str(job)})
            self.assertEqual(status,400)
            self.assertIsNone(self.app.task)
            status, data=self.request('/api/action', {'action':'cluster','output':str(job/'clusters')})
            self.assertEqual(status,400)
            self.assertIn('dna_whole.xtc',data['error'])

    def test_cluster_dispatches_vmd_not_md(self):
        with tempfile.TemporaryDirectory(prefix='dna_cluster_api_') as tmp:
            job=Path(tmp)
            (job/'workflow.json').write_text('{}')
            for name in ('dna.pdb','dna_whole.xtc'):
                (job/name).write_text('fixture')
            self.app.open_job({'path':str(job)})
            calls=[]
            with patch.object(self.app,'_start',side_effect=lambda action,work:work('test')), patch.object(self.app,'_run',side_effect=lambda tid,argv,cwd=gui.HERE:calls.append(argv)):
                status,data=self.request('/api/action', {'action':'cluster','output':str(job/'new'), 'cutoff':2.5, 'first':0,'last':20,'step':2})
            self.assertEqual(status,200)
            self.assertEqual(len(calls),1)
            self.assertIn('cluster',calls[0])
            self.assertIn('--cutoff',calls[0])
            self.assertEqual(calls[0][calls[0].index('--algorithm') + 1], 'gromos')
            self.assertNotIn('mdrun',calls[0])

    def test_trajectory_whole_cluster_center_order_and_no_md(self):
        for strands in (["AC"], ["AC", "GT"]):
            with self.subTest(strands=strands), tempfile.TemporaryDirectory(prefix='dna_trajectory_') as tmp:
                job = Path(tmp)
                wf.save_json(job/'workflow.json', {'strands_5to3':strands,'forcefield':'charmm.ff'})
                for name in ('md.tpr','md.xtc','topology.pdb'):
                    (job/name).write_text('fixture')
                atoms = [wf.Atom(k*2+i+1,'P','D'+b,chr(65+k),i+1,(0.,0.,0.),'P')
                         for k,s in enumerate(strands) for i,b in enumerate(s)]
                calls=[]
                def fake(argv,*a,**kw):
                    calls.append(argv)
                    return ''
                with patch.object(md,'stream_process',fake), patch.object(md,'read_pdb',return_value=atoms), patch.object(md,'validate_job_dna'):
                    md.run(SimpleNamespace(job=job,gmx='gmx',stage='handoff'))
                trj=[c for c in calls if 'trjconv' in c]
                pbc=[c[c.index('-pbc')+1] for c in trj if '-pbc' in c]
                self.assertEqual(pbc,['whole','cluster','none'] if len(strands)==2 else ['whole','none'])
                center=next(c for c in trj if '-center' in c)
                self.assertEqual(center[center.index('-o')+1], 'dna_whole.xtc')
                self.assertFalse(any('mdrun' in c for c in calls))
                self.assertEqual((job/'dna.ndx').read_text().split()[3:], [str(i) for i in range(1,len(atoms)+1)])
                self.assertTrue(json.loads((job/'handoff.complete.json').read_text())['centered'])

    def test_prepare_all_stages_does_not_run_md(self):
        with tempfile.TemporaryDirectory(prefix="prepare_only_") as tmp:
            job = Path(tmp)
            (job / "workflow.json").write_text('{}')
            for name in ("topol.top", "solv_ions.gro", "solvate.complete.json", "em.gro", "nvt.gro", "nvt.cpt", "npt.gro", "npt.cpt"):
                (job / name).write_text('fixture')
            for stage in md.MD_STAGES:
                (job / f"{stage}.mdp").write_text('nsteps = 100\n')
            calls = []
            def fake(argv, cwd, *a, **kw):
                calls.append(argv)
                if 'grompp' in argv:
                    (job / argv[argv.index('-o')+1]).write_text('tpr fixture')
                return ''
            with patch.object(md, 'stream_process', fake), patch.object(md, 'verified_return', return_value=True):
                for stage in md.MD_STAGES:
                    md.run(SimpleNamespace(job=job,gmx='gmx',stage=f'prepare_{stage}'))
            self.assertEqual(len([c for c in calls if 'grompp' in c]),4)
            self.assertFalse(any('mdrun' in c for c in calls))
            npt = next(c for c in calls if 'grompp' in c and 'npt.tpr' in c)
            self.assertEqual(npt[npt.index('-t')+1], 'nvt.cpt')
            prod = next(c for c in calls if 'grompp' in c and 'md.tpr' in c)
            self.assertEqual(prod[prod.index('-t')+1], 'npt.cpt')

    def test_server_return_without_local_marker_and_incomplete_result(self):
        with tempfile.TemporaryDirectory(prefix='remote_result_') as tmp:
            job = Path(tmp)
            (job/'workflow.json').write_text('{}')
            (job/'nvt.mdp').write_text('integrator = md\nnsteps = 100\ndt = 0.002\npcoupl = no\n')
            (job/'nvt.tpr').write_text('fixture')
            atom = '    1SOL     OW    1   0.000   0.000   0.000\n'
            (job/'nvt.gro').write_text('fixture\n1\n'+atom+'1 1 1\n')
            (job/'nvt.log').write_text('Finished mdrun\n')
            (job/'nvt.cpt').write_text('fixture')
            def command(argv,*a,**kw):
                if '-s' in argv:
                    return 'integrator = md\nnsteps = 100\ndt = 0.002\nnatoms = 1\npcoupl = No\n'
                return '#atoms = 1\nstep = 100\nt = 0.2\n'
            md.verify_return(job,'nvt',command)
            self.assertTrue(md.verified_return(job,'nvt'))
            self.assertFalse((job/'nvt.complete.json').exists())
            (job/'nvt.cpt').write_text('changed checkpoint')
            self.assertFalse(md.verified_return(job,'nvt'))
            def partial(argv,*a,**kw):
                return command(argv) if '-s' in argv else '#atoms = 1\nstep = 50\nt = 0.1\n'
            with self.assertRaisesRegex(ValueError,'incomplete'):
                md.verify_return(job,'nvt',partial)
            (job/'nvt.cpt').unlink()
            with self.assertRaises(FileNotFoundError):
                md.verify_return(job,'nvt',command)

    def test_imported_md_is_explicitly_unverified(self):
        with tempfile.TemporaryDirectory(prefix="dna_gui_") as temporary:
            job = Path(temporary)
            (job / "workflow.json").write_text(json.dumps({"forcefield": "charmm36.ff"}), encoding="utf-8")
            (job / "charmm36.ff").mkdir()
            for name in ("topology.pdb", "md.tpr", "md.xtc"):
                (job / name).write_text("fixture", encoding="utf-8")
            self.app.open_job({"path": str(job)})
            state = self.app.state()["job"]
            self.assertEqual(state["stages"]["md"], "unverified")
            self.assertEqual(state["production_md"], "生产MD完成情况未验证")

    def test_dos_inspection_does_not_require_md_or_cuda(self):
        with tempfile.TemporaryDirectory(prefix="dna_dos_api_") as tmp:
            job = Path(tmp)
            (job / "workflow.json").write_text('{}')
            source = job / "gaussian_input"
            source.mkdir()
            self.app.open_job({"path": str(job)})
            calls = []
            def fake_run(task_id, argv, cwd=gui.HERE):
                calls.append(argv)
                output = Path(argv[argv.index('--output') + 1])
                output.mkdir(parents=True)
                (output / 'inspection.json').write_text(json.dumps({'source': str(source), 'source_files': {}, 'state_count': 80}))
            with patch.object(self.app, '_start', side_effect=lambda action, work: work('test')), patch.object(self.app, '_run', side_effect=fake_run):
                status, data = self.request('/api/action', {'action': 'dos_inspect', 'source': str(source), 'dos_python': sys.executable})
            self.assertEqual(status, 200)
            self.assertEqual(data['job']['dos']['inspection']['state_count'], 80)
            self.assertIn('inspect', calls[0])
            self.assertNotIn('wsl.exe', calls[0])
            self.assertNotIn('prepare', calls[0])

    def test_dos_export_is_separate_from_compute_and_persists_handoff(self):
        with tempfile.TemporaryDirectory(prefix="dna_dos_export_api_") as tmp:
            job = Path(tmp)
            (job / 'workflow.json').write_text('{}')
            source = job / 'source'
            source.mkdir()
            inspection = job / 'dos_v2/inspection'
            inspection.mkdir(parents=True)
            (inspection / 'inspection.json').write_text(json.dumps({'source': str(source)}))
            self.app.open_job({'path': str(job)})
            output = job / 'new_gpu_package'
            calls = []
            def fake_run(task_id, argv, cwd=gui.HERE):
                calls.append(argv)
                output.mkdir()
                (output / 'manifest.json').write_text(json.dumps({'phase': 'exported', 'parameters': {}, 'run_id': 'example'}))
            with patch.object(self.app, '_start', side_effect=lambda action, work: work('test')), patch.object(self.app, '_run', side_effect=fake_run):
                status, data = self.request('/api/action', {'action': 'dos_export', 'source': str(source), 'output': str(output), 'dos_python': sys.executable})
            self.assertEqual(status, 200)
            self.assertEqual(data['job']['dos']['package_phase'], 'exported')
            self.assertIn('export', calls[0])
            self.assertNotIn('run', calls[0])
            fresh = gui.App()
            fresh.open_job({'path': str(job)})
            self.assertEqual(fresh.state()['job']['dos']['run_id'], 'example')
            status, _ = self.request('/api/action', {'action': 'dos_export', 'source': str(source), 'output': str(output), 'dos_python': sys.executable})
            self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
