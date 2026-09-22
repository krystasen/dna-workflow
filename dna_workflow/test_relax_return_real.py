"""Test EM-return extraction using an existing completed short EM as a labelled fixture.

No new EM is run; the fixture is not the newly prepared representative-relaxation job.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import dna_workflow as wf
import md_extensions as ext


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--gmx',required=True)
    args=p.parse_args()
    source=args.source.resolve()
    with tempfile.TemporaryDirectory(prefix='relax_return_fixture_') as temporary:
        job=Path(temporary)
        cfg=json.loads((source/'workflow.json').read_text())
        cfg.update(role='representative_relaxation',source_job=str(source),source_frame=str(source/'topology.pdb'))
        wf.save_json(job/'workflow.json',cfg)
        for name in ('em.tpr','em.gro','em.log','em.mdp','em.submitted.mdp','solv_ions.gro','topology.pdb'):
            shutil.copyfile(source/name,job/name)
        shutil.copytree(source/cfg['forcefield'],job/cfg['forcefield'])
        report=ext.verify_relax(SimpleNamespace(job=job,gmx=args.gmx))
        assert report['atom_count']==379 and report['charge']==-10
        assert {a.residue for a in wf.read_pdb(job/'relaxed_dna.pdb')} <= wf.BASES
        baseline = wf.read_pdb(job/'relaxed_dna.pdb')
        reference = wf.restore_identity(wf.read_pdb(job/'topology.pdb'), cfg['strands_5to3'])
        original_gro = (job/'em.gro').read_text()
        lines = original_gro.splitlines()
        box = list(map(float, lines[-1].split()))
        v1 = (box[0], box[3], box[4]) if len(box) == 9 else (box[0], 0, 0)
        v2 = (box[5], box[1], box[6]) if len(box) == 9 else (0, box[1], 0)
        # Reproduce a split nucleotide AND strands in different periodic images.
        for i, atom in enumerate(reference):
            if atom.chain != 'B':
                continue
            line = lines[i+2]
            xyz = [float(line[j:j+8]) + v2[k] + (v1[k] if i == len(reference)-1 else 0)
                   for k,j in enumerate((20,28,36))]
            lines[i+2] = line[:20] + ''.join(f'{x:8.3f}' for x in xyz) + line[44:]
        (job/'em.gro').write_text('\n'.join(lines)+'\n', encoding='ascii')
        try:
            wf.validate_dna(ext._gro_dna_atoms(job/'em.gro', reference), cfg['strands_5to3'], job/cfg['forcefield'], hydrogens=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Wrapped fixture must fail the former direct-coordinate validation')
        report = ext.verify_relax(SimpleNamespace(job=job,gmx=args.gmx))
        restored = wf.read_pdb(job/'relaxed_dna.pdb')
        shift = [sum(b.xyz[k]-a.xyz[k] for a,b in zip(baseline,restored))/len(baseline) for k in range(3)]
        assert max(math.dist(a.xyz, [b.xyz[k]-shift[k] for k in range(3)]) for a,b in zip(baseline,restored)) < 0.03
        wf.gaussian(SimpleNamespace(job=job,frame=job/'relaxed_dna.pdb',output=job/'gaussian',
                                    nproc=2,memory='2GB',time_ps=None,dispersion='none'))
        meta=json.loads((job/'gaussian'/'gaussian_preparation.json').read_text())
        assert meta['input_role']=='representative_relaxation'
        assert meta['provenance']['em_verified']
        assert (job/'gaussian'/'dna.gjf').read_text().count('\n')>379
        with (job/'relaxed_dna.pdb').open('a') as stream:
            stream.write('REMARK modified after validation\n')
        try:
            wf.gaussian(SimpleNamespace(job=job,frame=job/'relaxed_dna.pdb',output=job/'must_not_export',
                                        nproc=2,memory='2GB',time_ps=None,dispersion='none'))
        except RuntimeError:
            pass
        else:
            raise AssertionError('Stale EM/PDB export accepted')
        assert not (job/'must_not_export').exists()
        # A genuinely stretched covalent bond must still be rejected, not repaired.
        lines = original_gro.splitlines()
        last = len(reference)+1
        line = lines[last]
        lines[last] = line[:20] + f'{float(line[20:28])+0.4:8.3f}' + line[28:]
        (job/'em.gro').write_text('\n'.join(lines)+'\n', encoding='ascii')
        try:
            ext.verify_relax(SimpleNamespace(job=job,gmx=args.gmx))
        except ValueError as exc:
            assert 'covalent distance' in str(exc)
        else:
            raise AssertionError('Actual bond distortion was accepted')
        print('PASS real EM fixture, split nucleotide/strand PBC restored, distorted bond rejected, stale export rejected; no MD/EM/DFT run')


if __name__=='__main__':
    main()
