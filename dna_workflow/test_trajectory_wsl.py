"""Bounded real-GROMACS regression: shift chain B by a box vector, then process one frame.

Run in WSL with --source pointing at the existing six-bp short_ds_pass fixture.
Only temporary copies and trjconv/convert-tpr are used; no mdrun, no original writes.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

import dna_workflow as wf
import md_pipeline as md


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--gmx',required=True)
    args=parser.parse_args()
    source=args.source.resolve()
    cfg=json.loads((source/'workflow.json').read_text())
    if len(cfg['strands_5to3'])!=2:
        raise ValueError('Use a two-strand fixture')
    topology=wf.restore_identity(wf.read_pdb(source/'topology.pdb'),cfg['strands_5to3'])
    with tempfile.TemporaryDirectory(prefix='dna_pbc_regression_') as tmp:
        job=Path(tmp)
        for name in ('workflow.json','md.tpr','topology.pdb'):
            shutil.copyfile(source/name,job/name)
        shutil.copytree(source/cfg['forcefield'],job/cfg['forcefield'])
        lines=(source/'md.gro').read_text().splitlines()
        box=list(map(float,lines[-1].split()))
        box += [0.]*(9-len(box))
        vector=(box[0],box[3],box[4])
        native=[]
        for i,a in enumerate(topology):
            line=lines[i+2]
            xyz=tuple(float(line[p:p+8]) for p in (20,28,36))
            native.append(xyz)
            if a.chain=='B':
                moved=[x+v for x,v in zip(xyz,vector)]
                lines[i+2]=line[:20]+''.join(f'{v:8.3f}' for v in moved)+line[44:]
        (job/'split.gro').write_text('\n'.join(lines)+'\n')
        subprocess.run([args.gmx,'trjconv','-s','md.tpr','-f','split.gro','-o','md.xtc'],
                       cwd=job,input='0\n',text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        md.run(SimpleNamespace(job=job,gmx=args.gmx,stage='handoff'))
        result=wf.read_pdb(job/'dna.pdb')
        assert len(result)==len(topology)
        assert {a.residue for a in result} <= wf.BASES
        assert {a.chain for a in result}=={'A','B'}
        # GROMACS center_x centers the selection's bounding box, not its COM/centroid.
        center=[(min(a.xyz[d] for a in result)+max(a.xyz[d] for a in result))/20 for d in range(3)]
        expected=[(box[0]+box[5]+box[7])/2,(box[3]+box[1]+box[8])/2,(box[4]+box[6]+box[2])/2]
        error=math.dist(center,expected)
        assert error<0.003,(center,expected,error)
        # Paired sugar distances must recover the native duplex, not the displaced image.
        count=len(cfg['strands_5to3'][0])
        sugar={(a.chain,a.resid):i for i,a in enumerate(topology) if a.name=="C1'"}
        differences=[]
        for pos in range(1,count+1):
            i,j=sugar['A',pos],sugar['B',count+1-pos]
            native_distance=math.dist(native[i],native[j])
            output_distance=math.dist(result[i].xyz,result[j].xyz)/10
            differences.append(abs(native_distance-output_distance))
        assert max(differences)<0.004,differences
        check=subprocess.run([args.gmx,'check','-f','dna_whole.xtc'],cwd=job,text=True,capture_output=True,check=True)
        assert (job/'dna.tpr').stat().st_size>0
        assert (job/'md.xtc').stat().st_size>0
        print(json.dumps({'result':'PASS','atoms':len(result),'center_error_nm':error,
                          'max_pair_distance_error_nm':max(differences),'gmx_check_exit':check.returncode,
                          'simulation_run':False}))


if __name__=='__main__':
    main()
