"""Opt-in actual local API/WSL topology/solvation/TPR integration check; no mdrun."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
import modified_bases as mods


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8767)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base',choices=list(mods.library_entries()),default='O6MeG')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    url=f'http://127.0.0.1:{args.port}'
    def get():
        with urlopen(url+'/api/state',timeout=10) as response:
            return json.load(response)
    def post(path,payload):
        request=Request(url+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Origin':url})
        with urlopen(request,timeout=20) as response:
            return json.load(response)
    def wait():
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            state=get()
            if state['task'] and state['task']['status']!='running':
                if state['task']['status']!='succeeded':
                    raise RuntimeError(json.dumps(state['task'],ensure_ascii=False))
                return state
            time.sleep(.25)
        raise TimeoutError('Local workflow task did not finish within three minutes')
    reports=[]
    entry=mods.library_entries()[args.base]
    sequence='A'+entry['parent_base']+'C'
    residue=mods.residue_code(2,3,args.base)
    for name,mode,chain,position in (('single','single','A',2),('duplex','duplex','A',2)):
        job=args.output/name
        site={'chain':chain,'position':position,'base':args.base}
        payload=dict(sequence=sequence,mode=mode,output=str(job),md_model='ol21_opc',modifications=[site],
                     temperature=310,salt=.15,nvt_ps=100,npt_ps=1000,md_ns=100,xtc_ps=10,padding=1.2,
                     seed=42,box_type='octahedron',salt_cation='K+')
        post('/api/job/create',payload); wait()
        cfg=json.loads((job/'workflow.json').read_text())
        assert cfg['modifications']==[site]
        assert (job/'modified_base_library'/args.base/(args.base+'.lib')).is_file()
        for action in ('topology','solvate','prepare_em'):
            post('/api/action',{'action':action}); wait()
        manifest=json.loads((job/'dna_manifest.json').read_text())
        modified=[a for a in manifest['atoms'] if a['residue']==residue]
        template=next(row for row in entry['templates'] if row['name']==residue)
        assert len(modified)==template['atoms']
        expected_charge=(-2 if mode=='single' else -4)+entry.get('chemical_state',{}).get('nucleoside_charge',0)
        assert manifest['charge']==expected_charge
        assert (job/'em.tpr').is_file()
        gaussian=job/'gaussian_input'
        post('/api/action',{'action':'gaussian','frame':str(job/'topology.pdb'),
                            'output':str(gaussian),'nproc':4,'memory':'8GB'})
        wait()
        report_gaussian=json.loads((gaussian/'gaussian_preparation.json').read_text())
        assert report_gaussian['charge']==expected_charge
        if args.base=='Cl5dC':
            assert any(line.startswith('Cl ') for line in (gaussian/'dna.gjf').read_text().splitlines())
        report={'base':args.base,'case':name,'path':str(job),'DNA_atoms':len(manifest['atoms']),
                'modified_atoms':len(modified),'DNA_charge':manifest['charge'],
                'snapshot_present':True,'topology_solvation_EM_TPR':'passed',
                'Gaussian_charge':report_gaussian['charge'],'Gaussian_input':'passed','mdrun_executed':False}
        reports.append(report)
        print(json.dumps(report),flush=True)
    (args.output/'integration_report.json').write_text(json.dumps({'finished':datetime.now().isoformat(),'cases':reports},indent=2)+'\n')


if __name__=='__main__':
    main()
