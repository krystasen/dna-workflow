"""Opt-in actual local API/WSL topology/solvation/TPR integration check; no mdrun."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8767)
    parser.add_argument('--output',type=Path,required=True)
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
    for name,mode,chain,position in (('single','single','A',2),('duplex','duplex','A',2)):
        job=args.output/name
        site={'chain':chain,'position':position,'base':'O6MeG'}
        payload=dict(sequence='AGC',mode=mode,output=str(job),md_model='ol21_opc',modifications=[site],
                     temperature=310,salt=.15,nvt_ps=100,npt_ps=1000,md_ns=100,xtc_ps=10,padding=1.2,
                     seed=42,box_type='octahedron',salt_cation='K+')
        post('/api/job/create',payload); wait()
        cfg=json.loads((job/'workflow.json').read_text())
        assert cfg['modifications']==[site]
        assert (job/'modified_base_library/O6MeG/O6MeG.lib').is_file()
        for action in ('topology','solvate','prepare_em'):
            post('/api/action',{'action':action}); wait()
        manifest=json.loads((job/'dna_manifest.json').read_text())
        modified=[a for a in manifest['atoms'] if a['residue']=='OG6']
        assert len(modified)==36 and 'CM' in {a['name'] for a in modified} and 'H1' not in {a['name'] for a in modified}
        expected_charge=-2 if mode=='single' else -4
        assert manifest['charge']==expected_charge
        assert (job/'em.tpr').is_file()
        report={'case':name,'path':str(job),'DNA_atoms':len(manifest['atoms']),
                'modified_atoms':len(modified),'DNA_charge':manifest['charge'],
                'snapshot_present':True,'topology_solvation_EM_TPR':'passed','mdrun_executed':False}
        reports.append(report)
        print(json.dumps(report),flush=True)
    (args.output/'integration_report.json').write_text(json.dumps({'finished':datetime.now().isoformat(),'cases':reports},indent=2)+'\n')


if __name__=='__main__':
    main()
