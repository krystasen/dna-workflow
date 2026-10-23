"""Build an OL21-compatible O6MeG residue library from the completed RESP fit.

Run with the existing WSL AmberTools Python; no new charge fit or QM job.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import parmed as pmd
from parmed.amber.offlib import AmberOFFLibrary

LIBRARY = Path(__file__).resolve().parent
ENTRY = LIBRARY / 'O6MeG'
QM = Path('/mnt/d/data/DNA DoS/computation/O6MeG_parameterization_20261004/gaussian')
AMBER = Path('/home/chenz/.local/share/dna-amber/env')
WORK = QM / 'topology_build'


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def run(argv, cwd, log):
    env = dict(os.environ, AMBERHOME=str(AMBER))
    result = subprocess.run([str(arg) for arg in argv], cwd=cwd, env=env,
                            capture_output=True, text=True)
    (cwd / log).write_text('$ ' + ' '.join(map(str, argv)) + '\n' + result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'Command failed: {cwd / log}')
    return result.stdout + result.stderr


def parent_templates():
    # Extract the requested parent blocks; the installed OL21 file includes
    # extra content that the general ParmEd OFF reader does not accept.
    raw = (AMBER / 'dat/leap/lib/ff-nucleic-OL21.lib').read_text()
    parents = {}
    for name in ('DG', 'DG5', 'DG3', 'DGN'):
        start = raw.index(f'!entry.{name}.unit.atoms ')
        end = re.search(r'\n!entry\.(?!' + re.escape(name) + r'\.)', raw[start:])
        block = raw[start:start+end.start()] if end else raw[start:]
        parents.update(AmberOFFLibrary.parse(io.StringIO(f'!!index array str\n "{name}"\n' + block + '\n')))
    return parents


def mol2(path, names, xyz, types, charges, elements, bonds, residue):
    lines = ['@<TRIPOS>MOLECULE', residue, f'{len(names)} {len(bonds)} 1 0 0',
             'SMALL', 'USER_CHARGES', '', '@<TRIPOS>ATOM']
    lines += [f'{i+1:5d} {name:5s} {pos[0]: .8f} {pos[1]: .8f} {pos[2]: .8f} {typ:5s} 1 {residue} {charge: .6f}'
              for i, (name, pos, typ, charge) in enumerate(zip(names, xyz, types, charges))]
    lines += ['@<TRIPOS>BOND']
    lines += [f'{k:5d} {i:5d} {j:5d} {order}' for k, (i, j, order) in enumerate(bonds, 1)]
    lines += ['@<TRIPOS>SUBSTRUCTURE', f'1 {residue} 1 RESIDUE 1 A {residue} 0 ROOT']
    path.write_text('\n'.join(lines) + '\n')


def type_candidates():
    chemistry = json.loads((QM / 'models/O6MeG_B_180/chemistry.json').read_text())
    full = read_csv(ENTRY / 'nucleoside_charges.csv')
    xyz = np.array([[float(v) for v in line.split()[1:]]
                    for line in (ENTRY / 'nucleoside_B.xyz').read_text().splitlines()[2:]])
    names = [row['name'] for row in full]
    assert names == chemistry['atom_names'] and len(names) == len(xyz) == 35
    sybyl = ['H' if row['element'] == 'H' else 'O.3' if row['element'] == 'O'
             else 'N.3' if row['name'] == 'N2' else 'N.ar' if row['element'] == 'N'
             else 'C.3' if "'" in row['name'] or row['name'] == 'CM' else 'C.ar' for row in full]
    mol2(WORK / 'nucleoside_input.mol2', names, xyz, sybyl,
         [float(row['charge_e']) for row in full], [row['element'] for row in full],
         chemistry['bonds_1based'], 'OGN')
    (WORK / 'resp_charges.txt').write_text('\n'.join(row['charge_e'] for row in full) + '\n')
    for typ in ('amber', 'gaff2'):
        run([AMBER / 'bin/antechamber', '-i', 'nucleoside_input.mol2', '-fi', 'mol2',
             '-o', f'typed_{typ}.mol2', '-fo', 'mol2', '-at', typ, '-c', 'rc',
             '-cf', 'resp_charges.txt', '-nc', '0', '-rn', 'OGN', '-an', 'n', '-du', 'n', '-seq', 'n', '-pf', 'n'],
            WORK, f'antechamber_{typ}.log')
    amber = pmd.load_file(str(WORK / 'typed_amber.mol2'))
    gaff = pmd.load_file(str(WORK / 'typed_gaff2.mol2'))
    parents = parent_templates()
    dg = {a.name: a.type for a in parents['DG'].atoms}
    print(json.dumps([{'name': a.name, 'parent_DG': dg.get(a.name), 'amber_candidate': a.type,
                       'gaff2_candidate': b.type, 'RESP_charge': a.charge} for a, b in zip(amber.atoms, gaff.atoms)]))


def make_templates():
    typed = pmd.load_file(str(WORK / 'typed_amber.mol2'))
    types = {a.name: a.type for a in typed.atoms}
    base = {row['name']: float(row['charge_e']) for row in read_csv(ENTRY / 'base_charges.csv')}
    parents = parent_templates()
    full = read_csv(ENTRY / 'nucleoside_charges.csv')
    xyz = np.array([[float(v) for v in line.split()[1:]] for line in (ENTRY/'nucleoside_B.xyz').read_text().splitlines()[2:]])
    coords = {row['name']: pos for row,pos in zip(full,xyz)}
    residues = {}
    records = []
    for source_name, new_name in (('DG','OG6'), ('DG5','OG5'), ('DG3','OG3'), ('DGN','OGN')):
        parent = parents[source_name]
        residue = parent_templates()[source_name]
        residue.name = new_name
        residue.delete_atom('H1')
        anchor = ('N9','C4','C8')
        a = np.array([coords[n] for n in anchor])
        b = np.array([[parent[n].xx,parent[n].xy,parent[n].xz] for n in anchor])
        u, _, vt = np.linalg.svd((a-a.mean(0)).T @ (b-b.mean(0)))
        fix = np.eye(3); fix[2,2] = np.linalg.det(u @ vt)
        rotation = u @ fix @ vt
        for name in ('CM','HM1','HM2','HM3'):
            atom = pmd.Atom(name=name, type=types[name], atomic_number=6 if name=='CM' else 1,
                            charge=base[name], mass=12.01 if name=='CM' else 1.008)
            residue.add_atom(atom)
        residue.add_bond('O6','CM')
        for name in ('HM1','HM2','HM3'):
            residue.add_bond('CM',name)
        for atom in residue.atoms:
            if atom.name in base:
                atom.type, atom.charge = types[atom.name], base[atom.name]
                pos = (coords[atom.name]-a.mean(0)) @ rotation + b.mean(0)
                atom.xx,atom.xy,atom.xz = pos
            # Every sugar/phosphate charge and atom type remains from the parent.
        assert len(residue.atoms) == len(parent.atoms)+3
        expected = sum(atom.charge for atom in parent.atoms)-.000001
        assert abs(sum(atom.charge for atom in residue.atoms)-expected) < 1e-6
        for atom in parent.atoms:
            if atom.name not in base and atom.name!='H1':
                assert residue[atom.name].type == atom.type and residue[atom.name].charge == atom.charge
        residues[new_name] = residue
        records.append({'residue': new_name, 'parent':source_name, 'atoms':len(residue.atoms),
                        'charge_e':sum(a.charge for a in residue.atoms),
                        'head':getattr(residue.head,'name',None), 'tail':getattr(residue.tail,'name',None)})
    AmberOFFLibrary.write(residues,str(ENTRY/'O6MeG.lib'))
    for name,residue in residues.items():
        residue.save(str(WORK/f'{name}.mol2'),format='mol2',overwrite=True)
    with (ENTRY/'atom_types.csv').open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['atom_name','selected_AMBER_type','parent_DG_type','source'])
        dgtypes={a.name:a.type for a in parents['DG'].atoms}
        writer.writerows((a.name,a.type,dgtypes.get(a.name,''),
                          'antechamber_amber_chemistry' if a.name in base else 'OL21_DG_backbone')
                         for a in residues['OG6'].atoms)
    (WORK/'template_summary.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records))
    # parmchk2's correspondence table lacks OL21-only sugar types CJ/C7.
    # Search missing modified-base terms on the typed capped nucleoside;
    # the actual polymer retains CJ/C7 and is checked separately by LEaP.
    run([AMBER/'bin/parmchk2','-i','typed_amber.mol2','-f','mol2','-o','candidate.frcmod',
         '-p',AMBER/'dat/leap/parm/parm10.dat','-afrc',AMBER/'dat/leap/parm/ff-nucleic-OL21.frcmod','-w','N'],
        WORK,'parmchk2.log')
    print((WORK/'candidate.frcmod').read_text())


def inspect_parameters():
    run([AMBER/'bin/parmchk2','-i','typed_gaff2.mol2','-f','mol2','-o','gaff2_full.frcmod',
         '-s','gaff2','-a','Y'], WORK,'parmchk2_gaff2.log')
    from parmed.amber.parameters import AmberParameterSet
    parent = AmberParameterSet(str(AMBER/'dat/leap/parm/parm10.dat'),
                               str(AMBER/'dat/leap/parm/ff-nucleic-OL21.frcmod'))
    gaff = AmberParameterSet(str(AMBER/'dat/leap/parm/gaff2.dat'))
    for key in (('N2','CA','NC'), ('N2','CA','NA'), ('NC','CA','OH'), ('NC','CA','OS'), ('NC','CA','N2'),
                ('nv','ca','nb'), ('nb','ca','os'), ('ca','os','c3'), ('ca','ca','os')):
        source = parent if key[0][0].isupper() else gaff
        value = source.angle_types.get(key)
        print('ANGLE',key, repr(value), 'k=',getattr(value,'k',None),'theta=',getattr(value,'theteq',None))
    for key in (('ca','os'),('X','ca','os','X'),('ca','ca','os','c3'),('nb','ca','os','c3'),('X','ca','nv','X')):
        values = gaff.bond_types.get(key) if len(key)==2 else gaff.dihedral_types.get(key)
        print('GAFF',key,repr(values))
        if isinstance(values,list):
            print([{'k':v.phi_k,'per':v.per,'phase':v.phase} for v in values])
    for key in ('N*','NC','N2','NB','OS','CT','H1','CA','CQ'):
        value=parent.atom_types[key]
        print('LJ',key, value.rmin,value.epsilon)


def finalize_parameters():
    from parmed.amber.parameters import AmberParameterSet
    parent = AmberParameterSet(str(AMBER/'dat/leap/parm/parm10.dat'),
                               str(AMBER/'dat/leap/parm/ff-nucleic-OL21.frcmod'))
    gaff = AmberParameterSet(str(AMBER/'dat/leap/parm/gaff2.dat'))
    sections={name:[] for name in ('MASS','BOND','ANGLE','DIHE','IMPROPER','NONBON')}
    provenance=[]
    def add(section,target,source_key,source,reason):
        params=parent if source=='parm10.dat' else gaff
        if section=='BOND':
            value=params.bond_types[source_key]
            line=f'{"-".join(target):12s} {value.k:10.4f} {value.req:10.5f}'
            values={'k_kcal_mol_A2':value.k,'req_A':value.req}
        elif section=='ANGLE':
            value=params.angle_types[source_key]
            line=f'{"-".join(target):14s} {value.k:10.4f} {value.theteq:10.5f}'
            values={'k_kcal_mol_rad2':value.k,'theta_deg':value.theteq}
        elif section=='DIHE':
            value=params.dihedral_types[source_key][0]
            line=f'{"-".join(target):18s} 1 {value.phi_k:10.5f} {value.phase:10.5f} {value.per:10.5f}'
            values={'phi_k_kcal_mol':value.phi_k,'phase_deg':value.phase,'periodicity':value.per}
        else:
            value=params.improper_periodic_types[source_key]
            line=f'{"-".join(target):18s} {value.phi_k:10.5f} {value.phase:10.5f} {value.per:10.5f}'
            values={'phi_k_kcal_mol':value.phi_k,'phase_deg':value.phase,'periodicity':value.per}
        sections[section].append(line + '  ' + source + ':' + '-'.join(source_key))
        provenance.append({'section':section,'target_types':target,'source_file':source,
                           'source_types':source_key,'values':values,'reason':reason,
                           'basis':'force-field analogy; not newly QM fitted'})
    add('BOND',('CQ','N2'),('CA','N2'),'parm10.dat','exocyclic amino bond on purine C2')
    add('BOND',('CA','OS'),('ca','os'),'gaff2.dat','aromatic carbon-ether oxygen, not carbonyl/hydroxyl')
    add('ANGLE',('N2','CQ','NC'),('N2','CA','NC'),'parm10.dat','C2 ring with exocyclic amino')
    add('ANGLE',('CQ','N2','H'),('CA','N2','H'),'parm10.dat','amino hydrogen angle')
    add('ANGLE',('NC','CA','OS'),('nb','ca','os'),'gaff2.dat','ring N-aromatic C-ether O')
    add('ANGLE',('CA','OS','CT'),('ca','os','c3'),'gaff2.dat','aryl-O-methyl ether')
    add('ANGLE',('CB','CA','OS'),('ca','ca','os'),'gaff2.dat','ring C-aromatic C-ether O')
    # Source parm10 uses IDIVF=4 and PK=9.6; ParmEd stores the actual 2.4
    # coefficient. Write IDIVF=1 to preserve the same per-dihedral energy.
    add('DIHE',('NC','CQ','N2','H'),('X','CA','N2','X'),'parm10.dat','purine amino planar conjugation')
    add('DIHE',('NC','CA','OS','CT'),('X','ca','os','X'),'gaff2.dat','aryl ether torsion generic N-side')
    add('DIHE',('CB','CA','OS','CT'),('ca','ca','os','c3'),'gaff2.dat','aryl ether torsion carbon-side specific term')
    add('IMPROPER',('N2','NC','CQ','NC'),('N2','NA','CA','NC'),'parm10.dat','C2 trigonal planarity, guanine parent')
    add('IMPROPER',('CB','NC','CA','OS'),('CB','N2','CA','NC'),'parm10.dat','C6 trigonal planarity, purine parent')
    text='O6MeG RESP + OL21 backbone; local AMBER/GAFF2 analogy terms\n'
    for name,lines in sections.items():
        text+=name+'\n'+'\n'.join(lines)+'\n\n'
    assert 'ATTN' not in text
    (ENTRY/'O6MeG.frcmod').write_text(text.rstrip()+'\n')
    (ENTRY/'parameter_sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
    # The patch must not overwrite any exact parent type combination.
    for record in provenance:
        key=tuple(record['target_types'])
        table={'BOND':parent.bond_types,'ANGLE':parent.angle_types,'DIHE':parent.dihedral_types,
               'IMPROPER':parent.improper_periodic_types}[record['section']]
        if key in table:
            raise ValueError(f'Patch would overwrite parent term {key}')
    print(json.dumps({'local_patch_terms':len(provenance),'exact_parent_terms_overwritten':0}))


def leap(script,directory,label):
    path=directory/(label+'.in')
    path.write_text(script+'\nquit\n')
    includes=[f'-I{AMBER}/dat/leap/{d}' for d in ('prep','lib','parm','cmd')]+[f'-I{ENTRY}']
    text=run([AMBER/'bin/teLeap',*includes,'-f',path.name],directory,label+'.log')
    if re.search(r'FATAL|Could not find|does not have a type|No torsion terms|Errors = [1-9]',text,re.I):
        raise RuntimeError(f'LEaP parameter or structure errors in {directory / (label+".log")}')
    return text


def validate_templates():
    validation=ENTRY/'validation'
    validation.mkdir(exist_ok=True)
    preamble='source leaprc.DNA.OL21\nsource leaprc.O6MeG\n'
    cases=('internal','fiveprime','threeprime','nucleoside')
    summary=[]
    for name in cases:
        directory=validation/name
        directory.mkdir(exist_ok=True)
        leap(preamble+'x = loadPdb input.pdb\ncheck x\ncharge x\n'
             'saveAmberParm x test.prmtop test.inpcrd\nsavePdb x test.pdb\n',directory,'build')
        topology=pmd.load_file(str(directory/'test.prmtop'),xyz=str(directory/'test.inpcrd'))
        expected=-.000001 if name=='nucleoside' else -2.000001
        assert abs(sum(a.charge for a in topology.atoms)-expected)<2e-5
        assert all(b.type is not None and b.type.k>0 for b in topology.bonds)
        assert all(a.type is not None and a.type.k>0 for a in topology.angles)
        assert all(d.type is not None for d in topology.dihedrals)
        phosphate=[]
        for b in topology.bonds:
            if b.atom1.residue!=b.atom2.residue:
                phosphate.append(sorted([b.atom1.name,b.atom2.name]))
        assert len(phosphate)==(0 if name=='nucleoside' else 2)
        assert all(pair==["O3'",'P'] for pair in phosphate)
        topology.save(str(directory/'test.top'),overwrite=True)
        topology.save(str(directory/'test.gro'),overwrite=True)
        summary.append({'case':name,'atoms':len(topology.atoms),'residues':len(topology.residues),
                        'charge_e':sum(a.charge for a in topology.atoms),
                        'phosphodiester_connections':len(phosphate),'all_bonded_terms_assigned':True})
    (validation/'template_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


def read_pdb_rows(path):
    rows=[]
    for line in path.read_text().splitlines():
        if line.startswith(('ATOM  ','HETATM')):
            name=line[12:16].strip()
            name={'O1P':'OP1','O2P':'OP2','C5M':'C7'}.get(name,name)
            rows.append({'name':name,'residue':line[17:20].strip(),
                         'chain':line[21],'resid':int(line[22:26]),
                         'xyz':np.array([float(line[s:s+8]) for s in (30,38,46)]),
                         'element':line[76:78].strip() or line[12:16].strip()[0]})
    return rows


def write_pdb_rows(path,rows):
    text=[]
    last=None
    for serial,row in enumerate(rows,1):
        if last is not None and row['chain']!=last:
            text.append('TER')
        name=row['name'] if len(row['name'])==4 else ' '+row['name'].ljust(3)
        x,y,z=row['xyz']
        text.append(f'ATOM  {serial:5d} {name:4s} {row["residue"]:3s} {row["chain"]}{row["resid"]:4d}    '
                    f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {row["element"]:>2s}  ')
        last=row['chain']
    path.write_text('\n'.join(text+['TER','END'])+'\n')


def modify_rows(rows,chain,resid):
    target=[row for row in rows if row['chain']==chain and row['resid']==resid]
    if not target or target[0]['residue'] not in ('DG','DG5','DG3','DGN'):
        raise ValueError('Selected site must be an explicit deoxyguanosine residue')
    keys=list(dict.fromkeys(row['resid'] for row in rows if row['chain']==chain))
    terminal='OGN' if len(keys)==1 else 'OG5' if resid==keys[0] else 'OG3' if resid==keys[-1] else 'OG6'
    full=read_csv(ENTRY/'nucleoside_charges.csv')
    base_names={row['name'] for row in read_csv(ENTRY/'base_charges.csv')}
    xyz=np.array([[float(v) for v in line.split()[1:]] for line in (ENTRY/'nucleoside_B.xyz').read_text().splitlines()[2:]])
    coords={row['name']:pos for row,pos in zip(full,xyz)}
    target_coords={row['name']:row['xyz'] for row in target}
    anchor=('N9','C4','C8')
    a=np.array([coords[n] for n in anchor]); b=np.array([target_coords[n] for n in anchor])
    u,_,vt=np.linalg.svd((a-a.mean(0)).T@(b-b.mean(0)))
    fix=np.eye(3); fix[2,2]=np.linalg.det(u@vt)
    rotation=u@fix@vt
    changed=[]
    for row in target:
        if row['element']=='H' or row['name'] in base_names:
            continue
        changed.append(dict(row,residue=terminal))
    for row in full:
        if row['name'] in base_names:
            changed.append({'name':row['name'],'element':row['element'],'residue':terminal,
                            'chain':chain,'resid':resid,'xyz':(coords[row['name']]-a.mean(0))@rotation+b.mean(0)})
    result=[]; inserted=False
    for row in rows:
        if row['chain']==chain and row['resid']==resid:
            if not inserted:
                result.extend(changed); inserted=True
        elif row['element']!='H':
            result.append(row)
    return result


def prepare_validation_coordinates():
    validation=ENTRY/'validation'
    source=Path('/mnt/d/data/DNA DoS/DNA rask/test1/dna_27bp/amber_input.pdb')
    rows=read_pdb_rows(source)
    chainA={r['resid']:r['residue'] for r in rows if r['chain']=='A'}
    sequence=''.join(chainA.values()).replace('D','')
    assert sequence=='GATGATGGTTGCTGGAGTTGGTGTGAA'
    for name,ids,target in (('internal',(13,14,15),14),('fiveprime',(14,15,16),14),('threeprime',(12,13,14),14)):
        directory=validation/name; directory.mkdir(parents=True,exist_ok=True)
        subset=[dict(row) for row in rows if row['chain']=='A' and row['resid'] in ids and row['element']!='H']
        subset=[r for r in subset if not (r['resid']==ids[0] and r['name'] in ('P','OP1','OP2'))]
        modified=modify_rows(subset,'A',target)
        for row in modified:
            if row['residue'].startswith('DG') or row['residue'] in ('DA','DC','DT'):
                if row['resid']==ids[0]: row['residue']+='5'
                if row['resid']==ids[-1]: row['residue']+='3'
        write_pdb_rows(directory/'input.pdb',modified)
    directory=validation/'nucleoside'; directory.mkdir(parents=True,exist_ok=True)
    full=read_csv(ENTRY/'nucleoside_charges.csv')
    xyz=np.array([[float(v) for v in line.split()[1:]] for line in (ENTRY/'nucleoside_B.xyz').read_text().splitlines()[2:]])
    write_pdb_rows(directory/'input.pdb',[{'name':row['name'],'element':row['element'],'xyz':pos,
                                        'chain':'A','resid':1,'residue':'OGN'} for row,pos in zip(full,xyz)])
    duplex=validation/'duplex27'; duplex.mkdir(parents=True,exist_ok=True)
    write_pdb_rows(duplex/'input.pdb',modify_rows(rows,'A',14))
    (validation/'coordinate_source.json').write_text(json.dumps({'source':str(source),'modified_site':'A14',
                        'reference_sequence':sequence,'method':'B-optimized base aligned to N9/C4/C8; original DNA backbone retained'},indent=2)+'\n')


def validate_systems():
    preamble='source leaprc.DNA.OL21\nsource leaprc.O6MeG\n'
    directory=ENTRY/'validation/duplex27'
    leap(preamble+'x = loadPdb input.pdb\ncheck x\ncharge x\n'
         'saveAmberParm x dna.prmtop dna.inpcrd\nsavePdb x dna.pdb\n',directory,'build')
    topology=pmd.load_file(str(directory/'dna.prmtop'),xyz=str(directory/'dna.inpcrd'))
    charge=sum(a.charge for a in topology.atoms)
    assert len(topology.residues)==54 and abs(charge+52)<2e-5
    sites=[r for r in topology.residues if r.name=='OG6']
    assert len(sites)==1 and sites[0].idx==13 and len(sites[0].atoms)==36
    cross=[b for b in topology.bonds if b.atom1.residue!=b.atom2.residue]
    assert len(cross)==52 and all(sorted([b.atom1.name,b.atom2.name])==["O3'",'P'] for b in cross)
    assert all(b.type is not None for b in topology.bonds)
    assert all(a.type is not None for a in topology.angles)
    assert all(d.type is not None for d in topology.dihedrals)
    topology.save(str(directory/'dna.top'),overwrite=True)
    topology.save(str(directory/'dna.gro'),overwrite=True)
    report={'case':'27bp duplex A14 O6MeG:C','atoms':len(topology.atoms),'residues':54,
            'modified_sites':1,'charge_e':charge,'phosphodiester_connections':52,
            'all_parameters_assigned':True,'scope':'dry topology, not production trajectory'}
    (directory/'system_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    directory=ENTRY/'validation/opc_trimer'
    directory.mkdir(exist_ok=True)
    shutil.copy2(ENTRY/'validation/internal/input.pdb',directory/'input.pdb')
    leap(preamble+'source leaprc.water.opc\nx = loadPdb input.pdb\n'
         'solvateOct x OPCBOX 8.0\naddIonsRand x K+ 0\ncheck x\ncharge x\n'
         'saveAmberParm x solvated.prmtop solvated.inpcrd\nsavePdb x solvated.pdb\n',directory,'build')
    solvated=pmd.load_file(str(directory/'solvated.prmtop'),xyz=str(directory/'solvated.inpcrd'))
    waters=[r for r in solvated.residues if r.name=='WAT']
    assert waters and all(len(r.atoms)==4 for r in waters)
    assert abs(sum(a.charge for a in solvated.atoms))<2e-5
    solvated.save(str(directory/'solvated.top'),overwrite=True)
    solvated.save(str(directory/'solvated.gro'),overwrite=True)
    (directory/'em.mdp').write_text('integrator = steep\nnsteps = 25\nemtol = 1000\nemstep = 0.001\n'
        'pbc = xyz\ncutoff-scheme = Verlet\nnstlist = 20\ncoulombtype = PME\nrcoulomb = 1.0\n'
        'rvdw = 1.0\nvdwtype = Cut-off\nvdw-modifier = Potential-shift\nDispCorr = EnerPres\n'
        'constraints = none\nnstenergy = 1\nnstlog = 1\n')
    run(['gmx','grompp','-f','em.mdp','-c','solvated.gro','-p','solvated.top','-o','em.tpr'],directory,'grompp.log')
    run(['gmx','mdrun','-deffnm','em','-ntmpi','1','-ntomp','2','-nb','cpu','-pme','cpu'],directory,'gromacs_em.console.txt')
    log=(directory/'em.log').read_text()
    if not re.search(r'Potential Energy\s*=\s*[-+0-9.eE]+',log):
        raise ValueError('OPC minimization did not report finite potential energy')
    energy=float(re.findall(r'Potential Energy\s*=\s*([-+0-9.eE]+)',log)[-1])
    assert np.isfinite(energy)
    exported=pmd.load_file(str(directory/'solvated.top'),xyz=str(directory/'solvated.gro'))
    assert len(exported.atoms)==len(solvated.atoms)
    report={'case':'O6MeG trimer in OPC and neutralizing K+','atoms':len(solvated.atoms),
            'OPC_waters':len(waters),'charge_e':sum(a.charge for a in solvated.atoms),
            'GROMACS_tpr_built':True,'GROMACS_25step_EM_completed':True,
            'EM_final_potential_kJ_mol':energy,'scope':'compatibility smoke test; not 0.15 M equilibration or MD validation'}
    (directory/'system_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('types','templates','parameters','finalize','coordinates','validate','systems'), default='types')
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    if args.stage == 'types':
        type_candidates()
    elif args.stage == 'templates':
        make_templates()
    elif args.stage == 'parameters':
        inspect_parameters()
    elif args.stage == 'finalize':
        finalize_parameters()
    elif args.stage == 'coordinates':
        prepare_validation_coordinates()
    elif args.stage == 'validate':
        validate_templates()
    else:
        validate_systems()


if __name__ == '__main__':
    main()
