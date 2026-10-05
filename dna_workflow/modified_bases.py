"""Load the bundled modified-base library and preserve explicit chemical identity."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
LIBRARY_FILES = ('entry.json', 'O6MeG.lib', 'O6MeG.frcmod', 'leaprc.O6MeG',
                 'parameter_sources.json', 'atom_types.csv', 'nucleoside_charges.csv', 'nucleoside_B.xyz')
RESIDUE_PARENTS = {'OG6': 'DG', 'OG5': 'DG', 'OG3': 'DG', 'OGN': 'DG'}


def library_root(job: Path | None = None) -> Path:
    local = (job or HERE) / 'modified_base_library'
    return local if local.is_dir() else HERE.parent / 'modified_base_library'


def available_bases(root: Path | None = None) -> list[dict]:
    root = root or library_root()
    catalog = json.loads((root / 'catalog.json').read_text(encoding='utf-8'))
    entries = []
    for identifier, row in catalog['entries'].items():
        entry = json.loads((root / row['path']).read_text(encoding='utf-8'))
        if entry.get('MD_topology_ready'):
            entries.append({'id': identifier, 'name': entry['name'], 'parent_base': entry['parent_base'],
                            'md_models': ['ol21_opc'], 'status': entry['status'],
                            'production_MD_approved': entry.get('production_MD_approved', False),
                            'note': 'RESP + OL21/GAFF2 类比参数；技术验证通过，长期物理精度尚未验证。'})
    return entries


def validate_modifications(value, strands: list[str], model: str, root: Path | None = None) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError('modifications 必须是链、位置和碱基类型组成的列表。')
    if not value:
        return []
    entries = {row['id']: row for row in available_bases(root)}
    checked, occupied = [], set()
    for item in value:
        if not isinstance(item, dict) or item.get('base') not in entries:
            raise ValueError('修饰类型不在已完成的非标准碱基库中。')
        entry = entries[item['base']]
        if model not in entry['md_models']:
            raise ValueError(f'{item["base"]} 当前仅兼容 AMBER OL21 + OPC；不能套用 CHARMM 或 OL24。')
        chain, position = item.get('chain'), item.get('position')
        if chain not in [chr(65+i) for i in range(len(strands))]:
            raise ValueError('修饰链不存在；单链只能选择 A 链。')
        if isinstance(position, bool) or not isinstance(position, int) or not 1 <= position <= len(strands[ord(chain)-65]):
            raise ValueError('修饰位置必须是链内从5′端计数的整数，不能超出序列长度。')
        if strands[ord(chain)-65][position-1] != entry['parent_base']:
            raise ValueError(f'{chain}{position} 必须是 {entry["parent_base"]}，才能替换成 {item["base"]}。')
        if (chain, position) in occupied:
            raise ValueError('同一位点不能重复指定修饰。')
        occupied.add((chain, position))
        checked.append({'chain': chain, 'position': position, 'base': item['base']})
    return checked


def residue_code(position: int, length: int) -> str:
    return 'OGN' if length == 1 else 'OG5' if position == 1 else 'OG3' if position == length else 'OG6'


def expected_residues(strands: list[str], modifications: list[dict] | None = None) -> list[tuple[str, int, str]]:
    sites = {(r['chain'], r['position']): r['base'] for r in modifications or []}
    return [(chr(65+c), i+1, residue_code(i+1, len(s)) if (chr(65+c), i+1) in sites else 'D'+b)
            for c, s in enumerate(strands) for i, b in enumerate(s)]


def normalise_residue(name: str) -> str | None:
    import re
    name = name.strip().upper()
    if name in RESIDUE_PARENTS:
        return name
    match = re.fullmatch(r'D([ACGT])(?:[35])?', name)
    return 'D'+match[1] if match else None


def snapshot_library(job: Path, modifications: list[dict], source: Path | None = None) -> None:
    if not modifications:
        return
    source = source or library_root()
    destination = job / 'modified_base_library'
    destination.mkdir()
    catalog = json.loads((source / 'catalog.json').read_text(encoding='utf-8'))
    selected = {row['base'] for row in modifications}
    filtered = {'library': catalog['library'], 'entries': {k: catalog['entries'][k] for k in selected}}
    (destination / 'catalog.json').write_text(json.dumps(filtered, indent=2)+'\n', encoding='utf-8')
    for identifier in selected:
        folder = destination / identifier
        folder.mkdir()
        for name in LIBRARY_FILES:
            shutil.copy2(source / identifier / name, folder / name)


def load_commands(job: Path, modifications: list[dict]) -> str:
    root = library_root(job)
    lines = []
    for identifier in sorted({row['base'] for row in modifications}):
        for command, filename in (('loadoff', 'O6MeG.lib'), ('loadamberparams', 'O6MeG.frcmod')):
            target = root / identifier / filename
            if not target.is_file():
                raise FileNotFoundError(target)
            path = target.as_posix()
            if os.name == 'nt' and target.drive:
                path = '/mnt/' + target.drive[0].lower() + '/' + path[3:]
            lines.append(f'{command} "{path}"')
    return ''.join(line+'\n' for line in lines)


def apply_modifications(atoms, strands: list[str], modifications: list[dict], root: Path | None = None):
    """Rigidly place the optimized base using three named ring anchors; no numpy."""
    result = list(atoms)
    root = root or library_root()
    def subtract(a, b): return tuple(x-y for x,y in zip(a,b))
    def dot(a, b): return sum(x*y for x,y in zip(a,b))
    def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    def unit(a):
        length = math.sqrt(dot(a,a))
        if length < 1e-8: raise ValueError('修饰碱基对齐锚点退化。')
        return tuple(x/length for x in a)
    def axes(coords):
        x = unit(subtract(coords['C4'], coords['N9']))
        z = unit(cross(x, subtract(coords['C8'], coords['N9'])))
        return x, cross(z,x), z
    for site in modifications:
        folder = root / site['base']
        with (folder / 'nucleoside_charges.csv').open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        coords = [tuple(map(float, line.split()[1:])) for line in (folder / 'nucleoside_B.xyz').read_text().splitlines()[2:]]
        if len(rows) != len(coords): raise ValueError('非标准碱基库的原子映射与几何不一致。')
        source = {row['name']: point for row,point in zip(rows,coords)}
        base = [row for row in rows if not row['role'].startswith('fixed')]
        base_names = {row['name'] for row in base} | {'H1'}
        target = [a for a in result if a.chain == site['chain'] and a.resid == site['position']]
        target_coords = {a.name: a.xyz for a in target}
        a_axes, b_axes = axes(source), axes(target_coords)
        code = residue_code(site['position'], len(strands[ord(site['chain'])-65]))
        replacement = [replace(a, residue=code) for a in target if a.name not in base_names]
        for row in base:
            local = subtract(source[row['name']], source['N9'])
            components = [dot(local, axis) for axis in a_axes]
            point = tuple(target_coords['N9'][k]+sum(c*axis[k] for c,axis in zip(components,b_axes)) for k in range(3))
            replacement.append(replace(target[0], name=row['name'], element=row['element'], xyz=point, residue=code))
        result = [a for a in result if not (a.chain == site['chain'] and a.resid == site['position'])]
        # Preserve residue order; all replacement atoms stay within their site.
        result.extend(replacement)
        order = {(chain, resid): i for i,(chain,resid,_) in enumerate(expected_residues(strands,modifications))}
        result.sort(key=lambda a: order[(a.chain,a.resid)])
    return [replace(a,serial=i+1) for i,a in enumerate(result)]
