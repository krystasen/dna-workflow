"""Replace one explicit DG site by O6MeG in a dry DNA PDB, preserving its backbone."""
import argparse
from pathlib import Path

from build_o6meg import read_pdb_rows, modify_rows, write_pdb_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--chain',required=True)
    parser.add_argument('--resid',type=int,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserve existing PDB: {args.output}')
    rows=read_pdb_rows(args.input)
    canonical={'DG','DG5','DG3','DGN','DA','DA5','DA3','DAN','DC','DC5','DC3','DCN','DT','DT5','DT3','DTN'}
    unknown={r['residue'] for r in rows}-canonical
    if unknown:
        raise ValueError(f'Use a dry canonical DNA PDB; unexpected residues: {sorted(unknown)}')
    changed=modify_rows(rows,args.chain,args.resid)
    write_pdb_rows(args.output,changed)
    print(f'Prepared {args.output}; modified {args.chain}:{args.resid}; explicit terminal template selected')


if __name__=='__main__':
    main()
