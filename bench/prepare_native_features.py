#!/usr/bin/env python3
"""Export AX CSR (row=destination) and exact original-width features."""
import argparse,json,struct,sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataset.load import load_dataset,NAMES
out=ROOT/'data/native_features';out.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('--names',nargs='+',choices=NAMES,default=['pubmed']);args=parser.parse_args()
records=[]
for name in args.names:
    print('exporting',name,flush=True);d=load_dataset(name);x=d['x'];edge=d['edge_index'];n,f=x.shape
    a=sp.coo_matrix((np.ones(edge.shape[1],dtype=np.float32),(edge[1],edge[0])),shape=(n,n)).tocsr()
    a.sum_duplicates();a.sort_indices()
    assert int(a.data.sum(dtype=np.float64))==edge.shape[1]
    with (out/f'{name}.csr').open('wb') as stream:
        stream.write(b'TCCCSR1\0'+struct.pack('<III',n,n,a.nnz))
        stream.write(a.indptr.astype('<u4').tobytes());stream.write(a.indices.astype('<u4').tobytes());stream.write(a.data.astype('<f4').tobytes())
    x.tofile(out/f'{name}.x.bin')
    assert (out/f'{name}.x.bin').stat().st_size==n*f*4
    records.append(dict(name=name,nodes=n,F=f,stored_edges=edge.shape[1],csr_nnz=a.nnz,weighted_entries=int(np.count_nonzero(a.data!=1)),graph='AX: A[dst,src]; duplicate edges summed, no added/reversed/removed edges',source_files=d['metadata']['files']))
(ROOT/'results/native_features/inputs.json').write_text(json.dumps(records,indent=2)+'\n')
