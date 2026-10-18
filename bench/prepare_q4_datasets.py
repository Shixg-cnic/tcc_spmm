#!/usr/bin/env python3
"""Download graph topology only; retain hashes and exact graph transformations."""
import collections, hashlib, io, json, pickle, struct, urllib.request
from pathlib import Path
import numpy as np
import scipy.sparse as sp
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RAW = DATA / 'q4_raw'
RAW.mkdir(parents=True, exist_ok=True)
class GraphUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module,name)==('collections','defaultdict'):return collections.defaultdict
        if module in ('__builtin__','builtins') and name=='list':return list
        raise pickle.UnpicklingError('graph must contain only builtin containers and integers')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
def fetch(url, name):
    path=RAW/name
    if not path.exists():
        with urllib.request.urlopen(url,timeout=120) as response:path.write_bytes(response.read())
    return path
EXPECTED_RAW={
    'cora':'58f13302f39dde8852dad6fe6d15b89b077b6d7f837626bce671ef80344b383d',
    'pubmed':'5b89f0036ca22909471f1e6558eb42fa06e0c730a577bde5b058b40638f81dc9',
    'dblp':'46da600a15c0680d967b3c9ac83979e3c44dfe5a0ef86e1c42a94d8dee1c1da4',
}
records=[]
for name,n in [('cora',2708),('pubmed',19717),('dblp',17716)]:
    if name=='dblp':
        url='https://raw.githubusercontent.com/abojchevski/graph2gauss/master/data/dblp.npz'
        raw=fetch(url,'dblp.npz')
        assert sha(raw)==EXPECTED_RAW[name], 'upstream file changed or corrupt download'
        with np.load(raw,allow_pickle=False) as z:
            a=sp.csr_matrix((z['adj_data'],z['adj_indices'],z['adj_indptr']),shape=tuple(z['adj_shape']))
    else:
        url=f'https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.{name}.graph'
        raw=fetch(url,f'ind.{name}.graph')
        assert sha(raw)==EXPECTED_RAW[name], 'upstream file changed or corrupt download'
        graph=GraphUnpickler(io.BytesIO(raw.read_bytes())).load()
        assert isinstance(graph,dict) and set(graph)==set(range(n))
        rows=[];cols=[]
        for row,neighbors in graph.items():
            assert isinstance(neighbors,list) and all(type(c)==int and 0<=c<n for c in neighbors)
            rows.extend([row]*len(neighbors));cols.extend(neighbors)
        a=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n))
    assert a.shape==(n,n)
    a=a.maximum(a.T).tocsr();a.setdiag(0);a.eliminate_zeros();a.sum_duplicates();a.sort_indices();a.data[:]=1
    expected={'cora':10556,'pubmed':88648,'dblp':105734}[name]
    assert a.nnz==expected,(name,a.nnz)
    path=DATA/f'{name}_full.csr'
    with path.open('wb') as f:
        f.write(b'TCCCSR1\0');f.write(struct.pack('<III',n,n,a.nnz))
        f.write(a.indptr.astype('<u4').tobytes());f.write(a.indices.astype('<u4').tobytes());f.write(a.data.astype('<f4').tobytes())
    records.append(dict(name=name,source=url,raw_sha256=sha(raw),csr_sha256=sha(path),rows=n,cols=n,nnz=a.nnz,transform='symmetric union, remove self loops, deduplicate, unit weights, sorted CSR'))
    print(name,n,a.nnz,flush=True)
for name in ['arxiv','products']:
    path=DATA/f'ogbn_{name}_full.csr'
    with path.open('rb') as f:
        assert f.read(8)==b'TCCCSR1\0';r,c,nnz=struct.unpack('<III',f.read(12))
    records.append(dict(name='ogbn-'+name,source='existing project full CSR, unchanged',csr_sha256=sha(path),rows=r,cols=c,nnz=nnz,transform='no changes in this experiment'))
(ROOT/'results/q4_datasets/datasets.json').write_text(json.dumps(records,indent=2)+'\n')
