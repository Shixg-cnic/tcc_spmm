#!/usr/bin/env python3
"""Prepare ogbn-proteins for scalar AX; retain existing arxiv/products inputs."""
import hashlib,json,struct,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/ogb_extended';OUT=ROOT/'results/acc_dequant_ogb'
DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def header(p):
 with p.open('rb') as f:
  assert f.read(8)==b'TCCCSR1\0'
  return struct.unpack('<III',f.read(12))
records=[]
for name,width in [('arxiv',128),('products',100)]:
 p=ROOT/f'data/ogbn_{name}_full.csr';x=ROOT/f'data/native_features/ogbn-{name}.x.bin';n,k,e=header(p)
 records.append(dict(name='ogbn-'+name,nodes=n,nnz=e,native_F=width,csr=str(p.relative_to(ROOT)),x=str(x.relative_to(ROOT)),
  graph_policy='Reuse earlier symmetric/self-loop CSR without changes; no node reordering',
  feature_policy='Original node feature values; first F columns if F<native_F, cyclic repeat otherwise',csr_sha256=sha(p),x_sha256=sha(x)))
archive=DATA/'raw/proteins.zip'
assert archive.exists(),'Download https://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip first'
with zipfile.ZipFile(archive) as z:
 for info in z.infolist():
  target=(DATA/'raw'/info.filename).resolve()
  assert target.is_relative_to((DATA/'raw').resolve())
  if not target.exists() or target.stat().st_size!=info.file_size:z.extract(info,DATA/'raw')
raw=DATA/'raw/proteins/raw'
n=int(pd.read_csv(raw/'num-node-list.csv.gz',header=None).iloc[0,0]);e=int(pd.read_csv(raw/'num-edge-list.csv.gz',header=None).iloc[0,0])
assert n==132534 and e==39561252
# Original undirected edges; derive 8 node features by mean incident edge features.
src=np.empty(e,np.uint32);dst=np.empty(e,np.uint32)
sums=np.zeros((n,8),np.float64);counts=np.zeros(n,np.int64);offset=0
edges=pd.read_csv(raw/'edge.csv.gz',header=None,dtype=np.uint32,chunksize=1000000)
features=pd.read_csv(raw/'edge-feat.csv.gz',header=None,dtype=np.float32,chunksize=1000000)
for ec,fc in zip(edges,features):
 a=ec.to_numpy();v=fc.to_numpy();assert len(a)==len(v) and v.shape[1]==8 and a.max()<n and np.isfinite(v).all()
 size=len(a);src[offset:offset+size]=a[:,0];dst[offset:offset+size]=a[:,1]
 for endpoint in [a[:,0],a[:,1]]:
  counts+=np.bincount(endpoint,minlength=n)
  for f in range(8):sums[:,f]+=np.bincount(endpoint,weights=v[:,f],minlength=n)
 offset+=size;print('proteins edge/features processed',offset,'/',e,flush=True)
assert offset==e
x=np.divide(sums,counts[:,None],out=np.zeros_like(sums),where=counts[:,None]!=0).astype(np.float32)
x.tofile(DATA/'ogbn-proteins.x.bin')
# Unit-weight adjacency for this scalar SpMM; edge attributes only construct X.
a=sp.coo_matrix((np.ones(e,np.float32),(dst,src)),shape=(n,n)).tocsr()
a=a.maximum(a.T);a.setdiag(1);a.sum_duplicates();a.sort_indices();a.data.fill(1)
p=DATA/'ogbn-proteins.csr'
with p.open('wb') as f:
 f.write(b'TCCCSR1\0'+struct.pack('<III',n,n,a.nnz))
 for v in [a.indptr.astype('<u4'),a.indices.astype('<u4'),a.data.astype('<f4')]:f.write(v.tobytes())
records.append(dict(name='ogbn-proteins',nodes=n,raw_undirected_edges=e,nnz=a.nnz,native_F=8,
 csr=str(p.relative_to(ROOT)),x='data/ogb_extended/ogbn-proteins.x.bin',
 graph_policy='Undirected union, deduplicate, add unit self-loops, all A values=1; no node reordering',
 feature_policy='Derived, NOT provided native node features: 8D mean of incident edge features; duplicate both directions before mean, so raw self-edges count twice; isolated nodes zero; cyclic repeat to F',
 source='https://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip',archive_sha256=sha(archive),csr_sha256=sha(p),x_sha256=sha(DATA/'ogbn-proteins.x.bin')))
(OUT/'inputs.json').write_text(json.dumps(records,indent=2)+'\n')
print('prepared',[(r['name'],r['nodes'],r['nnz']) for r in records],flush=True)
