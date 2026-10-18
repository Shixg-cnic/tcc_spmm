#!/usr/bin/env python3
"""Prepare real node-classification datasets without synthetic features."""
import argparse, collections, hashlib, json, pickle, shutil, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
ROOT=Path(__file__).resolve().parent
SPECS={'ogbn-arxiv':(169343,128,40),'ogbn-products':(2449029,100,47),
       'cora':(2708,1433,7),'pubmed':(19717,500,3),'dblp':(17716,1639,4)}
class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('collections','defaultdict'):collections.defaultdict,
                 ('__builtin__','list'):list,('builtins','list'):list,
                 ('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,
                 ('scipy.sparse.csr','csr_matrix'):sp.csr_matrix,
                 ('scipy.sparse._csr','csr_matrix'):sp.csr_matrix}
        for prefix in ('numpy.core.multiarray','numpy._core.multiarray'):
            allowed[(prefix,'_reconstruct')]=np._core.multiarray._reconstruct
            allowed[(prefix,'scalar')]=np._core.multiarray.scalar
        if (module,name) not in allowed:raise pickle.UnpicklingError(f'unsupported type: {module}.{name}')
        return allowed[module,name]
def read_pickle(path):
    with path.open('rb') as f:return RestrictedUnpickler(f,encoding='latin1').load()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def fetch(url,path):
    if not path.exists():
        with urllib.request.urlopen(url,timeout=120) as response:payload=response.read()
        tmp=path.with_suffix(path.suffix+'.part');tmp.write_bytes(payload);tmp.replace(path)
def save(proc,name,array):np.save(proc/f'{name}.npy',array,allow_pickle=False)
def graph_edges(adj):
    adj=adj.maximum(adj.T).tocsr();adj.setdiag(0);adj.eliminate_zeros();adj.sum_duplicates();adj.sort_indices()
    coo=adj.tocoo();return np.stack([coo.row,coo.col]).astype(np.int64)
def finish(name,raw,proc,source,split_type,graph_policy,raw_edges):
    n,f,c=SPECS[name]
    x=np.load(proc/'x.npy',mmap_mode='r');y=np.load(proc/'y.npy',mmap_mode='r');edge=np.load(proc/'edge_index.npy',mmap_mode='r')
    assert x.shape==(n,f) and x.dtype==np.float32
    assert y.shape==(n,) and y.dtype==np.int64 and np.array_equal(np.unique(y),np.arange(c))
    assert edge.shape[0]==2 and edge.dtype==np.int64
    for start in range(0,n,10000):assert np.isfinite(x[start:start+10000]).all()
    for start in range(0,edge.shape[1],1000000):
        block=edge[:,start:start+1000000];assert block.min()>=0 and block.max()<n
    splits={key:np.load(proc/f'{key}_idx.npy') for key in ['train','valid','test']}
    for idx in splits.values():assert idx.ndim==1 and idx.dtype==np.int64 and len(idx)==len(np.unique(idx)) and idx.min()>=0 and idx.max()<n
    combined=np.concatenate(list(splits.values()));assert len(combined)==len(np.unique(combined))
    meta=dict(name=name,num_nodes=n,raw_edge_records=raw_edges,stored_edge_entries=edge.shape[1],feature_dim=f,num_classes=c,
              source=source,graph_policy=graph_policy,feature_policy='original supplied values converted to float32; no normalization, padding, truncation or synthetic values',
              split=split_type,split_sizes={key:len(idx) for key,idx in splits.items()},unassigned_nodes=n-len(combined),
              files={str(p.relative_to(raw.parent)):{'bytes':p.stat().st_size,'sha256':sha(p)} for folder in [raw,proc] for p in sorted(folder.rglob('*')) if p.is_file()})
    (raw.parent/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(name,'verified',n,edge.shape[1],f,meta['split_sizes'],flush=True)
    return meta
def planetoid(name):
    raw=ROOT/name/'raw';proc=ROOT/name/'processed';raw.mkdir(parents=True,exist_ok=True);proc.mkdir(exist_ok=True)
    base='https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/'
    parts=['x','tx','allx','y','ty','ally','graph','test.index']
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda p:fetch(base+f'ind.{name}.{p}',raw/f'ind.{name}.{p}'),parts))
    values={p:read_pickle(raw/f'ind.{name}.{p}') for p in parts[:-1]}
    test=np.loadtxt(raw/f'ind.{name}.test.index',dtype=np.int64);ordered=np.sort(test)
    x=sp.vstack([values['allx'],values['tx']]).toarray().astype(np.float32)
    x[test]=x[ordered].copy()
    labels=np.vstack([values['ally'],values['ty']]);assert (labels.sum(axis=1)==1).all()
    y=labels.argmax(axis=1).astype(np.int64);y[test]=y[ordered].copy()
    graph=values['graph'];r=[];c=[]
    for node,neighbors in graph.items():r.extend([node]*len(neighbors));c.extend(neighbors)
    n=SPECS[name][0];edge=graph_edges(sp.csr_matrix((np.ones(len(r)),(r,c)),shape=(n,n)))
    assert edge.shape[1]=={'cora':10556,'pubmed':88648}[name]
    save(proc,'x',x);save(proc,'y',y);save(proc,'edge_index',edge)
    train_count=len(values['y']);save(proc,'train_idx',np.arange(train_count,dtype=np.int64));save(proc,'valid_idx',np.arange(train_count,train_count+500,dtype=np.int64));save(proc,'test_idx',test)
    # Original shuffled test-row features and labels must align with node IDs.
    assert np.array_equal(x[test],values['tx'].toarray().astype(np.float32))
    assert np.array_equal(y[test],values['ty'].argmax(axis=1))
    return finish(name,raw,proc,base,'Planetoid public: original train/validation/test','symmetric union, remove self loops, deduplicate; no added self loops',len(r))
def dblp():
    name='dblp';raw=ROOT/name/'raw';proc=ROOT/name/'processed';raw.mkdir(parents=True,exist_ok=True);proc.mkdir(exist_ok=True)
    url='https://raw.githubusercontent.com/abojchevski/graph2gauss/master/data/dblp.npz'
    cached=ROOT.parent/'data/q4_raw/dblp.npz'
    if not (raw/'dblp.npz').exists() and cached.exists():shutil.copy2(cached,raw/'dblp.npz')
    fetch(url,raw/'dblp.npz')
    with np.load(raw/'dblp.npz',allow_pickle=False) as z:
        x=sp.csr_matrix((z['attr_data'],z['attr_indices'],z['attr_indptr']),shape=tuple(z['attr_shape'])).toarray().astype(np.float32)
        y=z['labels'].astype(np.int64);adj=sp.csr_matrix((z['adj_data'],z['adj_indices'],z['adj_indptr']),shape=tuple(z['adj_shape']))
        edge=graph_edges(adj);raw_edges=adj.nnz
    assert edge.shape[1]==105734
    save(proc,'x',x);save(proc,'y',y);save(proc,'edge_index',edge)
    rng=np.random.default_rng(42);groups=[[],[],[]]
    for label in np.unique(y):
        idx=rng.permutation(np.flatnonzero(y==label));a=int(len(idx)*.6);b=a+int(len(idx)*.2)
        for result,chunk in zip(groups,[idx[:a],idx[a:b],idx[b:]]):result.extend(chunk)
    for key,idx in zip(['train','valid','test'],groups):save(proc,key+'_idx',np.sort(np.array(idx,dtype=np.int64)))
    return finish(name,raw,proc,url,'project-generated stratified 60/20/20 per class, floor train/valid, remainder test; NumPy default_rng seed 42; NOT an official benchmark split','CitationFull homogeneous paper citation graph; symmetric union, remove self loops, deduplicate',raw_edges)
def ogb(name,source_root):
    raw=ROOT/name/'raw';proc=ROOT/name/'processed';raw.mkdir(parents=True,exist_ok=True);proc.mkdir(exist_ok=True)
    source=source_root/name.replace('-','_')
    if not (raw/'node-feat.csv.gz').exists():
        if not source.exists():raise FileNotFoundError(f'Original OGB data missing: {source}; download via OGB first')
        shutil.copytree(source/'raw',raw,dirs_exist_ok=True)
    split_name='time' if name=='ogbn-arxiv' else 'sales_ranking'
    if not (raw/'split'/split_name).exists():shutil.copytree(source/'split'/split_name,raw/'split'/split_name)
    n,f,_=SPECS[name];assert int(pd.read_csv(raw/'num-node-list.csv.gz',header=None).iloc[0,0])==n
    e=int(pd.read_csv(raw/'num-edge-list.csv.gz',header=None).iloc[0,0]);undirected=name=='ogbn-products'
    x=np.lib.format.open_memmap(proc/'x.npy',mode='w+',dtype=np.float32,shape=(n,f));offset=0
    for chunk in pd.read_csv(raw/'node-feat.csv.gz',header=None,dtype=np.float32,chunksize=20000):
        a=chunk.to_numpy();assert a.shape[1]==f and np.isfinite(a).all();x[offset:offset+len(a)]=a;offset+=len(a)
    assert offset==n;x.flush();del x
    y=pd.read_csv(raw/'node-label.csv.gz',header=None,dtype=np.int64).to_numpy().reshape(-1);save(proc,'y',y)
    edge=np.lib.format.open_memmap(proc/'edge_index.npy',mode='w+',dtype=np.int64,shape=(2,e*(2 if undirected else 1)));offset=0
    for chunk in pd.read_csv(raw/'edge.csv.gz',header=None,dtype=np.int64,chunksize=1000000):
        a=chunk.to_numpy().T;size=a.shape[1];assert a.shape[0]==2 and a.min()>=0 and a.max()<n
        edge[:,offset:offset+size]=a
        if undirected:edge[:,e+offset:e+offset+size]=a[::-1]
        offset+=size
    assert offset==e;edge.flush();del edge
    for key in ['train','valid','test']:save(proc,key+'_idx',pd.read_csv(raw/'split'/split_name/f'{key}.csv.gz',header=None,dtype=np.int64).to_numpy().reshape(-1))
    if name=='ogbn-arxiv':save(proc,'node_year',pd.read_csv(raw/'node_year.csv.gz',header=None,dtype=np.int64).to_numpy().reshape(-1))
    return finish(name,raw,proc,'https://ogb.stanford.edu/docs/nodeprop/ ; local original OGB raw CSV cache',f'official OGB {split_name}',
                  'raw undirected edges plus reverse edges, preserving original duplicates/self loops; no new self loops' if undirected else 'original directed citation edges unchanged; no reverse edges or self loops added',e)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ogb-root',type=Path,default=ROOT.parents[1]/'tcc/data');parser.add_argument('--names',nargs='+',choices=list(SPECS),default=list(SPECS));args=parser.parse_args()
    for name in args.names:
        print('Preparing',name,flush=True)
        if name.startswith('ogbn-'):ogb(name,args.ogb_root)
        elif name=='dblp':dblp()
        else:planetoid(name)
