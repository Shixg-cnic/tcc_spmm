#!/usr/bin/env python3
"""Validate file hashes, original OGB rows and small PyTorch message-passing steps."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from load import load_dataset,NAMES,ROOT
from prepare import sha
logs=[]
for name in NAMES:
    data=load_dataset(name,mmap_mode='c');meta=data['metadata'];folder=ROOT/name
    for relative,record in meta['files'].items():
        path=folder/relative
        assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],path
    x=data['x'];y=data['y'];edge=data['edge_index'];n,f=x.shape
    if name.startswith('ogbn-'):
        raw=folder/'raw'
        original=pd.read_csv(raw/'node-feat.csv.gz',header=None,nrows=32,dtype=np.float32).to_numpy()
        assert np.array_equal(x[:32],original)
        original_y=pd.read_csv(raw/'node-label.csv.gz',header=None,dtype=np.int64).to_numpy().reshape(-1)
        assert np.array_equal(y,original_y)
        original_edges=pd.read_csv(raw/'edge.csv.gz',header=None,nrows=1024,dtype=np.int64).to_numpy().T
        assert np.array_equal(edge[:,:1024],original_edges)
        if name=='ogbn-products':
            e=meta['raw_edge_records']
            assert edge.shape[1]==2*e
            for start in range(0,e,1000000):assert np.array_equal(edge[:,e+start:e+min(start+1000000,e)],edge[::-1,start:min(start+1000000,e)])
    # Exercise actual edges, features and labels in a bounded GNN computation;
    # this is a load/forward/backward smoke test, not training or accuracy eval.
    train=data['split_idx']['train'][:16]
    mask=np.isin(edge[1,:100000],train)
    sample=edge[:,:100000][:,mask][:,:512]
    nodes=np.unique(np.concatenate([train,sample.reshape(-1)]))
    local=np.searchsorted(nodes,sample)
    tx=torch.from_numpy(np.array(x[nodes],copy=True));ty=torch.from_numpy(np.array(y[nodes],copy=True))
    aggregated=tx.clone() # model-level self term; stored graph unchanged
    if local.shape[1]:aggregated.index_add_(0,torch.from_numpy(local[1]),tx[torch.from_numpy(local[0])])
    torch.manual_seed(42);model=torch.nn.Linear(f,meta['num_classes'])
    idx=torch.from_numpy(np.searchsorted(nodes,train));loss=torch.nn.functional.cross_entropy(model(aggregated)[idx],ty[idx])
    assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    logs.append(dict(name=name,hashes_verified=len(meta['files']),nodes=n,stored_edges=edge.shape[1],feature_dim=f,split_sizes=meta['split_sizes'],message_edges=int(local.shape[1]),smoke_loss=float(loss.detach()),passed=True))
    print(name,'hashes, real data and message-passing backward PASS',flush=True)
(ROOT/'validation.json').write_text(json.dumps(logs,indent=2)+'\n')
