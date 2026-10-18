"""Memory-mapped real node-classification datasets (NumPy only)."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
NAMES=('ogbn-arxiv','ogbn-products','pubmed','dblp','cora')
def load_dataset(name, mmap_mode='r'):
    """edge_index[0] is source, edge_index[1] destination; no graph transforms.

    Use mmap_mode='c' for writable copy-on-write arrays with torch.from_numpy;
    modifications then do not alter the stored dataset.
    """
    if name not in NAMES:raise ValueError(f'Choose one of {NAMES}')
    folder=ROOT/name
    metadata=json.loads((folder/'metadata.json').read_text())
    p=folder/'processed'
    result={key:np.load(p/f'{key}.npy',mmap_mode=mmap_mode,allow_pickle=False)
            for key in ('x','y','edge_index')}
    result['split_idx']={key:np.load(p/f'{key}_idx.npy',allow_pickle=False)
                         for key in ('train','valid','test')}
    result['metadata']=metadata
    if (p/'node_year.npy').exists():result['node_year']=np.load(p/'node_year.npy',mmap_mode=mmap_mode,allow_pickle=False)
    return result
