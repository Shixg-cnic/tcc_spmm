#!/usr/bin/env python3
"""CSR thumbnails, adapting /home/shixingguo/work/GNN/unitls/pic_heatmap.py.
Same coordinate-bin nnz counting, palette and nearest-neighbor rendering.
Replace its per-edge Python loop/MTX reader with chunked CSR bincount.
"""
import argparse,json,struct,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,LogNorm
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/acc_dequant_ogb/figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'savefig.facecolor':'white'})
def aggregate(path,res):
 with path.open('rb') as f:
  assert f.read(8)==b'TCCCSR1\0';n,k,e=struct.unpack('<III',f.read(12))
 ptr=np.memmap(path,dtype='<u4',mode='r',offset=20,shape=(n+1,))
 idx=np.memmap(path,dtype='<u4',mode='r',offset=20+4*(n+1),shape=(e,))
 vals=np.memmap(path,dtype='<f4',mode='r',offset=20+4*(n+1+e),shape=(e,))
 res=min(res,n,k);h=np.zeros((res,res),np.uint64)
 for r0 in range(0,n,4096):
  r1=min(n,r0+4096);a,b=int(ptr[r0]),int(ptr[r1]);assert np.all(vals[a:b]!=0)
  rb=np.repeat(np.arange(r0,r1,dtype=np.int64)*res//n,np.diff(ptr[r0:r1+1]).astype(np.int64))
  cb=idx[a:b].astype(np.int64)*res//k
  h+=np.bincount(rb*res+cb,minlength=res*res).reshape(res,res).astype(np.uint64)
 assert int(h.sum())==e
 return h,n,k,e
parser=argparse.ArgumentParser();parser.add_argument('--resolution',type=int,default=768);args=parser.parse_args()
records=json.loads((ROOT/'results/acc_dequant_ogb/inputs.json').read_text())
all_data=[(r,*aggregate(ROOT/r['csr'],args.resolution)) for r in records]
colors=[(1,1,1),(.7,.85,1),(.3,.95,1),(1,.5,0),(1,0,0),(0,0,0)]
cmap=LinearSegmentedColormap.from_list('user_heatmap',colors,N=256);cmap.set_bad('white')
vmax=max(int(h.max()) for _,h,_,_,_ in all_data);norm=LogNorm(vmin=1,vmax=max(2,vmax))
metadata=[]
def draw(ax,r,h,n,k,e):
 im=ax.imshow(np.ma.masked_equal(h,0),cmap=cmap,norm=norm,origin='upper',extent=(0,k,n,0),interpolation='nearest',aspect='equal')
 ax.set_title(f"{r['name']}\nN={n:,}; nnz={e:,}",fontsize=11)
 ax.set_xlabel('Source node / column');ax.set_ylabel('Destination node / row')
 fmt=FuncFormatter(lambda value,pos:f'{value/1e6:g}M' if n>=1000000 else f'{value/1000:g}k')
 ax.xaxis.set_major_formatter(fmt);ax.yaxis.set_major_formatter(fmt)
 return im
for r,h,n,k,e in all_data:
 fig,ax=plt.subplots(figsize=(6.4,5.7),layout='constrained');im=draw(ax,r,h,n,k,e)
 fig.colorbar(im,ax=ax,shrink=.8,label='Nonzeros per bin (log scale)')
 for extension in ['png','pdf']:fig.savefig(OUT/f"{r['name']}.{extension}",dpi=200)
 plt.close(fig);np.save(OUT/f"{r['name']}_counts.npy",h)
 metadata.append(dict(name=r['name'],shape=[n,k],nnz=e,bins=list(h.shape),sum=int(h.sum()),max_bin=int(h.max()),vmin=1,vmax=vmax,source=r['csr']))
fig,axes=plt.subplots(1,3,figsize=(16.5,5.8),layout='constrained')
for ax,(r,h,n,k,e) in zip(axes,all_data):im=draw(ax,r,h,n,k,e)
fig.colorbar(im,ax=axes.tolist(),shrink=.65,label='Nonzeros per bin (shared log scale)')
for extension in ['png','pdf']:fig.savefig(OUT/f'ogbn_A_overview.{extension}',dpi=200)
plt.close(fig)
(OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Saved thumbnails; every bin-count sum matches CSR nnz.',flush=True)
