#!/usr/bin/env python3
"""Wait for an unoccupied GPU; abort/retry our own case if another job arrives.
Never signal other users' processes or change GPU/driver settings.
"""
import csv,json,os,subprocess,time,datetime,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/acc_dequant_ogb'
os.chdir(ROOT);OUT.mkdir(exist_ok=True)
STATE=OUT/'run_state.json'
def stamp():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def query(args):
 p=subprocess.run(['nvidia-smi',*args],text=True,capture_output=True,check=True)
 return list(csv.reader(p.stdout.splitlines(),skipinitialspace=True))
def gpus():return {int(r[0]):r[1] for r in query(['--query-gpu=index,uuid','--format=csv,noheader'])}
def compute():
 return [(r[0],int(r[1]),r[2]) for r in query(['--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader']) if len(r)>=3]
def save(state,**kw):
 state.update(kw);state['updated']=stamp();tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
def run():
 devices=gpus();state={'status':'waiting_for_idle_gpu','completed':[],'policy':'No other compute process on selected GPU. Poll every 2s during a case; abort and discard a contaminated case. No exclusive reservation against other users.'};save(state)
 chosen=None
 while chosen is None:
  occupied={r[0] for r in compute()}
  for device in sorted(devices,reverse=True):
   if devices[device] not in occupied:chosen=device;break
  if chosen is None:
   print(stamp(),'waiting; all GPUs have other compute processes',flush=True);save(state,occupants=compute());time.sleep(30)
 env=os.environ.copy();env['TMPDIR']=str(ROOT/'build/tmp')
 subprocess.run(['nvidia-smi'],stdout=(OUT/'environment.txt').open('w'),check=True)
 with (OUT/'environment.txt').open('a') as f:
  f.write(f'\nSelected GPU: {chosen}; launch time: {stamp()}\n')
  subprocess.run(['/home/shixingguo/miniforge3/envs/gnn/bin/nvcc','--version'],stdout=f,check=True)
 save(state,status='running',device=chosen,device_uuid=devices[chosen])
 records=json.loads((OUT/'inputs.json').read_text())
 with (OUT/'exclusive_monitor.jsonl').open('a') as monitor:
  for record in records:
   name=record['name'].removeprefix('ogbn-')
   for width in [64,128,256,512]:
    case=f'{name}_f{width}';prefix=OUT/case
    while True:
     foreign=[p for p in compute() if p[0]==devices[chosen]]
     if foreign:
      save(state,status='waiting_for_idle_gpu',case=case,occupants=foreign)
      print(stamp(),'waiting for GPU',chosen,case,flush=True);time.sleep(30);continue
     subprocess.run(['nvidia-smi'],stdout=Path(str(prefix)+'_gpu_before.txt').open('w'),check=True)
     command=[str(ROOT/'build/benchmark_acc_dequant'),record['csr'],record['x'],str(width),str(chosen),str(prefix)]
     print(stamp(),'running',case,'GPU',chosen,flush=True);save(state,status='running',case=case,occupants=[])
     with Path(str(prefix)+'.stdout.log').open('w') as stdout,Path(str(prefix)+'.stderr.log').open('w') as stderr:
      child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env)
      contaminated=False
      try:
       while child.poll() is None:
        processes=compute();foreign=[p for p in processes if p[0]==devices[chosen] and p[1]!=child.pid]
        monitor.write(json.dumps(dict(time=stamp(),case=case,our_pid=child.pid,processes=processes,foreign=foreign))+'\n');monitor.flush()
        if foreign:
         contaminated=True;child.terminate();child.wait(timeout=30);break
        time.sleep(2)
      except BaseException:
       if child.poll() is None:child.terminate();child.wait(timeout=30)
       raise
     contaminated = contaminated or any(p[0]==devices[chosen] and p[1]!=child.pid for p in compute())
     subprocess.run(['nvidia-smi'],stdout=Path(str(prefix)+'_gpu_after.txt').open('w'),check=True)
     if contaminated:
      archive=OUT/'discarded_contention'/f'{case}_{time.time_ns()}';archive.mkdir(parents=True)
      for p in OUT.glob(case+'*'):shutil.move(str(p),archive/p.name)
      print(stamp(),'discarded case due to concurrent GPU process; will retry',flush=True);continue
     if child.returncode:raise RuntimeError(f'{case} failed; see {case}.stderr.log')
     checks=list(csv.DictReader(Path(str(prefix)+'_validation.csv').open()))
     if len(checks)!=9 or any(int(r['sampled_bad']) or int(r['full_vs_xhat_bad']) for r in checks):raise RuntimeError(f'{case} validation failed')
     state['completed'].append(case);save(state);break
 subprocess.run(['python3','bench/summarize_acc_ogb.py'],check=True)
 save(state,status='complete',case=None)
 print(stamp(),'all 12 configurations completed and validated',flush=True)
if __name__=='__main__':
 try:run()
 except Exception as e:
  state=json.loads(STATE.read_text()) if STATE.exists() else {};save(state,status='failed',error=str(e));raise
