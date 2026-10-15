# Calibrated cost model for the Column pipeline (task 3.6).
#
# Structure: per-task estimate t_s = a*W_s*FB + d*N_CW,s + e_s, plus a global
# throughput-shared atomic term b*A and a per-SW cache-machinery term g*H_s:
#   t_pred = a*sum(W_s)*FB/P + g*H_total + d*N_total/P + b*A_total + base
#   where the serial critical path is captured by the tail feature
#   c*max(W_s)*FB (the longest single task cannot be hidden by pipelining).
#
# Calibration: b is FIXED from the no-atomic probe (graph-specific: arxiv's
# Y (87 MB) is L2-resident, products' Y (1.25 GB) is DRAM-backed, so the
# atomic throughput differs). The remaining coefficients are fit by least
# squares on the F=128 calibration suite (C0..C3 fixed-SW + DynCache /
# DynNoSplit / DynNoCache). F-scaling is validated on the full F sweep;
# cross-graph transfer is reported to show constants do not transfer.
#
# Usage: python3 model_calibration.py  -> prints all tables
import csv, math, os, sys
import numpy as np

ROOT = 'results/baseline'

def load_csv(name):
    rows = []
    with open(os.path.join(ROOT, name)) as f:
        for r in csv.reader(f):
            if not r or r[0].startswith('#') or r[0] == 'dataset': continue
            rows.append(r)
    return rows

def get(rows, method, f):
    for r in rows:
        if r[5] == method and int(r[4]) == f:
            return float(r[9]), float(r[10]), float(r[11])
    return None

def occupancy(method):
    # 4-warp CTAs; C3/DynCache carry 16896 B dynamic shared memory.
    if method in ('C3', 'DynCache'):
        return 170 * 12
    return 170 * 16

def workload(prefix):
    with open(os.path.join(ROOT, f'{prefix}_atomic_workload.csv')) as f:
        for r in csv.DictReader(f):
            if int(r['sw_size']) == 16:
                return dict(
                    occ=int(r['total_row_occurrences']),
                    tiles=int(r['total_tiles']),
                    atomic_before=int(r['atomic_before']) // 128,
                    atomic_after=int(r['atomic_after_cache32']) // 128,
                    cta=int(r['total_sw']),
                    tail_max=int(r['cta_tiles_max']),
                )
    raise SystemExit(f'no SW=16 row in {prefix}_atomic_workload.csv')

ARX = workload('arxiv_struct')
PRO = workload('products_struct')

def dyn_stats(*names):
    out = {}
    for name in names:
        with open(os.path.join(ROOT, name)) as f:
            for line in f:
                if not line.startswith('# dyn'): continue
                parts = line.split(' ')
                key = parts[1]
                kv = {}
                for tok in parts[2:]:
                    if '=' in tok:
                        k, v = tok.split('=', 1)
                        kv[k] = float(v) if ('.' in v) else int(v)
                out[key] = kv
    return out

def point(w, method, f, task_count=None, tail=None):
    fb = math.ceil(f / 128)
    if method in ('C0', 'C1', 'C2'):
        a = w['atomic_before'] * f
        h = 0
        n = w['cta']
        tail = w['tail_max']
    elif method == 'C3':
        a = w['atomic_after'] * f
        h = w['cta']          # per-SW cache clear+sync+write-back overhead
        n = w['cta']
        tail = w['tail_max']
    else:  # dynamic
        a = w['atomic_before'] * f
        h = w['cta'] if method == 'DynCache' else 0
        n = task_count if task_count else w['cta']
        tail = tail if tail else w['tail_max']
    p = occupancy(method)
    return dict(fb=fb, wsum=w['tiles'] * fb, tail=tail * fb, atomic=a, h=h, n=n, p=p)

def collect(rows, w, dyn):
    pts = []
    for f in sorted({int(r[4]) for r in rows if r[5] not in ('Acc-Original', 'Row')}):
        for method in ('C0', 'C1', 'C2', 'C3'):
            t = get(rows, method, f)
            if t is None: continue
            pts.append((method, f, point(w, method, f), t[0]))
    for method, key in (('DynCache', 'dyn:'), ('DynNoSplit', 'dyn-nosplit:'), ('DynNoCache', 'dyn:')):
        t = get(rows, method, 128)
        if t is None: continue
        st = dyn.get(key)
        pts.append((method, 128, point(w, method, 128,
                                       st['tasks'] if st else None,
                                       st['max_task_tiles'] if st else None), t[0]))
    return pts

def residual_features(fe):
    return [fe['wsum'] / fe['p'], fe['tail'], fe['h'], fe['n'] / fe['p'], 1.0]

def predict(coef, probe_atomic, fe):
    return (coef[0] * fe['wsum'] / fe['p'] + coef[1] * fe['tail'] +
            coef[2] * fe['h'] + coef[3] * fe['n'] / fe['p'] + coef[4] +
            probe_atomic * fe['atomic'])

def table(coef, probe_atomic, pts, fonly=None):
    rows = []
    for method, f, fe, measured in pts:
        if fonly and f != fonly: continue
        pr = predict(coef, probe_atomic, fe)
        rows.append((method, f, measured, pr, (pr - measured) / measured))
    return rows

def report(coef, probe_atomic, pts, name):
    errs = []
    for method, f, measured, pr, rel in table(coef, probe_atomic, pts):
        errs.append(rel)
        print(f'   {method:11s} F={f:4d} measured={measured:9.3f} pred={pr:9.3f} ({rel*100:+6.1f}%)')
    rmse = np.sqrt(np.mean(np.array([(predict(coef, probe_atomic, fe) - m) for _, _, fe, m in pts])**2))
    print(f'   {name}: n={len(pts)} rmse={rmse:.3f} ms median-rel={np.median(errs)*100:+.1f}% '
          f'max-abs-rel={max(abs(e) for e in errs)*100:.1f}%')

def main():
    arx_rows = load_csv('arxiv_fp32_opt.csv') + load_csv('arxiv_fp32_dyn2.csv')
    pro_rows = load_csv('products_fp32_opt.csv') + load_csv('products_fp32_dyn.csv')
    arx_dyn = dyn_stats('arxiv_fp32_opt.csv', 'arxiv_fp32_dyn2.csv')
    pro_dyn = dyn_stats('products_fp32_opt.csv', 'products_fp32_dyn.csv')
    probe = dict(arxiv=5.12e-9, products=6.72e-9)   # ms/atomicAdd from probes

    arx_pts = collect(arx_rows, ARX, arx_dyn)
    pro_pts = collect(pro_rows, PRO, pro_dyn)
    suites = (('arxiv', ARX, arx_pts, probe['arxiv']), ('products', PRO, pro_pts, probe['products']))
    coefs = {}
    for name, w, pts, p in suites:
        cal = [pt for pt in pts if pt[1] == 128]
        X = np.array([residual_features(fe) for _, _, fe, _ in cal])
        y = np.array([m - p * fe['atomic'] for _, _, fe, m in cal])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        coefs[name] = coef
        print(f'== fit on {name} (F=128 calibration suite, n={len(cal)})')
        print(f'   probe atomic rate b = {p:.3e} ms/atomic (fixed)')
        print(f'   residual coef [compute/P, tail, cache/SW, cta/P, base] =',
              [f'{c:.4e}' for c in coef])
        for method, f, measured, pr, rel in table(coef, p, pts, fonly=128):
            print(f'   {method:11s} F=128 measured={measured:9.3f} pred={pr:9.3f} ({rel*100:+6.1f}%)')
        print()
        # F-scaling validation (coefficients fixed from F=128)
        rels = [(pr - m) / m for method, f, m, pr, rel in table(coef, p, pts)]
        print(f'   F-scaling (all F, n={len(rels)}): median-rel={np.median(rels)*100:+.1f}% '
              f'max-abs-rel={max(abs(r) for r in rels)*100:.1f}%')
        print()
    # cross-graph transfer: arxiv coefs on products points (products probe)
    print('== cross-graph transfer: arxiv-fit coefs -> products points')
    rels = []
    for method, f, fe, measured in pro_pts:
        pr = predict(coefs['arxiv'], probe['products'], fe)
        rels.append((pr - measured) / measured)
    print(f'   median-rel={np.median(rels)*100:+.1f}% max-abs-rel={max(abs(r) for r in rels)*100:.1f}%'
          f' -> constants are graph-specific (L2-resident vs DRAM atomics), recalibrate per graph')
    print()
    print('== cross-graph transfer: products-fit coefs -> arxiv points')
    rels = []
    for method, f, fe, measured in arx_pts:
        pr = predict(coefs['products'], probe['arxiv'], fe)
        rels.append((pr - measured) / measured)
    print(f'   median-rel={np.median(rels)*100:+.1f}% max-abs-rel={max(abs(r) for r in rels)*100:.1f}%')
    print()
    # dynamic-schedule sanity: task-count and tail reduction stats
    for name, dyn in (('arxiv', arx_dyn), ('products', pro_dyn)):
        print(f'== {name} dynamic schedules:')
        for key in ('dyn:', 'dyn-nosplit:'):
            st = dyn[key]
            print(f'   {key} tasks={st["tasks"]} max_task_tiles={st["max_task_tiles"]} '
                  f'p99={st["p99"]} mean={st["mean"]:.1f} saved_atomics={st["saved"]}')

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'int4':
        import sys as _s
        # INT4 recalibration (M4): same model structure, coefficients re-fit on
        # the fused INT4 kernel times; the schedule features are identical to
        # FP32 (same schedule/task table), the atomic term is unchanged (same
        # output atomics), the per-task compute term changes because X loads
        # are 4x smaller and carry a register dequantisation.
        arx_rows = load_csv(os.path.join('..', 'int4', 'arxiv_int4_f128.csv')) + load_csv(os.path.join('..', 'int4', 'arxiv_int4.csv'))
        pro_rows = load_csv(os.path.join('..', 'int4', 'products_int4_f128.csv')) + load_csv(os.path.join('..', 'int4', 'products_int4.csv'))
        arx_dyn = dyn_stats('arxiv_fp32_opt.csv', 'arxiv_fp32_dyn2.csv')
        pro_dyn = dyn_stats('products_fp32_opt.csv', 'products_fp32_dyn.csv')
        name_map = {'Int4-C0': 'C0', 'Int4-C1': 'C1', 'Int4-C2': 'C2', 'Int4-C3': 'C3',
                    'Int4-DynCache': 'DynCache', 'Int4-DynNoSplit': 'DynNoSplit',
                    'Int4-DynNoCache': 'DynNoCache'}
        def get4(rows, method, f):   # INT4 CSV: kernel_ms is column 6
            for r in rows:
                if r[5] == method and int(r[4]) == f:
                    return float(r[6])
            return None
        def icollect(rows, w, dyn):
            pts = []
            for f in sorted({int(r[4]) for r in rows}):
                for m2, m1 in name_map.items():
                    t = get4(rows, m2, f)
                    if t is None: continue
                    if m1.startswith('Dyn'):
                        st = dyn.get(m1 == 'DynNoSplit' and 'dyn-nosplit:' or 'dyn:')
                        pts.append((m1, f, point(w, m1, f,
                                                 st['tasks'] if st else None,
                                                 st['max_task_tiles'] if st else None), t))
                    else:
                        pts.append((m1, f, point(w, m1, f), t))
            return pts
        probe = dict(arxiv=5.12e-9, products=6.72e-9)
        for name, w, rows, dyn in (('arxiv', ARX, arx_rows, arx_dyn), ('products', PRO, pro_rows, pro_dyn)):
            pts = icollect(rows, w, dyn)
            cal = [pt for pt in pts if pt[1] == 128]
            X = np.array([residual_features(fe) for _, _, fe, _ in cal])
            y = np.array([m - probe[name] * fe['atomic'] for _, _, fe, m in cal])
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            print(f'== INT4 recalibration on {name} (F=128 suite, n={len(cal)})')
            print(f'   probe atomic rate b = {probe[name]:.3e} ms/atomic (unchanged)')
            print(f'   residual coef [compute/P, tail, cache/SW, cta/P, base] =',
                  [f'{c:.4e}' for c in coef])
            report(coef, probe[name], pts, name)
            print()
        _s.exit(0)
    main()
