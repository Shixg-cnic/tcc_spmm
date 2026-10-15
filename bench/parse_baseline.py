import csv
import glob
import os

for fn in sorted(glob.glob("results/kernel_rebuild/baseline_arxiv_f*.csv")):
    name = os.path.basename(fn)
    parts = name.replace(".csv", "").split("_")
    print("==", parts[2], "sw", parts[3])
    with open(fn) as f:
        for r in csv.reader(f):
            if not r or r[0].startswith("#") or r[0] == "dataset":
                continue
            if r[5] in ("Acc-Original", "Row", "C0", "C1", "C2", "C3"):
                print(f"  {r[5]:15s} kernel={float(r[9]):8.3f} complete={float(r[13]):8.3f} rmse={float(r[16]):.2e}")
