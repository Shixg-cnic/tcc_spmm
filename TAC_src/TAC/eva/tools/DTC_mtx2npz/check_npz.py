import numpy as np

def check_npz_file(filename):
    data = np.load(filename)
    print("Keys in NPZ file:", list(data.keys()))
    for key in data.keys():
        arr = data[key]
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
        print(f"{key}: first 10 values = {arr[:10] if arr.ndim > 0 else arr}")

if __name__ == "__main__":
    check_npz_file("/work/gaohy/baseline/DTC-SpMM_ASPLOS24/dataset/0-0.npz")
    #check_npz_file("/work/gaohy/baseline/DTC-SpMM_ASPLOS24/dataset/Yeast.npz")
