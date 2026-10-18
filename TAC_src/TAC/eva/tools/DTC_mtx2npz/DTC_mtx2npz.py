import numpy as np
from scipy.io import mmread
from scipy.sparse import coo_matrix
import argparse

def mtx_to_npz(mtx_file, npz_file):
    # 读取MTX文件
    coo = mmread(mtx_file)
    
    # 转换为COO格式
    if not isinstance(coo, coo_matrix):
        coo = coo_matrix(coo)
    
    # 提取数据
    src_li = coo.row.astype(np.int64)
    dst_li = coo.col.astype(np.int64)
    num_nodes = max(coo.shape)
    
    # 保存为NPZ
    np.savez(npz_file, src_li=src_li, dst_li=dst_li, num_nodes=num_nodes)
    print(f"Converted {mtx_file} to {npz_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算CSV文件指定列的平均值并保存结果')
    parser.add_argument('--input_mtx', help='mtx文件路径')
    parser.add_argument('--output_npz', default='output.npz',
        help='输出npz文件路径（默认为output.npz）')

    args = parser.parse_args()

    mtx_to_npz(args.input_mtx, args.output_npz)

    #import sys
    #if len(sys.argv) != 3:
    #    print("Usage: python mtx_to_npz.py <input.mtx> <output.npz>")
    #    sys.exit(1)
    
    #mtx_to_npz(sys.argv[1], sys.argv[2])