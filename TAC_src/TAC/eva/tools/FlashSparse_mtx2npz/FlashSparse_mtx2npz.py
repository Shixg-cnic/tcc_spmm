import os
import argparse
import numpy as np
import scipy.sparse as sp

def mtx_to_npz(mtx_file_path, npz_output_path):
    """
    将MTX文件转换为NPZ格式
    """
    src_li = []
    dst_li = []
    
    with open(mtx_file_path, 'r') as file:
        # Skip comment lines
        for line in file:
            if line.startswith('%'):
                continue
            else:
                # Read header information: rows, cols, entries
                head = line.split()
                break
        
        # Read data lines
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                # Convert from 1-based to 0-based indexing
                src_li.append(int(parts[0]) - 1)
                dst_li.append(int(parts[1]) - 1)
    
    # Get matrix dimensions from header
    num_nodes_src = int(head[0])  # 行数（源节点）
    num_nodes_dst = int(head[1])  # 列数（目标节点）
    num_edges = int(head[2])     # 非零元素数量
    
    print(f"Matrix dimensions: {num_nodes_src} x {num_nodes_dst}")
    print(f"Number of edges: {num_edges}")
    print(f"Number of entries read: {len(src_li)}")
    
    # Create COO matrix with correct syntax
    coo_mat = sp.coo_matrix(
        (np.ones(len(src_li)), (src_li, dst_li)),  # 数据和索引
        shape=(num_nodes_src, num_nodes_dst),      # 形状
        dtype=np.float32                           # 数据类型
    )
    
    # Verify the created matrix
    print(f"Created COO matrix with shape: {coo_mat.shape}")
    print(f"COO matrix nnz: {coo_mat.nnz}")
    
    # Save to NPZ format
    np.savez(npz_output_path,
             num_nodes_src=num_nodes_src,
             num_nodes_dst=num_nodes_dst,
             num_edges=num_edges,
             src_li=coo_mat.row,   # 源节点索引
             dst_li=coo_mat.col)    # 目标节点索引
    
    print(f"Successfully converted {os.path.basename(mtx_file_path)} to {npz_output_path}")

def verify_npz_file(npz_file):
    """
    验证NPZ文件格式是否正确
    """
    try:
        data = np.load(npz_file)
        print("\nVerifying NPZ file:")
        print("NPZ文件内容:")
        for key in data.files:
            arr = data[key]
            if hasattr(arr, 'shape'):
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
                if key in ['src_li', 'dst_li'] and len(arr) > 5:
                    print(f"    first 5 values: {arr[:5]}")
            else:
                print(f"  {key}: {arr}")
        
        # 检查必需字段
        required_fields = ['num_nodes_src', 'num_nodes_dst', 'num_edges', 'src_li', 'dst_li']
        for field in required_fields:
            if field not in data.files:
                print(f"错误: 缺少必需字段 {field}")
                return False
        
        print("NPZ文件格式验证通过!")
        return True
        
    except Exception as e:
        print(f"验证错误: {e}")
        return False


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算CSV文件指定列的平均值并保存结果')
    parser.add_argument('--input_mtx', help='mtx文件路径')
    parser.add_argument('--output_npz', default='output.npz',
        help='输出npz文件路径（默认为output.npz）')
    args = parser.parse_args()

    mtx_to_npz(args.input_mtx, args.output_npz)

    # 验证生成的文件
    verify_npz_file(args.output_npz)