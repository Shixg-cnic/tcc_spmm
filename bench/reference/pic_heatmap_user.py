
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python pic_heatmap.py $_.mtx --output-dir $outDir
"""
MTX稀疏矩阵热力图绘制工具
读取.mtx文件并生成热力图，nnz越多颜色越深，无nnz位置为白色
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.io import mmread
from scipy.sparse import coo_matrix
import seaborn as sns

def read_mtx_file(filepath):
    """
    读取MTX格式的稀疏矩阵文件

    Args:
        filepath (str): MTX文件路径

    Returns:
        scipy.sparse matrix: 稀疏矩阵
    """
    try:
        matrix = mmread(filepath)
        print(f"成功读取MTX文件: {filepath}")
        print(f"矩阵维度: {matrix.shape}")
        print(f"非零元素数量: {matrix.nnz}")
        return matrix
    except Exception as e:
        print(f"读取MTX文件失败: {e}")
        sys.exit(1)

def create_heatmap_data(sparse_matrix, max_resolution=2000):
    """
    将稀疏矩阵转换为热力图数据

    Args:
        sparse_matrix: 稀疏矩阵
        max_resolution (int): 最大分辨率限制，避免内存溢出

    Returns:
        tuple: (密集矩阵用于绘制热力图, 缩放因子)
    """
    rows, cols = sparse_matrix.shape
    scale_factor = 1.0

    # 如果矩阵太大，进行下采样
    if rows > max_resolution or cols > max_resolution:
        scale_factor = min(max_resolution / rows, max_resolution / cols)
        new_rows = int(rows * scale_factor)
        new_cols = int(cols * scale_factor)
        print(f"矩阵过大，将从 {rows}x{cols} 下采样到 {new_rows}x{new_cols}")

        # 创建下采样后的矩阵
        dense_matrix = np.zeros((new_rows, new_cols))

        # 转换为COO格式以便访问坐标
        if not isinstance(sparse_matrix, coo_matrix):
            sparse_matrix = sparse_matrix.tocoo()

        # 将原始坐标映射到新的坐标系
        scaled_rows = (sparse_matrix.row * scale_factor).astype(int)
        scaled_cols = (sparse_matrix.col * scale_factor).astype(int)

        # 确保坐标在有效范围内
        valid_indices = (scaled_rows < new_rows) & (scaled_cols < new_cols)
        scaled_rows = scaled_rows[valid_indices]
        scaled_cols = scaled_cols[valid_indices]

        # 累加非零元素
        for i, j in zip(scaled_rows, scaled_cols):
            dense_matrix[i, j] += 1

    else:
        # 直接转换为密集矩阵
        print("转换为密集矩阵...")
        dense_matrix = sparse_matrix.toarray()

        # 计算每个位置的非零元素数量
        dense_matrix = np.where(dense_matrix != 0, 1, 0)

    return dense_matrix, scale_factor

def plot_heatmap(data, title, save_path, original_shape=None, scale_factor=1.0, dpi=300, format_type='pdf'):
    """
    绘制并保存热力图

    Args:
        data: 密集矩阵数据
        title (str): 图表标题
        save_path (str): 保存路径
        original_shape (tuple): 原始矩阵的形状 (rows, cols)
        scale_factor (float): 缩放因子
        dpi (int): 图片分辨率
        format_type (str): 保存格式 ('png', 'pdf', 'svg')
    """
    # 设置图片大小和DPI，限制最大尺寸避免内存溢出
    # 根据DPI调整最大尺寸，DPI越高，最大尺寸越小
    max_size = max(20, 6000 / (dpi / 100))  # 基础最大尺寸为20英寸，随DPI调整

    fig_width = min(max_size, max(10, data.shape[1] / 100))
    fig_height = min(max_size, max(8, data.shape[0] / 100))

    print(f"图片尺寸: {fig_width:.1f} x {fig_height:.1f} 英寸, DPI: {dpi}")

    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    # 自定义colormap：0为白色，低nnz为冷色，高nnz为暖色（无金黄色）
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    colors = [
        (1.0, 1.0, 1.0),   # 0: 白色
        (0.7, 0.85, 1.0),  # 0.1: 浅蓝
        (0.3, 0.95, 1.0),  # 0.5: 青色
        (1.0, 0.5, 0.0),   # 1: 橙色
        (1.0, 0.0, 0.0),   # 10: 红色
        (0.0, 0.0, 0.0)    # max: 黑色
    ]
    cmap = LinearSegmentedColormap.from_list('custom_coolwarm', colors, N=256)
    cmap.set_under('white')  # 0值为白色

    # 对数归一化
    import numpy as np
    max_val = np.max(data) if np.max(data) > 0 else 1
    norm = LogNorm(vmin=0.1, vmax=min(max_val, 1e4), clip=True)

    # 绘制热力图
    im = plt.imshow(data, cmap=cmap, norm=norm, aspect='equal', interpolation='nearest')


    # 设置真实坐标轴（如果进行了下采样）
    if original_shape is not None and scale_factor < 1.0:
        original_rows, original_cols = original_shape

        # y轴（行）刻度
        num_y_ticks = min(10, data.shape[0])
        y_tick_positions = np.linspace(0, data.shape[0]-1, num_y_ticks)
        y_tick_labels = [f"{int(pos / scale_factor):,}" for pos in y_tick_positions]
        plt.yticks(y_tick_positions, y_tick_labels)

        # x轴（列）刻度
        num_x_ticks = min(10, data.shape[1])
        x_tick_positions = np.linspace(0, data.shape[1]-1, num_x_ticks)
        x_tick_labels = [f"{int(pos / scale_factor):,}" for pos in x_tick_positions]
        plt.xticks(x_tick_positions, x_tick_labels)

    # 设置标题和标签（英文）
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Column Index', fontsize=12)
    plt.ylabel('Row Index', fontsize=12)

    # 添加颜色条，确保和主图一致
    cbar = plt.colorbar(im, shrink=0.8, aspect=20, extend='max')
    cbar.set_label('Non-zero Element Density', fontsize=12)

    # 调整布局
    plt.tight_layout()

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存图片
    plt.savefig(save_path, format=format_type, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"热力图已保存为: {save_path}")

    plt.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MTX稀疏矩阵热力图绘制工具')
    parser.add_argument('mtx_file', help='MTX文件路径')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                        help='输出图片格式')
    parser.add_argument('--dpi', type=int, default=300,
                        help='图片分辨率DPI (默认: 300)')
    parser.add_argument('--max-resolution', type=int, default=2000,
                        help='最大矩阵分辨率，超过此值将进行下采样 (默认: 2000)')
    parser.add_argument('--output-dir', type=str, default='./pictures',
                        help='输出图片目录 (默认: ./pictures)')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.mtx_file):
        print(f"错误: 文件 {args.mtx_file} 不存在")
        sys.exit(1)

    # 生成输出文件路径
    base_name = os.path.basename(args.mtx_file)
    name_without_ext = os.path.splitext(base_name)[0]
    output_dir = args.output_dir
    output_filename = f"{name_without_ext}.{args.format}"
    output_path = os.path.join(output_dir, output_filename)

    print(f"开始处理文件: {args.mtx_file}")

    # 读取MTX文件
    sparse_matrix = read_mtx_file(args.mtx_file)

    # 保存原始矩阵形状
    original_shape = sparse_matrix.shape

    # 创建热力图数据
    heatmap_data, scale_factor = create_heatmap_data(sparse_matrix, args.max_resolution)

    # 绘制并保存热力图
    title = f"Sparse Matrix Heatmap: {base_name}"
    plot_heatmap(heatmap_data, title, output_path, original_shape, scale_factor, args.dpi, args.format)

    print("处理完成!")

if __name__ == "__main__":
    main()

    # python pic_heatmap.py /work/lisan/datasets/test_mtx_accreorder/Flan_1565.my_reorder.mtx --output-dir ./pictures/suitesparse_414_accReorder/
