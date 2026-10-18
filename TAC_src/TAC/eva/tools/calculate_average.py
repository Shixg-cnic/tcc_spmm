import csv
import argparse
import sys
import os
from datetime import datetime

def calculate_column_average(filename, column_index=3, has_header=None):
    """
    计算CSV文件指定列的平均值
    
    Args:
        filename: CSV文件名
        column_index: 列索引（从0开始）
        has_header: 是否有表头（None表示自动检测）
    """
    values = []
    header_detected = False
    total_rows = 0
    
    try:
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            total_rows = len(rows)
            
            if not rows:
                print("错误: 文件为空")
                return None, 0, 0
            
            # 自动检测是否有表头（如果未指定）
            if has_header is None:
                first_row = rows[0]
                if len(first_row) > column_index:
                    try:
                        float(first_row[column_index])
                        has_header = False  # 第一行是数字，可能是无表头
                    except ValueError:
                        has_header = True   # 第一行不是数字，可能是有表头
                header_detected = has_header
                print(f"自动检测: 文件 {'有' if has_header else '无'} 表头")
            
            # 确定起始行
            start_index = 1 if has_header else 0
            
            # 处理数据行
            for i in range(start_index, len(rows)):
                row = rows[i]
                if len(row) > column_index:
                    try:
                        value = float(row[column_index])
                        values.append(value)
                    except ValueError:
                        print(f"警告: 第{i+1}行数据 '{row[column_index]}' 不是有效数字，跳过")
                        continue
            
            print(f"共处理了 {len(values)} 行有效数据（总共 {total_rows} 行）")
    
    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 不存在")
        return None, 0, 0
    except Exception as e:
        print(f"错误: 读取文件时发生异常 - {e}")
        return None, 0, 0
    
    # 计算平均值
    if not values:
        print("错误: 没有找到有效的数据")
        return None, 0, 0
    
    average = sum(values) / len(values)
    return average, len(values), total_rows

def write_result_to_csv(output_file, input_file, column_index, average, valid_rows, total_rows, has_header):
    """
    将结果写入CSV文件
    """
    try:
        # 检查文件是否存在，如果不存在则创建并写入表头
        file_exists = os.path.exists(output_file)
        
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writerow([
                    'timestamp', 
                    'input_file', 
                    'column_index', 
                    'column_name',
                    'average_value', 
                    'valid_rows', 
                    'total_rows',
                    'has_header',
                    'processing_date'
                ])
            
            # 写入结果数据
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                input_file,
                column_index,
                f'Column_{column_index + 1}',
                f'{average:.6f}',
                valid_rows,
                total_rows,
                str(has_header),
                datetime.now().strftime('%Y-%m-%d')
            ])
        
        print(f"结果已写入: {output_file}")
        return True
        
    except Exception as e:
        print(f"错误: 写入结果文件时发生异常 - {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='计算CSV文件指定列的平均值并保存结果')
    parser.add_argument('--filename', help='CSV文件名')
    parser.add_argument('-c', '--column', type=int, default=3, 
                       help='列索引（从0开始，默认为3即第4列）')
    parser.add_argument('--header', action='store_true', 
                       help='文件有表头')
    parser.add_argument('--no-header', action='store_true', 
                       help='文件无表头')
    parser.add_argument('-o', '--output', default='column_average_results.csv',
                       help='输出CSV文件名（默认为column_average_results.csv）')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='显示详细信息')
    
    args = parser.parse_args()
    
    # 处理表头参数冲突
    if args.header and args.no_header:
        print("错误: --header 和 --no-header 不能同时使用")
        sys.exit(1)
    
    has_header = None  # 默认自动检测
    if args.header:
        has_header = True
    elif args.no_header:
        has_header = False
    
    # 计算平均值
    average, valid_rows, total_rows = calculate_column_average(args.filename, args.column, has_header)
    
    if average is not None:
        print(f"文件: {args.filename}")
        print(f"列索引: {args.column} (第{args.column + 1}列)")
        print(f"平均值: {average:.6f}")
        print(f"有效数据行数: {valid_rows}")
        print(f"总行数: {total_rows}")
        
        # 写入结果文件
        success = write_result_to_csv(
            args.output, 
            args.filename, 
            args.column, 
            average, 
            valid_rows, 
            total_rows,
            has_header if has_header is not None else "auto"
        )
        
        if success:
            return average
        else:
            sys.exit(1)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()