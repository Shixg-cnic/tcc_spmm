#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <cnpy.h> // 需要cnpy库来处理NPZ文件

using namespace std;

typedef int64_t output_data_type;

struct MatrixMarketData {
    output_data_type num_nodes;
    output_data_type num_edges;
    vector<output_data_type> src_li;
    vector<output_data_type> dst_li;
    vector<double> values;
};

MatrixMarketData read_mtx_file(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    MatrixMarketData data;
    string line;
    bool header_processed = false;
    bool symmetric = false;
    bool pattern = false;
    
    // 读取文件头
    while (getline(file, line)) {
        if (line.empty() || line[0] == '%') {
            if (line.find("symmetric") != string::npos) {
                symmetric = true;
            }
            if (line.find("pattern") != string::npos) {
                pattern = true;
            }
            continue;
        }
        
        if (!header_processed) {
            // 读取矩阵维度信息
            istringstream iss(line);
            int64_t rows, cols, entries;
            iss >> rows >> cols >> entries;
            data.num_nodes = max(rows, cols);
            data.num_edges = entries;
            if (symmetric) {
                data.num_edges *= 2; // 对称矩阵需要两倍的边数
            }
            header_processed = true;
            continue;
        }
        
        // 读取数据行
        istringstream iss(line);
        int64_t row, col;
        double value = 1.0; // 默认值，对于pattern矩阵
        
        if (pattern) {
            iss >> row >> col;
        } else {
            iss >> row >> col >> value;
        }
        
        // MTX文件使用1-based索引，转换为0-based
        data.src_li.push_back(row - 1);
        data.dst_li.push_back(col - 1);
        data.values.push_back(value);
        
        // 如果是对称矩阵，添加对称边
        if (symmetric && row != col) {
            data.src_li.push_back(col - 1);
            data.dst_li.push_back(row - 1);
            data.values.push_back(value);
        }
    }
    
    file.close();
    return data;
}

void convert_mtx_to_npz(const string& mtx_filename, const string& npz_filename) {
    // 读取MTX文件
    MatrixMarketData data = read_mtx_file(mtx_filename);
    
    cout << "Converting MTX to NPZ..." << endl;
    cout << "Number of nodes: " << data.num_nodes << endl;
    cout << "Number of edges: " << data.src_li.size() << endl;
    
    // 转换为numpy数组
    vector<output_data_type> src_arr(data.src_li.begin(), data.src_li.end());
    vector<output_data_type> dst_arr(data.dst_li.begin(), data.dst_li.end());
    
    // 保存为NPZ文件
    cnpy::npz_save(npz_filename, "src_li", &src_arr[0], {src_arr.size()}, "w");
    cnpy::npz_save(npz_filename, "dst_li", &dst_arr[0], {dst_arr.size()}, "a");
    cnpy::npz_save(npz_filename, "num_nodes", &data.num_nodes, {}, "a");
    
    cout << "Conversion completed. Output file: " << npz_filename << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <input.mtx> <output.npz>" << endl;
        return 1;
    }
    
    string input_file = argv[1];
    string output_file = argv[2];
    
    try {
        convert_mtx_to_npz(input_file, output_file);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}