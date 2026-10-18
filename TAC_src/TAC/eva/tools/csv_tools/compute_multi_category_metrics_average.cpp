#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <map>
#include <unordered_map>

// 去除字符串首尾的空格和制表符
std::string trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = str.find_last_not_of(" \t");
    return str.substr(start, end - start + 1);
}

// 解析CSV行（处理简单的引号情况）
std::vector<std::string> parseCSVLine(const std::string& line) {
    std::vector<std::string> tokens;
    std::string token;
    bool inQuotes = false;
    
    for (size_t i = 0; i < line.length(); ++i) {
        char c = line[i];
        
        if (c == '"') {
            inQuotes = !inQuotes;
        } else if (c == ',' && !inQuotes) {
            tokens.push_back(trim(token));
            token.clear();
        } else {
            token += c;
        }
    }
    tokens.push_back(trim(token)); // 添加最后一个token
    
    return tokens;
}

// 分组统计数据结构
struct GroupStats {
    std::vector<double> sums;
    std::vector<int> counts;
    int metricCount;
    
    // 默认构造函数
    GroupStats() : metricCount(0) {}
    
    // 带参数的构造函数
    GroupStats(int count) : sums(count, 0.0), counts(count, 0), metricCount(count) {}
    
    // 初始化方法
    void init(int count) {
        sums.resize(count, 0.0);
        counts.resize(count, 0);
        metricCount = count;
    }
    
    void addValue(int metricIndex, double value) {
        if (metricIndex < sums.size()) {
            sums[metricIndex] += value;
            counts[metricIndex]++;
        }
    }
    
    double getAverage(int metricIndex) const {
        if (metricIndex < counts.size() && counts[metricIndex] > 0) {
            return sums[metricIndex] / counts[metricIndex];
        }
        return 0.0;
    }
    
    bool isInitialized() const {
        return metricCount > 0;
    }
};

int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc < 4) {
        std::cerr << "使用方法: " << argv[0] << " <csv文件> <分类列名> <metric1> [metric2 ...]" << std::endl;
        std::cerr << "示例: " << argv[0] << " data.csv category cpu_usage memory_usage" << std::endl;
        std::cerr << "示例: " << argv[0] << " data.csv region sales profit" << std::endl;
        return 1;
    }

    const std::string filename = argv[1];
    const std::string categoryColumn = argv[2];
    std::vector<std::string> targetMetrics;
    
    // 收集要统计的metric名称
    for (int i = 3; i < argc; ++i) {
        targetMetrics.push_back(argv[i]);
    }

    // 打开文件
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "错误: 无法打开文件 " << filename << std::endl;
        return 1;
    }

    // 读取表头
    std::string headerLine;
    if (!std::getline(file, headerLine)) {
        std::cerr << "错误: 文件为空或无法读取表头" << std::endl;
        return 1;
    }

    // 解析表头
    std::vector<std::string> headers = parseCSVLine(headerLine);
    
    // 查找分类列的索引
    auto categoryIt = std::find(headers.begin(), headers.end(), categoryColumn);
    if (categoryIt == headers.end()) {
        std::cerr << "错误: 未找到分类列 '" << categoryColumn << "'" << std::endl;
        return 1;
    }
    int categoryIndex = std::distance(headers.begin(), categoryIt);
    std::cout << "分类列 '" << categoryColumn << "' 在第 " << categoryIndex + 1 << " 列" << std::endl;

    // 查找目标metric在表中的列索引
    std::vector<int> targetIndices;
    for (const auto& metric : targetMetrics) {
        auto it = std::find(headers.begin(), headers.end(), metric);
        if (it == headers.end()) {
            std::cerr << "错误: 未找到metric '" << metric << "'" << std::endl;
            return 1;
        } else {
            int index = std::distance(headers.begin(), it);
            targetIndices.push_back(index);
            std::cout << "metric '" << metric << "' 在第 " << index + 1 << " 列" << std::endl;
        }
    }

    // 使用map来存储每个分类的统计信息
    std::map<std::string, GroupStats> categoryStats;
    int totalRows = 0;
    int validRows = 0;

    // 读取数据行
    std::string line;
    int lineNumber = 1;
    while (std::getline(file, line)) {
        lineNumber++;
        totalRows++;
        
        // 跳过空行
        if (line.empty()) {
            continue;
        }

        std::vector<std::string> row = parseCSVLine(line);
        
        // 检查列数是否匹配
        if (row.size() != headers.size()) {
            std::cerr << "警告: 第 " << lineNumber << " 行列数不匹配 (" 
                      << row.size() << " != " << headers.size() << ")，跳过此行" << std::endl;
            continue;
        }

        // 获取分类值
        std::string categoryValue = row[categoryIndex];
        if (categoryValue.empty()) {
            std::cerr << "警告: 第 " << lineNumber << " 行分类值为空，跳过此行" << std::endl;
            continue;
        }

        // 检查这个分类是否已经有统计记录，如果没有则初始化
        if (categoryStats.find(categoryValue) == categoryStats.end()) {
            categoryStats[categoryValue] = GroupStats(targetMetrics.size());
        }

        bool rowValid = true;
        
        // 处理每个目标metric
        for (size_t i = 0; i < targetIndices.size(); ++i) {
            int colIndex = targetIndices[i];
            
            if (colIndex >= row.size()) {
                std::cerr << "警告: 第 " << lineNumber << " 行索引 " << colIndex << " 超出范围" << std::endl;
                rowValid = false;
                continue;
            }

            try {
                double value = std::stod(row[colIndex]);
                categoryStats[categoryValue].addValue(i, value);
            } catch (const std::exception& e) {
                std::cerr << "警告: 第 " << lineNumber << " 行第 " << colIndex + 1 
                          << " 列的值 '" << row[colIndex] << "' 不是有效数字" << std::endl;
                rowValid = false;
            }
        }

        if (rowValid) {
            validRows++;
        }
    }

    file.close();

    // 输出结果
    std::cout << "\n分组统计结果 (" << validRows << "/" << totalRows << " 行有效数据):" << std::endl;
    std::cout << "==================================================================" << std::endl;
    
    // 输出表头
    std::cout << std::left << std::setw(20) << categoryColumn;
    for (const auto& metric : targetMetrics) {
        std::cout << std::setw(50) << (metric + "_avg");
    }
    for (const auto& metric : targetMetrics) {
        std::cout << std::setw(50) << (metric + "_cnt");
    }
    std::cout << std::endl;

    // 输出每个分类的统计结果
    for (const auto& [category, stats] : categoryStats) {
        std::cout << std::left << std::setw(20) << category;
        
        // 输出每个metric的平均值
        for (size_t i = 0; i < targetMetrics.size(); ++i) {
            double avg = stats.getAverage(i);
            std::cout << std::setw(15) << std::fixed << std::setprecision(4) << avg;
        }
        
        // 输出每个metric的数据点数量
        for (size_t i = 0; i < targetMetrics.size(); ++i) {
            std::cout << std::setw(10) << stats.counts[i];
        }
        
        std::cout << std::endl;
    }

    // 可选：输出总体统计（所有分类的加权平均）
    std::cout << "\n总体统计:" << std::endl;
    std::cout << "==================================================================" << std::endl;
    
    GroupStats overallStats(targetMetrics.size());
    int totalCategoryCount = 0;
    
    for (const auto& [category, stats] : categoryStats) {
        totalCategoryCount++;
        for (size_t i = 0; i < targetMetrics.size(); ++i) {
            overallStats.sums[i] += stats.sums[i];
            overallStats.counts[i] += stats.counts[i];
        }
    }
    
    std::cout << std::left << std::setw(20) << "ALL_CATEGORIES";
    for (size_t i = 0; i < targetMetrics.size(); ++i) {
        double avg = overallStats.getAverage(i);
        std::cout << std::setw(15) << std::fixed << std::setprecision(4) << avg;
    }
    for (size_t i = 0; i < targetMetrics.size(); ++i) {
        std::cout << std::setw(10) << overallStats.counts[i];
    }
    std::cout << std::endl;

    return 0;
}