#include "../include/class.h"

#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

// ======================== COO ========================

template <class dataType>
COO<dataType>::COO(vint num_rows, vint num_cols, vint nnz, vint *row_indices, vint *col_indices) 
    : rows(num_rows), cols(num_cols), nnz(nnz), row(row_indices), col(col_indices) {}

template <class dataType>
COO<dataType>::~COO() {
    // 因原始数据是 torch::Torch，format_transform 结束后空间自然会被释放，这里无需手动释放
}

template <class dataType>
std::string COO<dataType>::toString() const {
    std::ostringstream oss;

    // 输出基本信息
    oss << "COO Structure:\n";
    oss << "num_rows: " << rows << "\n";
    oss << "num_cols: " << cols << "\n";
    oss << "nnz: " << nnz << "\n";


    // 输出 rowWindowOffset
    oss << "[Info] row_indices: [\n\t";
    thrust::device_vector<vint> d_row_indices(row, row + nnz);
    thrust::host_vector<vint> h_row_indices = d_row_indices;
    for (size_t i = 0; i < h_row_indices.size(); ++i) {
        oss << h_row_indices[i];
        if (i != h_row_indices.size() - 1) oss << ", ";
    }
    oss << "]\n";

    // 输出 tcOffset
    oss << "[Info] col_indices: [\n\t";
    thrust::device_vector<vint> d_col_indices(col, col + nnz);
    thrust::host_vector<vint> h_col_indices = d_col_indices;
    for (size_t i = 0; i < h_col_indices.size(); ++i) {
        oss << h_col_indices[i];
        if (i != h_col_indices.size() - 1) oss << ", ";
    }
    oss << "]\n";

    return oss.str();
}

// ======================== BCSR ========================

template <class dataType>
BCSR<dataType>::BCSR(const COO<dataType>& coo, vint* rowWindowOffset, vint* tcOffset, vint* sparseA2B, TCLOCAL_TYPE* tcLocalBit, int pipe) : 
    num_rows(coo.rows), num_cols(coo.cols), nnz(coo.nnz), 
    rowWindowOffset(rowWindowOffset), tcOffset(tcOffset), 
    sparseA2B(sparseA2B), tcLocalBit(tcLocalBit)
{
    // arxiv 和 mag 可能因为节点数量太少存在这种情况
    //if (coo.nnz == 0) {
    //    num_rowWindows = 0;
    //    num_TCBlocks = 0;
    //    return;
    //}
    
    thrust::device_vector<vint> x(coo.row, coo.row + coo.nnz);
    thrust::device_vector<vint> y(coo.col, coo.col + coo.nnz);

    COO2BCSR(x, y, pipe);
}

template <class dataType>
void BCSR<dataType>::COO2BCSR(thrust::device_vector<vint> &x, thrust::device_vector<vint> &y, int pipe) {
    vint window_num = (num_rows-1)/ROW_WINDOW + 2;
    num_rowWindows = window_num - 1;

    cudaMemset(rowWindowOffset, 0, (num_rowWindows + 1) * sizeof(vint)); // 使用原始指针时必须进行手动初始化，可以在 memorypool get 时就开始异步初始化了
    auto rowWindowOffset_dp = thrust::device_pointer_cast(rowWindowOffset); // device pointer

    thrust::device_vector<vint> group(x.size());
    // 组内对y进行去重(去重列)
    thrust::device_vector<vint> uni_y(x.size());
    thrust::device_vector<vint> uni_group(x.size());
    
    // 先计算每个点的group
    thrust::transform(
        thrust::device,
        x.begin(), x.end(), 
        group.begin(), 
        thrust::placeholders::_1 / ROW_WINDOW
    );
    
    // 对group内的y进行排序，x随动（先排group再排y）
    auto order_begin = thrust::make_zip_iterator(thrust::make_tuple(group.begin(), y.begin(), x.begin()));
    auto order_end = thrust::make_zip_iterator(thrust::make_tuple(group.end(), y.end(), x.end()));
    thrust::sort(
        thrust::device,
        order_begin, order_end,
        [] __device__ (const auto& a, const auto& b) {
            // 先比较组号
            if (thrust::get<0>(a) != thrust::get<0>(b))
                return thrust::get<0>(a) < thrust::get<0>(b);
            // 同一组内按 y 排序
            return thrust::get<1>(a) < thrust::get<1>(b);
        }
    );    // group_id、y和x都变得有序了
    
    // 对group内y进行去重，然后计数group内元素数量，得到最大tc块值
    auto group_begin = thrust::make_zip_iterator(thrust::make_tuple(group.begin(), y.begin()));
    auto uni_begin = thrust::make_zip_iterator(thrust::make_tuple(uni_group.begin(), uni_y.begin()));
    auto new_end = thrust::unique_copy(group_begin, group_begin + uni_y.size(), uni_begin);
    vint unique_count = new_end - uni_begin;
    // 获得了去重的group和col值(每个非零元都存在一个 group 编号，即所在 window 编号，以及真实列号，这里把重复的真实列号进行了去重，后面计数统计的就是每个 window 内的实际包含非零元的列数)
    uni_y.resize(unique_count);
    uni_group.resize(unique_count);
    
    // 对每个window中的去重行数进行计数
    vint* rwo_ptr = rowWindowOffset; // change3
    thrust::for_each(
        thrust::device,
        uni_group.begin(), uni_group.end(), 
        [=] __device__ (const vint idx) {
            atomicAdd(&rwo_ptr[idx + 1], 1);
        });
    
    // 计算每个row_window的tc块数
    thrust::transform(
        thrust::device,
        rowWindowOffset_dp, rowWindowOffset_dp + (num_rowWindows + 1), 
        rowWindowOffset_dp, 
        [=] __device__ (vint y) {
            return static_cast<vint>(std::ceil(static_cast<float>(y) / COL_WINDOW));
        }
    );
    // 前缀和计算rowWindowOffset
    thrust::inclusive_scan(
        thrust::device,
        rowWindowOffset_dp, rowWindowOffset_dp + (num_rowWindows + 1), 
        rowWindowOffset_dp
    );
    
    // rowWindowOffset中最后一个元素作为其他数组的size
    vint size;
    cudaMemcpy(&size, rwo_ptr + window_num - 1, sizeof(vint), cudaMemcpyDeviceToHost);

    num_TCBlocks = size;

    // 其他数组
    cudaMemset(tcOffset, 0, (num_TCBlocks + 1) * sizeof(vint));
    auto tcOffset_dp = thrust::device_pointer_cast(tcOffset);

    auto sparseA2B_dp = thrust::device_pointer_cast(sparseA2B);
    thrust::fill(sparseA2B_dp, sparseA2B_dp + (num_TCBlocks * COL_WINDOW), UINT32_MAX);

    cudaMemset(tcLocalBit, 0, num_TCBlocks * sizeof(TCLOCAL_TYPE));
    auto tcLocalBit_dp = thrust::device_pointer_cast(tcLocalBit);
    
    vint* tco_ptr = tcOffset;

    // flag，记录每个位置是否与前一位置相同。这里 flag 是想获取每个非零元所在 window 内的 local 列号，用于定位所在 tc block 的 local id
    thrust::device_vector<vint> flags(nnz);

    // 组内第一个元素 flag 固定为 0
    flags[0] = 0;
    // 如果同组且 col 与前一元素不同，则标记 1，否则标记 0
    thrust::transform(
        thrust::device,
        thrust::make_counting_iterator(1), thrust::make_counting_iterator(static_cast<int>(nnz)),
        flags.begin() + 1,
        [group_ptr = thrust::raw_pointer_cast(group.data()),
        row_ptr   = thrust::raw_pointer_cast(y.data())] __device__ (vint i) {
            // 如果不在同一组，则重启，新组的第一个固定为 0
            if (group_ptr[i] != group_ptr[i - 1])
                return 0;
            // 同一组内，如果 col 发生变化，则认为是新值
            return (row_ptr[i] != row_ptr[i - 1]) ? 1 : 0;
        }
    );
    
    // 对组内元素求前缀和(不去重)
    thrust::inclusive_scan_by_key(
        thrust::device,
        group.begin(), group.end(), 
        flags.begin(), 
        flags.begin()
    );        
    
    // 通过（cwo[group_id]+tc_Index）位置自增计数 -> tcOffset
    auto tc_begin = thrust::make_zip_iterator(thrust::make_tuple(group.begin(), flags.begin()));
    auto tc_end   = thrust::make_zip_iterator(thrust::make_tuple(group.end(), flags.end()));
    thrust::for_each(
        thrust::device,
        tc_begin, tc_end, 
        [=] __device__ (const thrust::tuple<vint, vint>& t) {
            vint base_val   = thrust::get<0>(t);
            vint offset_val = thrust::get<1>(t);
            vint index = rwo_ptr[base_val] + offset_val/ROW_WINDOW; // tc block global id + tc block local id
            // printf("%d",index);
            atomicAdd(&tco_ptr[index+1], 1);
        }
    );
    thrust::inclusive_scan(
        thrust::device,
        tcOffset_dp, tcOffset_dp + (size + 1), 
        tcOffset_dp
    );        
    
    // // uni元素的连续索引
    thrust::device_vector<vint> uni_idx(uni_group.size());
    
    // 生成组内的连续索引编号
    thrust::exclusive_scan_by_key(
        thrust::device,
        uni_group.begin(),                            // 键的起始迭代器
        uni_group.end(),                              // 键的结束迭代器
        thrust::make_constant_iterator(1),       // 值迭代器，全1
        uni_idx.begin(),                            // 输出索引数组
        0,                                      // 初始值
        [] __device__ (vint a, vint b) {           // Lambda比较键是否相等
            return a == b;
        },
        thrust::plus<vint>()                      // 扫描操作为加法
    );
    
    // 计算去重元插入sparseA2B的位置
    auto sp_begin = thrust::make_zip_iterator(thrust::make_tuple(uni_group.begin(), uni_idx.begin()));
    auto sp_end   = thrust::make_zip_iterator(thrust::make_tuple(uni_group.end(), uni_idx.end()));
    thrust::transform(
        thrust::device,
        sp_begin, sp_end, uni_idx.begin(),
        [=] __device__ (const thrust::tuple<vint, vint>& t) {
            vint group = thrust::get<0>(t);     // 组号
            vint lid = thrust::get<1>(t);     // local_id

            return rwo_ptr[group] * ROW_WINDOW + lid;
            // return row * COL_WINDOW + lid % COL_WINDOW;
        }
    );
    
    thrust::scatter(
        thrust::device,
        uni_y.begin(),  // 值0,1,...,n-1
        uni_y.end(),
        uni_idx.begin(),                    // 目标位置由indices数组指定
        sparseA2B_dp                 // 结果存入sparseA2B
    );
    
    // step 3. 填充TCLocalBit
    // ss1. 计算每个非零元的左移位数
    // x是每个不去重元素的行号，flags不去重局部id(列)
    auto move_begin = thrust::make_zip_iterator(thrust::make_tuple(x.begin(), flags.begin()));
    auto move_end   = thrust::make_zip_iterator(thrust::make_tuple(x.end(), flags.end()));
    TCLOCAL_TYPE* tcLB_ptr = tcLocalBit;   
    thrust::for_each(
        thrust::device,
        move_begin, move_end, 
        [=] __device__ (const thrust::tuple<vint, vint>& t) {
        vint row = thrust::get<0>(t);
        vint lid = thrust::get<1>(t);
        vint rwo_index = row / ROW_WINDOW;     // 组号
        vint tc_lid = lid / COL_WINDOW;        // window内的tc块号
        int part = ((row%ROW_WINDOW) * COL_WINDOW + lid % COL_WINDOW);
        // printf("%d:%d \n",row,part);
        atomicOr(reinterpret_cast<unsigned long long*>(&tcLB_ptr[rwo_ptr[rwo_index] + tc_lid]), (1ULL << part));
    });
}

template <class dataType>
std::string BCSR<dataType>::toString() const {
    std::ostringstream oss;

    // 输出基本信息
    oss << "GBCSR Structure:\n";
    oss << "num_rows: " << num_rows << "\n";
    oss << "num_cols: " << num_cols << "\n";
    oss << "nnz: " << nnz << "\n";
    oss << "num_TCBlocks: " << num_TCBlocks << "\n";
    oss << "num_rowWindows: " << num_rowWindows << "\n";



    // 输出 rowWindowOffset
    oss << "\t[Info] rowWindowOffset: [";
    thrust::device_vector<vint> d_rowWindowOffset(rowWindowOffset, rowWindowOffset + (num_rowWindows + 1));
    thrust::host_vector<vint> h_rowWindowOffset = d_rowWindowOffset;
    for (size_t i = 0; i < h_rowWindowOffset.size(); ++i) {
        oss << h_rowWindowOffset[i];
        if (i != h_rowWindowOffset.size() - 1) oss << ", ";
    }
    oss << "]\n";

    // 输出 tcOffset
    oss << "\t[Info] tcOffset: [";
    thrust::device_vector<vint> d_tcOffset(tcOffset, tcOffset + (num_TCBlocks + 1));
    thrust::host_vector<vint> h_tcOffset = d_tcOffset;
    for (size_t i = 0; i < h_tcOffset.size(); ++i) {
        oss << h_tcOffset[i];
        if (i != h_tcOffset.size() - 1) oss << ", ";
    }
    oss << "]\n";

    // 输出 sparseA2B
    oss << "\t[Info] sparseA2B: [";
    thrust::device_vector<vint> d_sparseA2B(sparseA2B, sparseA2B + num_TCBlocks * COL_WINDOW);
    thrust::host_vector<vint> h_sparseA2B = d_sparseA2B;
    for (size_t i = 0; i < h_sparseA2B.size(); ++i) {
        oss << h_sparseA2B[i];
        if (i != h_sparseA2B.size() - 1) oss << ", ";
    }
    oss << "]\n";

    // 输出 tcLocalBit
    oss << "\t[Info] tcLocalBit: [";
    thrust::device_vector<TCLOCAL_TYPE> d_tcLocalBit(tcLocalBit, tcLocalBit + num_TCBlocks);
    thrust::host_vector<TCLOCAL_TYPE> h_tcLocalBit = d_tcLocalBit;
    for (size_t i = 0; i < h_tcLocalBit.size(); ++i) {
        oss << h_tcLocalBit[i];
        if (i != h_tcLocalBit.size() - 1) oss << ", ";
    }
    oss << "]\n";

    return oss.str();
}

// ======================== BCSC ========================

template <class dataType>
BCSC<dataType>::BCSC(const COO<dataType>& coo, vint* colWindowOffset, vint* tcOffset, vint* sparseA2C, TCLOCAL_TYPE* tcLocalBit, int pipe) : 
    num_rows(coo.rows), num_cols(coo.cols), nnz(coo.nnz),
    colWindowOffset(colWindowOffset), tcOffset(tcOffset), 
    sparseA2C(sparseA2C), tcLocalBit(tcLocalBit)
{
    // arxiv 和 mag 可能因为节点数量太少存在这种情况
    //if (coo.nnz == 0) {
    //    num_colWindows = 0;
    //    num_TCBlocks = 0;
    //    return;
    //}

    thrust::device_vector<vint> x(coo.row, coo.row + coo.nnz);
    thrust::device_vector<vint> y(coo.col, coo.col + coo.nnz);

    COO2BCSC(x, y, pipe);
}

template <class dataType>
void BCSC<dataType>::COO2BCSC(thrust::device_vector<vint> &x, thrust::device_vector<vint> &y, int pipe) {
    num_colWindows = (num_cols - 1) / COL_WINDOW + 1;

    cudaMemset(colWindowOffset, 0, (num_colWindows + 1) * sizeof(vint));
    auto colWindowOffset_dp = thrust::device_pointer_cast(colWindowOffset);

    thrust::device_vector<vint> group(x.size());
    // 组内对x进行去重
    thrust::device_vector<vint> uni_x(x.size());
    thrust::device_vector<vint> uni_group(x.size());
    
    // 先计算每个点的group
    thrust::transform(
        thrust::device,
        y.begin(), y.end(), 
        group.begin(), 
        thrust::placeholders::_1 / COL_WINDOW
    );

    // 对group内的x进行排序，y随动（先排group再排x）
    auto order_begin = thrust::make_zip_iterator(thrust::make_tuple(group.begin(), x.begin(), y.begin()));
    auto order_end = thrust::make_zip_iterator(thrust::make_tuple(group.end(), x.end(), y.end()));
    thrust::sort(
        thrust::device,
        order_begin, order_end,
        [] __device__ (const auto& a, const auto& b) {
            // 先比较组号
            if (thrust::get<0>(a) != thrust::get<0>(b))
                return thrust::get<0>(a) < thrust::get<0>(b);
            // 同一组内按 offset 排序
            return thrust::get<1>(a) < thrust::get<1>(b);
        }
    );    // group_id、x和y都变得有序了
    

    // 对group内x进行去重，然后计数group内元素数量，得到最大tc块值
    auto group_begin = thrust::make_zip_iterator(thrust::make_tuple(group.begin(), x.begin()));
    auto uni_begin = thrust::make_zip_iterator(thrust::make_tuple(uni_group.begin(), uni_x.begin()));
    auto new_end = thrust::unique_copy(group_begin, group_begin + uni_x.size(), uni_begin);
    vint unique_count = new_end - uni_begin;
    uni_x.resize(unique_count);
    uni_group.resize(unique_count);
    
    // 对每个window中的去重行数
    vint* cwo_ptr = colWindowOffset;
    thrust::for_each(
        thrust::device,
        uni_group.begin(), uni_group.end(), 
        [=] __device__ (const vint idx) {
            atomicAdd(&cwo_ptr[idx + 1], 1);
        });
    
    // 计算每个window的tc块数
    thrust::transform(
        thrust::device,
        colWindowOffset_dp, colWindowOffset_dp + (num_colWindows + 1), colWindowOffset, 
        [=] __device__ (vint x) {
            return static_cast<vint>(std::ceil(static_cast<float>(x) / ROW_WINDOW));
        }
    );
    // 前缀和计算colWindowOffset
    thrust::inclusive_scan(
        thrust::device,
        colWindowOffset_dp, colWindowOffset_dp + (num_colWindows + 1), 
        colWindowOffset_dp
    );
    
    // colWindowOffset中最后一个元素作为其他数组的size
    vint size;
    cudaMemcpy(&size, cwo_ptr + num_colWindows, sizeof(vint), cudaMemcpyDeviceToHost);

    num_TCBlocks = size;

    // 其他数组
    cudaMemset(tcOffset, 0, (num_TCBlocks + 1) * sizeof(vint));
    auto tcOffset_dp = thrust::device_pointer_cast(tcOffset);

    auto sparseA2C_dp = thrust::device_pointer_cast(sparseA2C);
    thrust::fill(
        thrust::device,
        sparseA2C_dp, sparseA2C_dp + (num_TCBlocks * ROW_WINDOW), 
        UINT32_MAX
    );

    cudaMemset(tcLocalBit, 0, num_TCBlocks * sizeof(TCLOCAL_TYPE));
    auto tcLocalBit_dp = thrust::device_pointer_cast(tcLocalBit);

    vint* tco_ptr = tcOffset;

    // flag，记录每个位置是否与前一位置相同。
    thrust::device_vector<vint> flags(nnz);

    // 组内第一个元素 flag 固定为 0
    flags[0] = 0;
    // 如果同组且 row 与前一元素不同，则标记 1，否则标记 0
    thrust::transform(
        thrust::device,
        thrust::make_counting_iterator(1),
        thrust::make_counting_iterator(static_cast<int>(nnz)),
        flags.begin() + 1,
        [group_ptr = thrust::raw_pointer_cast(group.data()),
        col_ptr   = thrust::raw_pointer_cast(x.data())] __device__ (vint i) {    /// x????
            // 如果不在同一组，则重启，新组的第一个固定为 0
            if (group_ptr[i] != group_ptr[i - 1])
                return 0;
            // 同一组内，如果 col 发生变化，则认为是新值
            return (col_ptr[i] != col_ptr[i - 1]) ? 1 : 0;
        }
    );
    
    // 对组内元素求前缀和(不去重)-> 得到组内连续索引id
    thrust::inclusive_scan_by_key(
        thrust::device,
        group.begin(), group.end(), 
        flags.begin(), 
        flags.begin()
    );

    // 通过（rwo[group_id]+tc_Index）位置自增计数 -> tcOffset
    auto tc_begin = thrust::make_zip_iterator(thrust::make_tuple(group.begin(), flags.begin()));
    auto tc_end   = thrust::make_zip_iterator(thrust::make_tuple(group.end(), flags.end()));
    thrust::for_each(
        thrust::device,
        tc_begin, tc_end, 
        [=] __device__ (const thrust::tuple<vint, vint>& t) {
            vint base_val   = thrust::get<0>(t);
            vint offset_val = thrust::get<1>(t);
            vint index = cwo_ptr[base_val] + offset_val/COL_WINDOW;
            // printf("%d",index);
            atomicAdd(&tco_ptr[index+1], 1);
        }
    );
    thrust::inclusive_scan(
        thrust::device,
        tcOffset_dp, tcOffset_dp + (num_TCBlocks + 1), 
        tcOffset_dp
    );
    
    // // uni元素的连续索引
    thrust::device_vector<vint> uni_idx(uni_group.size());
    
    // 生成组内的连续索引编号
    thrust::exclusive_scan_by_key(
        thrust::device,
        uni_group.begin(),                            // 键的起始迭代器
        uni_group.end(),                              // 键的结束迭代器
        thrust::make_constant_iterator(1),         // 值迭代器，全1
        uni_idx.begin(),                            // 输出索引数组
        0,                                      // 初始值
        [] __device__ (vint a, vint b) {           // Lambda比较键是否相等
            return a == b;
        },
        thrust::plus<vint>()                      // 扫描操作为加法
    );
    
    // 计算去重元插入sparseA2B的位置
    auto sp_begin = thrust::make_zip_iterator(thrust::make_tuple(uni_group.begin(), uni_idx.begin()));
    auto sp_end   = thrust::make_zip_iterator(thrust::make_tuple(uni_group.end(), uni_idx.end()));
    thrust::transform(
        thrust::device,
        sp_begin, sp_end, uni_idx.begin(),
        [=] __device__ (const thrust::tuple<vint, vint>& t) {
            vint group = thrust::get<0>(t);     // 组号
            vint lid = thrust::get<1>(t);     // local_id

            return cwo_ptr[group] * COL_WINDOW + lid;
            // return row * COL_WINDOW + lid % COL_WINDOW;
        }
    );
    
    thrust::scatter(
        thrust::device,
        uni_x.begin(),  // 值0,1,...,n-1
        uni_x.end(),
        uni_idx.begin(),                    // 目标位置由indices数组指定
        sparseA2C_dp                 // 结果存入sparseA2C
    );
    
    // step 3. 填充TCLocalBit
    // ss1. 计算每个非零元的左移位数
    // y是每个不去重元素的列号，flags不去重局部id(行)
    auto move_begin = thrust::make_zip_iterator(thrust::make_tuple(y.begin(), flags.begin()));
    auto move_end   = thrust::make_zip_iterator(thrust::make_tuple(y.end(), flags.end()));
    TCLOCAL_TYPE* tcLB_ptr = tcLocalBit;   
    thrust::for_each(
        thrust::device,
        move_begin, move_end, 
        [=] __device__ (const thrust::tuple<vint, vint>& t) {
        vint col = thrust::get<0>(t);
        vint lid = thrust::get<1>(t);
        vint cwo_index = col / COL_WINDOW;     // 组号
        vint tc_lid = lid / ROW_WINDOW;
        // vint part = (ROW_WINDOW * COL_WINDOW) - ((lid%ROW_WINDOW) * COL_WINDOW + col % COL_WINDOW) - 1;
        vint part = (lid%ROW_WINDOW) * COL_WINDOW + col % COL_WINDOW;
        atomicOr(reinterpret_cast<unsigned long long*>(&tcLB_ptr[cwo_ptr[cwo_index] + tc_lid]), (1ULL << part));
    });
}

template <class dataType>
std::string BCSC<dataType>::toString() const {
    std::ostringstream oss;

    // 输出基本信息
    oss << "GBCSC Structure:\n";
    oss << "num_rows: " << num_rows << "\n";
    oss << "num_cols: " << num_cols << "\n";
    oss << "nnz: " << nnz << "\n";
    oss << "num_TCBlocks: " << num_TCBlocks << "\n";
    oss << "num_colWindows: " << num_colWindows << "\n";

    // 输出 rowWindowOffset
    oss << "\t[Info] colWindowOffset: [";
    thrust::device_vector<vint> d_colWindowOffset(colWindowOffset, colWindowOffset + (num_colWindows + 1));
    thrust::host_vector<vint> h_colWindowOffset = d_colWindowOffset;
    for (size_t i = 0; i < h_colWindowOffset.size(); ++i) {
        oss << h_colWindowOffset[i];
        if (i != h_colWindowOffset.size() - 1) oss << ", ";
    }
    oss << "]\n";

    // 输出 tcOffset
    oss << "\t[Info] tcOffset: [";
    thrust::device_vector<vint> d_tcOffset(tcOffset, tcOffset + (num_TCBlocks + 1));
    thrust::host_vector<vint> h_tcOffset = d_tcOffset;
    for (size_t i = 0; i < h_tcOffset.size(); ++i) {
        oss << h_tcOffset[i];
        if (i != h_tcOffset.size() - 1) oss << ", ";
    }
    oss << "]\n";

    // 输出 sparseA2B
    oss << "\t[Info] sparseA2C: [";
    thrust::device_vector<vint> d_sparseA2C(sparseA2C, sparseA2C + num_TCBlocks * ROW_WINDOW);
    thrust::host_vector<vint> h_sparseA2C = d_sparseA2C;
    for (size_t i = 0; i < h_sparseA2C.size(); ++i) {
        oss << h_sparseA2C[i];
        if (i != h_sparseA2C.size() - 1) oss << ", ";
    }
    oss << "]\n";

    // 输出 tcLocalBit
    oss << "\t[Info] tcLocalBit: [";
    thrust::device_vector<TCLOCAL_TYPE> d_tcLocalBit(tcLocalBit, tcLocalBit + num_TCBlocks);
    thrust::host_vector<TCLOCAL_TYPE> h_tcLocalBit = d_tcLocalBit;
    for (size_t i = 0; i < h_tcLocalBit.size(); ++i) {
        oss << h_tcLocalBit[i];
        if (i != h_tcLocalBit.size() - 1) oss << ", ";
    }
    oss << "]\n";

    return oss.str();
}

// 必须将模板实例化写在实现文件的末尾
template struct COO<float>;
//template struct COO<double>;

template class BCSR<float>;
//template class BCSR<double>;

template class BCSC<float>;
//template class BCSC<double>;