#include "storage_management.cuh"
#include "storage_management_impl.cuh"

// 导入亲和算法的根节点
void readCSVToIntArray(std::vector<int32_t>& data, const std::string& filename, int start, int end) {
    std::ifstream file(filename);

    if (file.is_open()) {
        std::string line;
        int currentLine = 0;
        // 遍历文件中的所有行
        while (std::getline(file, line)) {
            if (currentLine >= start && (end == -1 || currentLine <= end)) {
                try {
                    // 将每一行的字符串转换为 int32，并加入 data 动态数组
                    int32_t num = std::stoi(line);
                    data.push_back(num);
                } catch (const std::exception& e) {
                    std::cerr << "Error converting line to int: " << line << std::endl;
                }
            }
            if (currentLine > end && end != -1) {
                break;
            }
            currentLine++;
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

void StorageManagement::EnableP2PAccess(){
    int32_t device_count = -1;
    cudaGetDeviceCount(&device_count);
    //std::printf("p2p detect device count: %d\n", device_count);
    //device_count = 4;
    for(int32_t i = 0; i < device_count; i++){
        cudaSetDevice(i);
        cudaCheckError();
        for(int32_t j = 0; j < device_count; j++){
          if(j != i){
            int32_t accessible = 0;
            cudaDeviceCanAccessPeer(&accessible, i, j);
            cudaCheckError();
            if(accessible){
              cudaDeviceEnablePeerAccess(j, 0);
              cudaCheckError();
            }
          }
        }
    }
}

void StorageManagement::ConfigPartition(BuildInfo* info, int32_t partition_count){
    info->partition_count = partition_count;
}

void StorageManagement::ReadMetaFIle(BuildInfo* info){
    std::istringstream iss;
    std::string buff;
    std::ifstream Metafile("./meta_config");
    if(!Metafile.is_open()){
     std::cout<<"unable to open meta config file"<<"\n";
    }
    getline(Metafile, buff);
    iss.clear();
    iss.str(buff);
    if(in_memory_mode_){
        iss >> dataset_path_;
        std::cout<<"Dataset path:       "<<dataset_path_<<"\n";
        iss >> raw_batch_size_;
        std::cout<<"Raw Batchsize:      "<<raw_batch_size_<<"\n";
        info->raw_batch_size = raw_batch_size_;
        iss >> node_num_;
        std::cout<<"Graph nodes num:    "<<node_num_<<"\n";
        iss >> edge_num_;
        std::cout<<"Graph edges num:    "<<edge_num_<<"\n";
        iss >> float_feature_len_;
        std::cout<<"Feature dim:        "<<float_feature_len_<<"\n";
        iss >> training_set_num_;
        std::cout<<"Training set num:   "<<training_set_num_<<"\n";
        iss >> validation_set_num_;
        std::cout<<"Validation set num: "<<validation_set_num_<<"\n";
        iss >> testing_set_num_;
        std::cout<<"Testing set num:    "<<testing_set_num_<<"\n";
        iss >> cache_memory_;
        std::cout<<"Cache memory:       "<<cache_memory_<<"\n";
        iss >> epoch_;
        std::cout<<"Train epoch:        "<<epoch_<<"\n";
        info->epoch = epoch_;
    }else{
        iss >> dataset_path_;
        std::cout<<"Dataset path:       "<<dataset_path_<<"\n";
        iss >> raw_batch_size_;
        std::cout<<"Raw Batchsize:      "<<raw_batch_size_<<"\n";
        info->raw_batch_size = raw_batch_size_;
        iss >> node_num_;
        std::cout<<"Graph nodes num:    "<<node_num_<<"\n";
        iss >> edge_num_;
        std::cout<<"Graph edges num:    "<<edge_num_<<"\n";
        iss >> float_feature_len_;
        std::cout<<"Feature dim:        "<<float_feature_len_<<"\n";
        iss >> training_set_num_;
        std::cout<<"Training set num:   "<<training_set_num_<<"\n";
        iss >> validation_set_num_;
        std::cout<<"Validation set num: "<<validation_set_num_<<"\n";
        iss >> testing_set_num_;
        std::cout<<"Testing set num:    "<<testing_set_num_<<"\n";
        iss >> cache_memory_;
        std::cout<<"Cache memory:       "<<cache_memory_<<"\n";
        iss >> epoch_;
        std::cout<<"Train epoch:        "<<epoch_<<"\n";
        info->epoch = epoch_;
        iss >> partition_;
        std::cout<<"Partition?:         "<<partition_<<"\n";
        iss >> num_ssd_;
        std::cout<<"SSD Num?:           "<<num_ssd_<<"\n";
        iss >> num_queues_per_ssd_;
        std::cout<<"Q/SSD    ?:         "<<num_queues_per_ssd_<<"\n";
        iss >> cpu_cache_capacity_;
        std::cout<<"CPU Cache Capacity: "<<cpu_cache_capacity_<<"\n";
        iss >> gpu_cache_capacity_;
        std::cout<<"GPU Cache Capacity: "<<gpu_cache_capacity_<<"\n";
    }
    

}

void StorageManagement::LoadGraph(BuildInfo* info){

    int32_t node_num = node_num_;
    int64_t edge_num = edge_num_;
    info->total_edge_num = edge_num;
    info->cache_edge_num = cache_edge_num_;

    //uva
    cudaHostAlloc(&(info->csr_node_index), int64_t(int64_t(node_num + 1)*sizeof(int64_t)), cudaHostAllocMapped);
    cudaHostAlloc(&(info->csr_dst_node_ids), int64_t(int64_t(edge_num) * sizeof(int32_t)), cudaHostAllocMapped);
    std::string edge_src_path = dataset_path_ + "edge_src";
    std::string edge_dst_path = dataset_path_ + "edge_dst";

    mmap_indptr_read(edge_src_path, info->csr_node_index);
    mmap_indices_read(edge_dst_path, info->csr_dst_node_ids);
}


void StorageManagement::LoadFeature(BuildInfo* info){

    int32_t partition_count = info->partition_count;

    int32_t node_num = node_num_;
    int32_t nf = float_feature_len_;

    info->numElems = uint64_t(node_num) * nf;

    (info->training_set_ids).resize(partition_count);
    (info->training_labels).resize(partition_count);
    (info->validation_set_ids).resize(partition_count);
    (info->validation_labels).resize(partition_count);
    (info->testing_set_ids).resize(partition_count);
    (info->testing_labels).resize(partition_count);

    std::string training_path = dataset_path_  + "trainingset";
    std::string validation_path = dataset_path_  + "validationset";
    std::string testing_path = dataset_path_  + "testingset";
    // std::string training_path = dataset_path_  + "train_ids";
    // std::string validation_path = dataset_path_  + "valid_ids";
    // std::string testing_path = dataset_path_  + "test_ids";
    std::string features_path = dataset_path_ + "features";
    std::string labels_path = dataset_path_ + "labels";
    // std::string labels_path = dataset_path_ + "labels_raw";

    std::string partition_path = dataset_path_ + "partition";

    std::vector<int32_t> training_ids;
    training_ids.resize(training_set_num_);
    std::vector<int32_t> validation_ids;
    validation_ids.resize(validation_set_num_);
    std::vector<int32_t> testing_ids;
    testing_ids.resize(testing_set_num_);
    std::vector<int32_t> all_labels;
    #define no_read_label
    #ifdef no_read_label
    all_labels.resize(node_num);      // add feature
    #endif
    #undef no_read_label
    int32_t* partition_index = (int32_t*)malloc(int64_t(node_num) * sizeof(int32_t));
    float* host_float_feature;

    mmap_trainingset_read(training_path, training_ids);
    mmap_trainingset_read(validation_path, validation_ids);
    mmap_trainingset_read(testing_path, testing_ids);
    if(in_memory_mode_){
        cudaHostAlloc(&host_float_feature, int64_t(int64_t(int64_t(node_num) * nf) * sizeof(float)), cudaHostAllocMapped);
        // mmap_features_read(features_path, host_float_feature);
        //#define read_feature
        #ifdef read_feature
        read_features_from_csv(features_path, host_float_feature);
        std::cout << "[Info] Finish Reading features" << std::endl;
        #endif
    }
    // mmap_labels_read(labels_path, all_labels);
    //#define read_label
    #ifdef read_label
    read_labels_from_csv(labels_path, all_labels);
    std::cout << "[Info] Finish Reading labels" << std::endl;
    #endif

    int32_t fdret = mmap_partition_read(partition_path, partition_index);

    std::cout<<"Finish Reading All Files\n";
    // partition nodes

    int affinity_method = 0;
    if (affinity_method == 0){
        int trainingset_count = 0;
        for(int32_t i = 0; i < training_set_num_; i+=1){
            int32_t tid = training_ids[i];
            int32_t part_id;
            if(fdret >= 0){
                part_id = partition_index[tid];
            }else{
                part_id = tid % partition_count;
            }
            if(part_id < partition_count){
                (info->training_set_ids[part_id]).push_back(tid);
                trainingset_count ++ ;
            }
        }
        std::cout<<"training set count "<<trainingset_count<<"\n";

        for(int32_t i = 0; i < validation_set_num_; i++){
            int32_t tid = validation_ids[i];
            int32_t part_id = tid % partition_count;

            if(part_id < partition_count){
                (info->validation_set_ids[part_id]).push_back(tid);
            }
        }

        for(int32_t i = 0; i < testing_set_num_; i++){
            int32_t tid = testing_ids[i];
            int32_t part_id = tid % partition_count;
            
            if(part_id < partition_count){
                (info->testing_set_ids[part_id]).push_back(tid);
            }
        }
        free(partition_index);
    }else{
        printf("loading affinity root_node...\n");

        std::vector<std::string> filename(partition_count);
        for(int i = 0; i < partition_count; i++){
            filename[i] = "/work/liangzq/Qugion/preprocess/shuffle/" + std::to_string(i+1) + ".csv";
        }
        
        int local_train_num = training_set_num_ / partition_count;
        int local_valid_num = validation_set_num_ / partition_count;
        // int local_test_num = testing_set_num_ / partition_count;

        printf("local_train_num: %d\n", local_train_num - 1);
        // assign trainset root
        for(int i = 0; i < partition_count; i++){
            readCSVToIntArray(info->training_set_ids[i], filename[i], 0, local_train_num-1);
        }

        // assign validset root
        for(int i = 0; i < partition_count; i++){
            readCSVToIntArray(info->validation_set_ids[i], filename[i], local_train_num, local_train_num+local_valid_num-1);
        }

        // assign testset root
        for(int i = 0; i < partition_count; i++){
           readCSVToIntArray(info->testing_set_ids[i], filename[i], local_train_num+local_valid_num, -1);
        }

        printf("已导入亲和节点\n");
        for(int i = 0; i < partition_count; i++){
            printf("gpu%d number:\n", i);
            printf("%d, %d, %d\n", info->training_set_ids[i].size(), info->validation_set_ids[i].size(), info->testing_set_ids[i].size());
        }
    }

    //partition labels
    for(int32_t part_id = 0; part_id < partition_count; part_id++){
        for(int32_t i = 0; i < info->training_set_ids[part_id].size(); i++){
            int32_t ts_label = all_labels[info->training_set_ids[part_id][i]];
            info->training_labels[part_id].push_back(ts_label);
        }
        info->training_set_num.push_back(info->training_set_ids[part_id].size());
    }
    for(int32_t part_id = 0; part_id < partition_count; part_id++){
        for(int32_t i = 0; i < info->validation_set_ids[part_id].size(); i++){
            int32_t ts_label = all_labels[info->validation_set_ids[part_id][i]];
            info->validation_labels[part_id].push_back(ts_label);
        }
        info->validation_set_num.push_back(info->validation_set_ids[part_id].size());
    }
    for(int32_t part_id = 0; part_id < partition_count; part_id++){
        for(int32_t i = 0; i < info->testing_set_ids[part_id].size(); i++){
            int32_t ts_label = all_labels[info->testing_set_ids[part_id][i]];
            info->testing_labels[part_id].push_back(ts_label);
        }
        info->testing_set_num.push_back(info->testing_set_ids[part_id].size());
    }

    info->host_float_feature = host_float_feature;
    info->float_feature_len = float_feature_len_;
    info->total_num_nodes = node_num_;
}

void StorageManagement::Initialze(int32_t partition_count, int32_t in_memory_mode){

    in_memory_mode_ = in_memory_mode;

    BuildInfo* info = new BuildInfo();

    EnableP2PAccess();
    
    ConfigPartition(info, partition_count);

    ReadMetaFIle(info);

    LoadGraph(info);

    LoadFeature(info);

    env_ = NewIPCEnv(partition_count);
    env_ -> Coordinate(info);

    feature_ = NewCompleteFeatureStorage();
    feature_ -> Build(info, in_memory_mode_);   

    graph_ = NewCompleteGraphStorage();
    graph_ -> Build(info);

    cudaCheckError();

    cache_ = new UnifiedCache();

    int32_t train_step = env_->GetTrainStep();

    cudaSetDevice(0);
    cache_ -> Initialize(cache_memory_, float_feature_len_, train_step, partition_count, cpu_cache_capacity_, gpu_cache_capacity_);
    cudaSetDevice(0);
    std::cout<< "Storage Initialized" << std::endl;
}

GraphStorage* StorageManagement::GetGraph(){
    return graph_;
}

FeatureStorage* StorageManagement::GetFeature(){
    return feature_;
}

UnifiedCache* StorageManagement::GetCache(){
    return cache_;
}

IPCEnv* StorageManagement::GetIPCEnv(){
    return env_;
}
