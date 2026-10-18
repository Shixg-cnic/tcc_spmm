import os 
import argparse 
import subprocess
import re
import networkx as nx
import math

def parse_topo_output(output, gpu_number):
    """
    Parses the output from `nvidia-smi topo -m` to extract the NVLink connections between GPUs.
    This function is adjusted based on your provided example output.
    """
    connections = []
    lines = output.splitlines()
    gpu_lines = [line for line in lines if line.startswith("GPU")]
    for i, line in enumerate(gpu_lines):
        elements = line.split()
        for j, elem in enumerate(elements[1:], start=0):  # Start from the first GPU column
            if elem.startswith("NV"):
                connections.append((i, j))

    # 自定义 gpu 数量
    connections = []
    for i in range(0, gpu_number):
        for j in range(0, gpu_number):
            if i != j:
                connections.append((i, j))

    return connections

def get_nvlink_topology(gpu_number):
    # Execute the `nvidia-smi topo -m` command to get the topology matrix
    result = subprocess.run(['nvidia-smi', 'topo', '-m'], stdout=subprocess.PIPE, text=True)
    connections = parse_topo_output(result.stdout, gpu_number)
    return connections

def find_largest_fully_connected_group(G):
    """
    Finds the largest fully connected group (clique) in the graph and returns its size
    and the list of such groups if there are multiple of the same size.
    """
    cliques = list(nx.find_cliques(G))
    max_size = max(len(clique) for clique in cliques) if cliques else 1
    max_cliques = [clique for clique in cliques if len(clique) == max_size]
    return max_size, max_cliques
        
"""
path =  args.dataset_path + "/products/"
vertices_num = 2449029
edges_num = 123718280
features_dim = 100
train_set_num = 196615
valid_set_num = 39323
test_set_num = 2213091

path =  args.dataset_path + "/products/"
vertices_num = 2449029
edges_num = 61859140
features_dim = 100
train_set_num = 2213091
valid_set_num = 196615
test_set_num = 39323
"""
def Run(args):

    """
    关于数据处理的注意事项：
    我们需要首先理解的是，为了减少数据占用量，无向图数据往往只存储单向边，但是我们的实际计算为了得到完整的信息传输链，是需要手动添加反向边的;
    对于有向图，理论上不需要考虑添加反向边，因为有向图本身就已经规定了信息传播的方向，但是因为当时认为 dgl 会自动添加反向边，所以在格式转换中我们会添加反向边
    所以，在实际生成数据的过程中，有向图和无向图的处理规则是相同的，即直接按照给定数据输出为 csr 格式即可，不需要额外添加反向边
    不过对于 edges_num 为了保证空间开辟足够，所以按照单向边*2开辟即可，由于无论是有向图还是无向图我们都会添加反向边，所以有向图也需要将此值*2

    输入数据为 csr 结构，并且 row_offsets 为 8B，col_indices 为 4B
    """

    # 当时好像对这块的数据集进行了修改，节点数量和原始值并不相同，目前还是先采用原始值了，修改后的情况见 Legion_fuse
    # 无向图(需要从 coo.csv 自己构建 csr)，单向边是 61859140；原始数据存在 128 条自环

    # 没有添加反向边的数据可以正常跑 800 和 8000
    # Number of nodes (vertices): 2400608(? 这里有点问题，为什么点数还少了)
    # Number of edges (stored in CSR): 61859012（这是去掉自环后的边数）

    # 对于添加了反向边的数据无法跑 8000，可以跑 800
    # Number of nodes (vertices): 2449029
    # Number of edges (stored in CSR): 123718152（添加反向边并去自环）

    if args.dataset_name == "products":
        path =  args.dataset_path + "/products/"
        vertices_num = 2449029
        edges_num = 123718280   
        features_dim = 100
        train_set_num = 2213091
        valid_set_num = 196615
        test_set_num = 39323
    
    # 有向图（直接从 npz 中提取数据）
    # Number of nodes (vertices): 111059956
    # Number of edges (stored in CSR): 1615685872
    elif args.dataset_name == "paper100m":
        path = args.dataset_path + "/paper100M/"
        vertices_num = 111059956
        edges_num = 1615685872
        features_dim = 128
        train_set_num = 11105995
        valid_set_num = 100000
        test_set_num = 100000
    
    # 无向图(需要从 coo 自己构建 csr)
    # 单向边是 1806067135; 无自环
    elif args.dataset_name == "com-friendster":
        path = args.dataset_path + "/com-friendster/"
        vertices_num = 65608366
        edges_num = 3612134270  
        features_dim = 256
        train_set_num = 6560836
        valid_set_num = 100000
        test_set_num = 100000

    # 有向图（利用 legion 提供的脚本生成）
    # Number of nodes (vertices): 131814559
    # Number of edges (stored in CSR): 5475109924
    elif args.dataset_name == "ukunion":
        path = args.dataset_path + "/ukunion/"
        vertices_num = 133633040
        edges_num = 5507679822
        features_dim = 256
        train_set_num = 13363304
        valid_set_num = 100000
        test_set_num = 100000
    
    elif args.dataset_name == "uk2014":
        path = args.dataset_path + "/uk2014/"
        vertices_num = 787801471
        edges_num = 47284178505
        features_dim = 128
        train_set_num = 78780147
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "clueweb":
        path = args.dataset_path + "/clueweb/"
        vertices_num = 955207488
        edges_num = 42574107469
        features_dim = 128
        train_set_num = 95520748
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "arxiv":
        path = args.dataset_path + "/arxiv/"
        vertices_num = 169343	
        edges_num = 1166243
        features_dim = 128
        train_set_num = 161343
        valid_set_num = 4000
        test_set_num = 4000
    elif args.dataset_name == "mag":
        path = args.dataset_path + "/mag/"
        vertices_num = 736389	
        edges_num = 5416271
        features_dim = 128
        train_set_num = 728389
        valid_set_num = 4000
        test_set_num = 4000
    else:
        print("invalid dataset path")
        exit
    

    with open("meta_config","w") as file:
        file.write("{} {} {} {} {} {} {} {} {} {}".format(path, args.train_batch_size, vertices_num, edges_num, features_dim, train_set_num, valid_set_num, test_set_num, args.cache_memory, args.epoch))

    gpu_number = args.gpu_number
    
    if args.usenvlink == 1:
        connections = get_nvlink_topology(gpu_number)
        G = nx.Graph()
        G.add_edges_from(connections)
        group_size, fully_connected_groups = find_largest_fully_connected_group(G)
        print(f"gpu_number:{gpu_number},group_size:{group_size},fully_connected_groups:{fully_connected_groups}", flush=True)
        if fully_connected_groups or group_size == 1:
            print(f"NVLink clique size: {group_size}, Number of NVLink cliques: {int(gpu_number/group_size)}")
        cache_agg_mode = math.log2(group_size)
    else:
        cache_agg_mode = 0

    print("start parameter-1: {}, parameter-2: {}".format(gpu_number, cache_agg_mode), flush=True)
    #os.system("/work/liangzq/cuda/cuda-12.1/bin/nsys profile --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop -o /work/gaohy/Legion_fuse/profile/profile-report ./sampling_server/build/bin/sampling_server {} {}".format(gpu_number, cache_agg_mode))
    #os.system("sudo /work/gaohy/usr/local/cuda-12.3/bin/cuda-gdb --silent --args ./sampling_server/build/bin/sampling_server {} {}".format(gpu_number, cache_agg_mode))
    os.system("./sampling_server/build/bin/sampling_server {} {}".format(gpu_number, cache_agg_mode))
    #os.system("CUDA_LAUNCH_BLOCKING=1 cuda-gdb --silent --args ./sampling_server/build/bin/sampling_server {} {}".format(gpu_number, cache_agg_mode))
    ## TODO, integrate Legion server in python module


if __name__ == "__main__":

    print("========================= legion_server.py begin =========================", flush=True)

    argparser = argparse.ArgumentParser("Legion Server.")
    argparser.add_argument('--dataset_path', type=str, default="./dataset")
    argparser.add_argument('--dataset_name', type=str, default="ukunion")
    argparser.add_argument('--train_batch_size', type=int, default=8000)
    argparser.add_argument('--fanout', type=list, default=[25, 10])
    argparser.add_argument('--gpu_number', type=int, default=2)
    argparser.add_argument('--epoch', type=int, default=2)
    argparser.add_argument('--cache_memory', type=int, default=38000000)
    argparser.add_argument('--usenvlink', type=int, default=1)
    args = argparser.parse_args()

    Run(args)
