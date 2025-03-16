import networkx as nx
import random
import os

def load_graph_from_txt(file_path):
    G = nx.DiGraph() 
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                head, relation, tail = parts
                G.add_edge(head, tail, relation=relation)
    return G


def directed_random_walk_sampling(graph, seed_nodes, walk_length=10, sample_size=20000, p_restart=0.7):

    sampled_nodes = set(seed_nodes)

    for node in seed_nodes:
        current_node = node
        for _ in range(walk_length):
            if random.random() < p_restart:
                current_node = random.choice(seed_nodes)  # 以一定概率重置到起始节点
            else:
                neighbors = list(graph.successors(current_node))  # 只看出边
                if not neighbors:
                    break  # 没有出边就终止当前游走
                current_node = random.choice(neighbors)

            sampled_nodes.add(current_node)
            if len(sampled_nodes) >= sample_size:
                break
        if len(sampled_nodes) >= sample_size:
            break

    return graph.subgraph(sampled_nodes)


project_dir = os.getcwd()
data_dir = os.path.join(project_dir, "data", "FB15kET")
data_sample_dir = os.path.join(project_dir, "data", "FB15kET_sample")

# Load relationships
G = load_graph_from_txt(os.path.join(data_dir, "KG_train.txt"))

# Sample random nodes with highest degree
num_hubs = 300 
high_outdegree_nodes = sorted(G.out_degree, key=lambda x: x[1], reverse=True)[:num_hubs]
hub_seeds = [node for node, _ in high_outdegree_nodes]

# Sample additional random nodes
valid_seeds = [node for node in G.nodes() if len(list(G.successors(node))) > 0]
seed_nodes1 = random.sample(valid_seeds, 500)  
seed_nodes = hub_seeds + seed_nodes1

# Choose parameters for random-walk sampling of Graph
G_sampled = directed_random_walk_sampling(G, seed_nodes, walk_length=5, sample_size=10000, p_restart=0.7)

# Debug print
print(f"Graph sampled has {G_sampled.number_of_nodes()} nodes and {G_sampled.number_of_edges()} edges")
print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Save results
def save_triplets_to_txt(edges, file_path="triplets.txt"):
    with open(file_path, "w", encoding="utf-8") as f:
        for head, relation, tail in edges:
            f.write(f"{head}\t{relation}\t{tail}\n")

# Save sample Graph
edges = [(u, d["relation"], v, ) for u, v, d in G_sampled.edges(data=True)]
save_triplets_to_txt(edges, file_path=os.path.join(data_sample_dir, "KG_train.txt"))


def filter_types_data(sampled_triplets_file, et_train_file, output_file):
    entities_in_sampled = set()
    with open(sampled_triplets_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                entities_in_sampled.add(parts[0])  
    
    with open(et_train_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[0] in entities_in_sampled:  
                fout.write(line)

# Filter train types with sampled entities
filter_types_data(os.path.join(data_sample_dir, "KG_train.txt"), os.path.join(data_dir, "ET_train.txt"),
                os.path.join(data_sample_dir, "ET_train.txt"))

# Filter valid types with sampled entities
filter_types_data(os.path.join(data_sample_dir, "KG_train.txt"), os.path.join(data_dir, "ET_valid.txt"),
                os.path.join(data_sample_dir, "ET_valid.txt"))

# Filter test types with sampled entities
filter_types_data(os.path.join(data_sample_dir, "KG_train.txt"), os.path.join(data_dir, "ET_test.txt"),
                os.path.join(data_sample_dir, "ET_test.txt"))


