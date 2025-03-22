import networkx as nx
import random
import os
from tqdm import tqdm
import pandas as pd
import pickle
import shutil

tqdm.pandas() 


# Function to load a directed graph from a text file
def load_graph_from_txt(file_path):
    G = nx.DiGraph()  # Create a directed graph
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")  # Split line by tab
            if len(parts) == 3:  # Ensure the line has exactly three parts
                head, relation, tail = parts  # Unpack the values
                G.add_edge(head, tail, relation=relation)  # Add a directed edge with relation as an attribute
    return G

# Function to perform a directed random walk sampling
def directed_random_walk_sampling(graph, seed_nodes, walk_length=10, sample_size=20000, p_restart=0.7):
    
    sampled_nodes = set(seed_nodes)  # Initialize the sampled nodes set with seed nodes
    
    # Iterate over each seed node to start a random walk
    for node in seed_nodes:
        current_node = node  # Start walk from the seed node
        
        for _ in range(walk_length):  # Perform a random walk of given length
            if random.random() < p_restart:  # With probability p_restart, restart the walk from a seed node
                current_node = random.choice(seed_nodes)
            else:
                neighbors = list(graph.successors(current_node))  # Get all outgoing neighbors
                if not neighbors:
                    break  # Stop walk if there are no outgoing edges
                current_node = random.choice(neighbors)  # Randomly choose a neighbor as the next node
            
            sampled_nodes.add(current_node)  # Add the visited node to the sampled set
            
            if len(sampled_nodes) >= sample_size:  # Stop if the sample size is reached
                break
        
        if len(sampled_nodes) >= sample_size:  # Stop the whole process if sample size is reached
            break

    return graph.subgraph(sampled_nodes)  # Return the subgraph induced by sampled nodes

# Save new sampled KG_train result
def save_triplets_to_txt(edges, file_path="triplets.txt"):
    with open(file_path, "w", encoding="utf-8") as f:
        for head, relation, tail in edges:
            f.write(f"{head}\t{relation}\t{tail}\n")

# Save new sampled KG_train result
def filter_types_data(sampled_triplets_file, et_file, output_file):
    entities_in_sampled = set()
    # Open KG triples to acquire all entities in sampled graph
    with open(sampled_triplets_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                entities_in_sampled.add(parts[0])  
    
    # Open write all types for only entities still in graph
    with open(et_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[0] in entities_in_sampled:  
                fout.write(line)


def get_cluster_name_from_type_FB(type_name):
    if len(type_name.split('/')) < 2:
        return type_name
    
    cluster_name = type_name.split('/')
    if cluster_name[1] == 'base':
        return f"/{cluster_name[1]}/{cluster_name[2]}"
    
    return cluster_name[1]

def get_cluster_name_from_type_YG(type_name):
    if len(type_name.split('_')) < 2:
        return type_name
    cluster_name = type_name.split('_')[1]
    return cluster_name


def convert_to_pkl(KG_path, KG_file, ET_file, output_file, get_clust_name_fonc=get_cluster_name_from_type_FB):
    """
    @brief convert a knowledge graph into a pkl accepted by SSET
    @param KG_path : folder where clusters.tsv, entities.tsv and type.tsv are present
    @param KG_file : KG file to convert (Formatted as -> entity_A\trel\tentity_B)
    @param ET_file : ET file to convert (Formatted as -> entity\ttype)
    @param get_clust_name_fonc : function to get cluster from type must take type and df_clust as input
    @param output_file : pkl generated
    """
    rel_path = os.path.join(KG_path, 'relations.tsv')
    df_rel = pd.read_csv(rel_path, sep='\t', header=None, names=['relation', 'id'])
    rel_count = len(df_rel)

    ent_path = os.path.join(KG_path, 'entities.tsv')
    df_ent = pd.read_csv(ent_path, sep='\t', header=None, names=['entity', 'id'])
    ent_count = len(df_ent)

    type_path = os.path.join(KG_path, 'types.tsv')
    df_type = pd.read_csv(type_path, sep='\t', header=None, names=['type', 'id'])
    df_type['id'] += ent_count

    clust_path = os.path.join(KG_path, 'clusters.tsv')
    df_clust = pd.read_csv(clust_path, sep='\t', header=None, names=['cluster', 'id'], keep_default_na=False)
    df_clust['id'] += rel_count
    
    df_kg = pd.read_csv(KG_file, sep='\t', header=None, names=['entity', 'relation', 'entity_2'])

    df_kg_id = (
        df_kg.merge(df_ent, on="entity")
            .rename(columns={"id": "id_1", "entity": "entity_1", "entity_2": "entity"})
            .merge(df_ent, on="entity")
            .rename(columns={"id": "id_2", "entity": "entity_2"})
            .merge(df_rel, on="relation")
            .rename(columns={"id": "rel_id"})
            [["id_1", "rel_id", "id_2"]]
    )
    df_et = pd.read_csv(ET_file, sep='\t', header=None, names=['entity', 'type'])
    
    df_et_id = (
        df_et.merge(df_type.assign(clust_id=df_type['type'].progress_apply(lambda t: df_clust[df_clust['cluster'] == get_clust_name_fonc(t)]['id'].iloc[0])), on='type')
        .rename(columns={'id': 'type_id'})[['entity', 'clust_id', 'type_id']]
        .merge(df_ent, on='entity')
        .rename(columns={'id': 'ent_id'})[['ent_id', 'clust_id', 'type_id']]
    )

    triplets_list = [
    (df_et_id[(df_et_id['ent_id'] == ent_id)].values.tolist()
     , df_kg_id[(df_kg_id['id_1'] == ent_id) | (df_kg_id['id_2'] == ent_id)].values.tolist()
     , ent_id)    
    for ent_id in tqdm(df_ent['id'])
    ]

    if output_file is not None:
        with open(output_file, 'wb') as f:
            pickle.dump(triplets_list, f)

    return triplets_list


def filter_pkl(pkl_file, entities):
    # Open the original pickle file in read-binary mode
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    # Filter out empty (et, kg) tuples
    filtered_data = [(et, kg, eid) for et, kg, eid in data if eid in entities]

    # Save the filtered data to a new pickle file in write-binary mode
    with open(pkl_file, "wb") as f:
        pickle.dump(filtered_data, f)


def copy_other_kg_files(source_folder, destination_folder):

    # Loop over all items in the source folder
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        
        # Check if the current item is a file
        if os.path.isfile(source_file):
            # Copy the file only if it doesn't already exist in the destination
            if not os.path.exists(destination_file):
                shutil.copy2(source_file, destination_file)
                print(filename)
