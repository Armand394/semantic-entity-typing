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
                entities_in_sampled.add(parts[2])  
    
    # Open write all types for only entities still in graph
    with open(et_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[0] in entities_in_sampled:  
                fout.write(line)

# Make args dictionary for SEMDataset
def args_dict(data_dir, dataset, sample_et_size=0, sample_kg_size=0):
    args = {}
    args["sample_et_size"] = sample_et_size
    args["sample_kg_size"] = sample_kg_size
    args["data_dir"] = data_dir
    args["dataset"] = dataset
    return args

# === 1. CHARGEMENT DES FICHIERS TSV ===
def load_tsv(file_path):
    """Charge un fichier TSV sous forme de dictionnaire {id: nom}"""
    data_dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                data_dict[parts[1]] = parts[0]  
    return data_dict

# === 2. CHARGEMENT DU FICHIER KG_train.txt AVEC RELATIONS INVERSÉES ET FILTRAGE DES SELF-LOOPS ===
def load_kg(file_path):
    """Charge les relations sous forme de dictionnaire {entité: [(relation, objet)]}, sans self-loop"""
    kg = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                head, relation, tail = parts
                
                if head == tail:
                    continue  

                if head not in kg:
                    kg[head] = []
                kg[head].append((relation, tail))

                inverse_relation = "inv-" + relation
                if tail not in kg:
                    kg[tail] = []
                kg[tail].append((inverse_relation, head))
    return kg

# === 3. CHARGEMENT DU FICHIER ET_train.txt ET AJOUT DES ENTITÉS ===
def load_et(file_path):
    """Charge les types sous forme de dictionnaire {entité: [types]}"""
    et = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity, entity_type = parts
                if entity not in et:
                    et[entity] = []
                et[entity].append(entity_type)
    return et

# === 4. FILTRER LES ENTITÉS DE ET_test.txt ET ET_valid.txt ===
def filter_et_by_kg(et_dict, kg_dict):
    """Filtre les entités de et_dict pour ne garder que celles qui sont dans kg_dict"""
    return {entity: types for entity, types in et_dict.items() if entity in kg_dict}

# === 5. FONCTION D'EXTRACTION DU CLUSTER ===
def extract_cluster(type_str):
    """Extrait le cluster à partir du type donné en suivant les règles spécifiées"""
    parts = type_str.split("/")
    
    if type_str.startswith("/base/") and len(parts) > 2:
        return f"/base/{parts[2]}" 
    elif len(parts) > 1:
        return parts[1] 
    else:
        return "unknown"

# === 6. CONSTRUCTION DU FICHIER FINAL ===
def construct_output(kg_dict, et_train_dict, et_valid_dict, et_filter_dict, entite_dict, relation_dict, type_dict, cluster_dict, output_file, mode="train"):
    """Construit le fichier texte final avec la structure demandée, incluant les entités supplémentaires"""
    
    relation_0 = relation_dict.get("0", "0")  # Relation correspondant à l'ID 0
    entity_0 = entite_dict.get("0", "0")  # Entité correspondant à l'ID 0
    
    last_cluster = cluster_dict.get(list(cluster_dict.keys())[-1], "unknown")
    last_type = type_dict.get(list(type_dict.keys())[-1], "unknown")
    
    if mode == "train":
        all_entities = set(kg_dict.keys()).union(set(et_train_dict.keys())).union(set(et_valid_dict.keys()))
    else: 
        all_entities = set(et_filter_dict.keys())

    with open(output_file, "w", encoding="utf-8") as f:
        for entity in all_entities:
            entity_name = entite_dict.get(entity, entity)  

            types = et_train_dict.get(entity, []) + et_valid_dict.get(entity, [])
            types_part = " [SEP] ".join([ 
                f"{entity_name} {extract_cluster(type_dict.get(t, t))} {type_dict.get(t, t)}"
                for t in types
            ])

            if not types:
                types_part = f"{entity_name} {last_cluster} {last_type}"

            relations_part = " [SEP] ".join([ 
                f"{entity_name} {relation_dict.get(rel, rel)} {entite_dict.get(tail, tail)}"
                for rel, tail in kg_dict.get(entity, [])
                if entity != tail 
            ])

            if not relations_part:
                relations_part = f"{entity_name} {relation_0} {entity_0}"

            f.write(f"{entity_name} ||| {types_part} ||| {relations_part} ||| cluster \n")


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