import networkx as nx
import random
import os
from tqdm import tqdm
import pandas as pd
import pickle
import shutil
import numpy as np

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
def sample_type_train(sampled_triplets_file, kg_ratios, et_file, output_file):
    # Open KG triples to acquire all entities in sampled graph
    kg_df = pd.read_csv(sampled_triplets_file, sep='\t', header=None)  
    entities_sampled = set(kg_df[0].unique()).union(set(kg_df[2].unique()))
    
    # Open et train and filter existent entities in kg
    df_train = pd.read_csv(et_file, sep='\t', header=None)
    df_train = df_train[df_train[0].isin(entities_sampled)]
    
    # Dictionary of original type/kg ratios
    et_ratios = kg_ratios.set_index('entity').to_dict()['type_kg_ratio']

    # Create type-log occurences dictionary
    df_train_types = df_train.groupby(1).count().reset_index(drop=False)
    df_train_types.columns = ['type_train', 'count']
    df_train_types['count'] = np.log1p(df_train_types['count'])
    et_counts = df_train_types.set_index('type_train').to_dict()['count']

    # write on new type file --> only sampled entities & maintain ratios
    with open(output_file, "w") as f1:
        for entity in tqdm(entities_sampled, desc='Sampling train types'):
            # Get KG degree and current types
            kg_degree = kg_df[(kg_df[0] == entity) | (kg_df[2] == entity)].shape[0]
            types = df_train[df_train[0] == entity][1].unique()

            # Compute two type/kg ratios 
            origin_ratio = et_ratios[entity]

            # total count of types --> use for sampling probability (high occurency <-> high probability)
            counts_types = np.array([et_counts.get(t, 0) for t in types])
            sampling_probs = counts_types / counts_types.sum()

            # Sample types such that new ratio is similar to original (with margin of 0.1)
            k_sample_types = min(int((origin_ratio+0.1)*kg_degree), len(types))
            
            if k_sample_types != len(types):
                # Sample without replacement using probabilities
                types = np.random.choice(types, size=k_sample_types, replace=False, p=sampling_probs)

            for type in types:
                line_e = f"{entity}\t{type}\n"
                f1.write(line_e)


def sample_valid_test_types(kg_train_path, data_dir, data_sample_dir):
    # Open KG file and select sampled entities
    kg_df = pd.read_csv(kg_train_path, sep='\t', header=None)
    entities_sampled = set(kg_df[0].unique()).union(set(kg_df[2].unique()))
        
    # Open Type files
    df_valid = pd.read_csv(os.path.join(data_dir, "ET_valid.txt"), sep='\t', header=None)
    df_test = pd.read_csv(os.path.join(data_dir, "ET_test.txt"), sep='\t', header=None)

    # Filter only for existing entities
    df_train_sample = pd.read_csv(os.path.join(data_sample_dir, "ET_train.txt"), sep='\t', header=None)
    df_valid_sample = df_valid[df_valid[0].isin(entities_sampled)]
    df_test_sample = df_test[df_test[0].isin(entities_sampled)]

    # Filter for only types existing in train types
    df_valid_sample = df_valid_sample[df_valid_sample[1].isin(df_train_sample[1].unique())]
    df_test_sample = df_test_sample[df_test_sample[1].isin(df_train_sample[1].unique())]

    # Save sampled valid/test files
    df_valid_sample.to_csv(os.path.join(data_sample_dir,"ET_valid.txt"), sep='\t', header=False, index=False)
    df_test_sample.to_csv(os.path.join(data_sample_dir,"ET_test.txt"), sep='\t', header=False, index=False)


def create_tsv_files(data_sample_dir):
    # Open KG triples to acquire all entities in sampled graph
    kg_df = pd.read_csv(os.path.join(data_sample_dir, "KG_train.txt"), sep='\t', header=None)  

    # Open et train and filter existent entities in kg
    df_train = pd.read_csv(os.path.join(data_sample_dir, "ET_train.txt"), sep='\t', header=None)

    entities_sampled = set(kg_df[0].unique()).union(set(kg_df[2].unique()))
    relations_sampled = set(kg_df[1].unique())
    types_sampled = set(df_train[1].unique())

    clusters = set()

    for type_name in types_sampled:
        if len(type_name.split('/')) < 2:
            clusters.add(type_name)
            continue

        cluster_name = type_name.split('/')

        if cluster_name[1] == 'base':
            clusters.add(f"/{cluster_name[1]}/{cluster_name[2]}")
            continue

        clusters.add(cluster_name[1])

    # Convert sets to sorted lists
    entities = sorted(entities_sampled)
    relations = sorted(relations_sampled)
    types = sorted(types_sampled)
    clusters = sorted(clusters)

    # Create DataFrames with incremental IDs starting from 0
    df_entities = pd.DataFrame({'entity': entities, 'id': range(len(entities))})
    df_relations = pd.DataFrame({'relation': relations, 'id': range(len(relations))})
    df_types = pd.DataFrame({'type': types, 'id': range(len(types))})
    df_clusters = pd.DataFrame({'cluster': clusters, 'id': range(len(clusters))})

    # Save as .tsv files
    df_entities.to_csv(os.path.join(data_sample_dir, 'entities.tsv'), sep='\t', index=False, header=False)
    df_relations.to_csv(os.path.join(data_sample_dir, 'relations.tsv'), sep='\t', index=False, header=False)
    df_types.to_csv(os.path.join(data_sample_dir, 'types.tsv'), sep='\t', index=False, header=False)
    df_clusters.to_csv(os.path.join(data_sample_dir, 'clusters.tsv'), sep='\t', index=False, header=False)


def filter_description_files(data_dir, data_sample_dir):
    # Desciprtion files needed to be filtered 
    r2text = pd.read_csv(os.path.join(data_dir, "relation2text.txt"), sep='\t', header=None)
    t2desc = pd.read_csv(os.path.join(data_dir, "hier_type_desc.txt"), sep='\t', header=None)

    kg_sample = pd.read_csv(os.path.join(data_sample_dir, "KG_train.txt"), sep='\t', header=None)  
    et_sample = pd.read_csv(os.path.join(data_sample_dir, "ET_train.txt"), sep='\t', header=None)  

    sampled_rel = kg_sample[1].unique()
    r2text = r2text[r2text[0].isin(sampled_rel)]

    sampled_type = et_sample[1].unique()
    t2desc = t2desc[t2desc[0].isin(sampled_type)]

    r2text.to_csv(os.path.join(data_sample_dir, "relation2text.txt"), sep='\t', header=None, index=False)
    t2desc.to_csv(os.path.join(data_sample_dir, "hier_type_desc.txt"), sep='\t', header=None, index=False)

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