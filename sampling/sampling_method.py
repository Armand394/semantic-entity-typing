import networkx as nx
import random
import os
from sampling_utils import *
from utils import read_id
from dataloader import SEMdataset
import numpy as np
import json

dataset = "FB15kET"

# Get data file paths
project_path = os.getcwd()
data_folder = os.path.join(project_path, "data")
data_dir = os.path.join(project_path, "data", dataset)
data_sample_dir = os.path.join(project_path, "data", f"{dataset}_sample")
result_path = os.path.join(project_path, "data_entity_metrics")

if not os.path.exists(data_sample_dir):
    os.makedirs(data_sample_dir, exist_ok=True)

    # Load relationships
    G = load_graph_from_txt(os.path.join(data_dir, "KG_train.txt"))

    # Sample random nodes with highest degree
    num_hubs = 400 
    high_outdegree_nodes = sorted(G.out_degree, key=lambda x: x[1], reverse=True)[:num_hubs]
    hub_seeds = [node for node, _ in high_outdegree_nodes]

    # Sample additional random nodes
    valid_seeds = [node for node in G.nodes() if len(list(G.successors(node))) > 0]
    seed_nodes1 = random.sample(valid_seeds, 800)  
    seed_nodes = hub_seeds + seed_nodes1

    # Choose parameters for random-walk sampling of Graph
    G_sampled = directed_random_walk_sampling(G, seed_nodes, walk_length=7, sample_size=100000, p_restart=0.7)

    # Save sample Graph
    edges = [(u, d["relation"], v, ) for u, v, d in G_sampled.edges(data=True)]
    save_triplets_to_txt(edges, file_path=os.path.join(data_sample_dir, "KG_train.txt"))

    # Debug print
    print(f"Graph sampled has {G_sampled.number_of_nodes()} nodes and {G_sampled.number_of_edges()} edges")
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Open kg and ratios
    kg_train_path = os.path.join(data_sample_dir, "KG_train.txt")
    kg_ratios = pd.read_csv(os.path.join(result_path, "type-kg-ratios.csv"))

    # Filter train types with sampled entities in KG
    sample_type_train(kg_train_path, kg_ratios, os.path.join(data_dir, "ET_train.txt"),
                    os.path.join(data_sample_dir, "ET_train.txt"))

    # Sample valid/test files based on existing entities and types in sampled train files
    sample_valid_test_types(kg_train_path, data_dir, data_sample_dir)
    
    # Debug print
    print("ET files with sampled graph saved")
    
    # Create tsv files with sampled entities, relations, types, and clusters
    create_tsv_files(data_sample_dir)

    # Filter description files - only description of sampled relationships and types
    filter_description_files(data_dir, data_sample_dir)

    # Load dictionaries id -> entity
    entite_dict = load_tsv(os.path.join(data_sample_dir, "entities.tsv"))
    relation_dict = load_tsv(os.path.join(data_sample_dir, "relations.tsv"))
    type_dict = load_tsv(os.path.join(data_sample_dir, "types.tsv"))
    cluster_dict = load_tsv(os.path.join(data_sample_dir, "clusters.tsv"))

    # Chargement des données KG et ET
    kg_dict = load_kg(os.path.join(data_sample_dir,"KG_train.txt"))
    et_train_dict = load_et(os.path.join(data_sample_dir,"ET_train.txt"))
    et_valid_dict = load_et(os.path.join(data_sample_dir,"ET_valid.txt"))
    et_test_dict = load_et(os.path.join(data_sample_dir,"ET_test.txt"))

    # Construct LMET_train.txt files for dataloading
    construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict,
                    output_file=os.path.join(data_sample_dir, "LMET_train.txt"), mode="train")
    # Construct LMET_valid.txt files for dataloading
    construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict,
                    output_file=os.path.join(data_sample_dir, "LMET_valid.txt"), mode="train")
    # Construct LMET_test.txt files for dataloading
    construct_output(kg_dict, et_train_dict, et_valid_dict, et_test_dict, entite_dict, relation_dict, type_dict,
                    cluster_dict, output_file=os.path.join(data_sample_dir, "LMET_test.txt"), mode="test")

    print(".txt files with sampled graph saved")

    # Load dictionaries entity -> id
    e2id = read_id(os.path.join(data_sample_dir, 'entities.tsv'))
    r2id = read_id(os.path.join(data_sample_dir, 'relations.tsv'))
    t2id = read_id(os.path.join(data_sample_dir, 'types.tsv'))
    c2id = read_id(os.path.join(data_sample_dir, 'clusters.tsv'))

    args = args_dict(dataset=f"{dataset}_sample", data_dir=data_folder)
    SEMdataset(args, "LMET_train.txt", e2id, r2id, t2id, c2id, 'train')
    SEMdataset(args, "LMET_test.txt", e2id, r2id, t2id, c2id, 'test')
    SEMdataset(args, "LMET_valid.txt", e2id, r2id, t2id, c2id, 'valid')

    # Remove unnecessary .txt files
    os.remove(os.path.join(data_sample_dir, "LMET_train.txt"))
    os.remove(os.path.join(data_sample_dir, "LMET_valid.txt"))
    os.remove(os.path.join(data_sample_dir, "LMET_test.txt"))

    print(".pkl files with sampled graph saved")

    # Copy other dataset files (no sampling)
    copy_other_kg_files(data_dir, data_sample_dir)

generate_2hopsample = True

if generate_2hopsample:
    print("Generate sample graph with 2-hop neighbors")
    data_sample_dir_2hop = os.path.join(project_path, "data", f"{dataset}_sample_2hop")
    os.makedirs(data_sample_dir_2hop, exist_ok=True)

    # Chargement des dictionnaires à partir des fichiers TSV
    entite_dict = load_tsv(os.path.join(data_sample_dir, "entities.tsv"))
    relation_dict = load_tsv(os.path.join(data_sample_dir, "relations.tsv"))
    type_dict = load_tsv(os.path.join(data_sample_dir, "types.tsv"))
    cluster_dict = load_tsv(os.path.join(data_sample_dir, "clusters.tsv"))

    # Chargement des données KG et ET
    kg_dict = load_kg(os.path.join(data_sample_dir,"KG_train.txt"))
    et_train_dict = load_et(os.path.join(data_sample_dir_2hop,"ET_train.txt"))
    et_valid_dict = load_et(os.path.join(data_sample_dir,"ET_valid.txt"))
    et_test_dict = load_et(os.path.join(data_sample_dir,"ET_test.txt"))

    # Load processing dictionaries
    with open(os.path.join(data_sample_dir_2hop,"relation2hop.json"), "r") as f:
        r2hop = json.load(f)
    with open(os.path.join(data_sample_dir_2hop,"relation2remove.json"), "r") as f:
        r2move = json.load(f)

    # Construct LMET_train.txt files for dataloading
    construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict,
                    output_file=os.path.join(data_sample_dir_2hop, "LMET_train.txt"), mode="train", kg_dict2=r2hop, kg_remove=r2move)
    # Construct LMET_valid.txt files for dataloading
    construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict,
                    output_file=os.path.join(data_sample_dir_2hop, "LMET_valid.txt"), mode="train", kg_dict2=r2hop, kg_remove=r2move)
    # Construct LMET_test.txt files for dataloading
    construct_output(kg_dict, et_train_dict, et_valid_dict, et_test_dict, entite_dict, relation_dict, type_dict,
                    cluster_dict, output_file=os.path.join(data_sample_dir_2hop, "LMET_test.txt"), mode="test", kg_dict2=r2hop, kg_remove=r2move)

    # Load dictionaries entity -> id
    e2id = read_id(os.path.join(data_sample_dir, 'entities.tsv'))
    r2id = read_id(os.path.join(data_sample_dir, 'relations.tsv'))
    t2id = read_id(os.path.join(data_sample_dir, 'types.tsv'))
    c2id = read_id(os.path.join(data_sample_dir, 'clusters.tsv'))

    args = args_dict(dataset=f"{dataset}_sample_2hop", data_dir=data_folder)

    SEMdataset(args, "LMET_train.txt", e2id, r2id, t2id, c2id, 'train')
    SEMdataset(args, "LMET_test.txt", e2id, r2id, t2id, c2id, 'test')
    SEMdataset(args, "LMET_valid.txt", e2id, r2id, t2id, c2id, 'valid')

    # Remove unnecessary .txt files
    os.remove(os.path.join(data_sample_dir_2hop, "LMET_train.txt"))
    os.remove(os.path.join(data_sample_dir_2hop, "LMET_valid.txt"))
    os.remove(os.path.join(data_sample_dir_2hop, "LMET_test.txt"))

    print(".pkl files with sampled graph saved")

    # Copy other dataset files (no sampling)
    copy_other_kg_files(data_sample_dir, data_sample_dir_2hop)