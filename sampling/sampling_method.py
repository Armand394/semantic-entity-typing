import networkx as nx
import random
import os
from sampling_utils import *
from utils import read_id
from dataloader import SEMdataset

def main(dataset="FB15kET", sampled_folder="FB15kET_sample", generate_2hopsample=True):
    # Get data file paths
    project_path = os.getcwd()
    data_folder = os.path.join(project_path, "data")
    data_dir = os.path.join(project_path, "data", dataset)
    data_sample_dir = os.path.join(project_path, "data", f"{dataset}_sample")
    result_path = os.path.join(project_path, "data_entity_metrics")
    os.makedirs(data_sample_dir, exist_ok=True)

    # Load dictionaries
    e2id = read_id(os.path.join(data_dir, 'entities.tsv'))
    r2id = read_id(os.path.join(data_dir, 'relations.tsv'))
    t2id = read_id(os.path.join(data_dir, 'types.tsv'))
    c2id = read_id(os.path.join(data_dir, 'clusters.tsv'))

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

    # Save sample Graph
    edges = [(u, d["relation"], v, ) for u, v, d in G_sampled.edges(data=True)]
    save_triplets_to_txt(edges, file_path=os.path.join(data_sample_dir, "KG_train.txt"))

    kg_train_path = os.path.join(data_sample_dir, "KG_train.txt")

    # Filter train types with sampled entities in KG
    filter_types_data(kg_train_path, os.path.join(data_dir, "ET_train.txt"),
                    os.path.join(data_sample_dir, "ET_train.txt"))

    # Filter valid types with sampled entities in KG
    filter_types_data(kg_train_path, os.path.join(data_dir, "ET_valid.txt"),
                    os.path.join(data_sample_dir, "ET_valid.txt"))

    # Filter test types with sampled entities in KG
    filter_types_data(kg_train_path, os.path.join(data_dir, "ET_test.txt"),
                    os.path.join(data_sample_dir, "ET_test.txt"))

    # Debug print
    print("ET files with sampled graph saved")
    
    # Chargement des dictionnaires à partir des fichiers TSV
    entite_dict = load_tsv(os.path.join(data_dir, "entities.tsv"))
    relation_dict = load_tsv(os.path.join(data_dir, "relations.tsv"))
    type_dict = load_tsv(os.path.join(data_dir, "types.tsv"))
    cluster_dict = load_tsv(os.path.join(data_dir, "clusters.tsv"))

    # Chargement des données KG et ET
    kg_dict = load_kg(os.path.join(data_sample_dir,"KG_train.txt"))
    et_train_dict = load_et(os.path.join(data_sample_dir,"ET_train.txt"))
    et_valid_dict = load_et(os.path.join(data_sample_dir,"ET_valid.txt"))
    et_test_dict = load_et(os.path.join(data_sample_dir,"ET_test.txt"))

    # Construct train./valid .txt files for dataloading
    construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict,
                     output_file=os.path.join(data_sample_dir, "LMET_train.txt"), mode="train")
    construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict,
                     output_file=os.path.join(data_sample_dir, "LMET_valid.txt"), mode="train")

    # Construct test .txt files 
    filtered_et_test_dict = filter_et_by_kg(et_test_dict, kg_dict)
    construct_output(kg_dict, et_train_dict, et_valid_dict, filtered_et_test_dict, entite_dict, relation_dict, type_dict,
                     cluster_dict, output_file=os.path.join(data_sample_dir, "LMET_test.txt"), mode="test")
    
    print(".txt files with sampled graph saved")

    args = args_dict(dataset=f"{dataset}_sample", data_dir=data_folder)
    SEMdataset(args, "LMET_train.txt", e2id, r2id, t2id, c2id, 'train')
    SEMdataset(args, "LMET_test.txt", e2id, r2id, t2id, c2id, 'test')
    SEMdataset(args, "LMET_valid.txt", e2id, r2id, t2id, c2id, 'valid')
    
    print(".pkl files with sampled graph saved")

    # Copy other dataset files (no sampling)
    copy_other_kg_files(data_dir, data_sample_dir)

    if generate_2hopsample:
        print("Generate sample graph with 2-hop neighbors")
        data_sample_dir_2hop = os.path.join(project_path, "data", f"{dataset}_sample_2hop")
        os.makedirs(data_sample_dir_2hop, exist_ok=True)
        # Open files
        df_triples = pd.read_csv(os.path.join(data_sample_dir, "KG_train.txt"), sep='\t', header=None)
        df_train = pd.read_csv(os.path.join(data_sample_dir, "ET_train.txt"), sep='\t', header=None)
        
        df_train2hop = pd.read_csv(os.path.join(result_path, "KG_train_2hop.txt"), sep=' ', header=None)
        df_et2hop = pd.read_csv(os.path.join(result_path, "ET_train_2hop.txt"), sep=' ', header=None)

        sampled_entities = set(list(df_triples[0].unique()) + list(df_train[0].unique()) + list(df_triples[2].unique()))
        
        hop2_et_sample = df_et2hop[df_et2hop[0].isin(sampled_entities)]
        hop2_kg_sample = df_train2hop[(df_train2hop[0].isin(sampled_entities)) | (df_train2hop[2].isin(sampled_entities))]

        kg_new2hop = pd.concat([df_triples, hop2_kg_sample], axis=0)
        et_new2hop = pd.concat([df_train, hop2_et_sample], axis=0)

        kg_new2hop.to_csv(os.path.join(data_sample_dir_2hop,"KG_train.txt"),sep='\t', header=None, index=None)
        et_new2hop.to_csv(os.path.join(data_sample_dir_2hop,"ET_train.txt"), sep='\t', header=None, index=None)
        print("New 2-hop train files saved")

        # Load new train data with 2-hop knowledge
        kg_dict = load_kg(os.path.join(data_sample_dir_2hop,"KG_train.txt"))
        et_train_dict = load_et(os.path.join(data_sample_dir_2hop,"ET_train.txt"))

        # Construct train/valid .txt files for dataloading
        construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict,
                        output_file=os.path.join(data_sample_dir_2hop, "LMET_train.txt"), mode="train")
        construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict,
                        output_file=os.path.join(data_sample_dir_2hop, "LMET_valid.txt"), mode="train")

        # Construct test .txt files 
        filtered_et_test_dict = filter_et_by_kg(et_test_dict, kg_dict)
        construct_output(kg_dict, et_train_dict, et_valid_dict, filtered_et_test_dict, entite_dict, relation_dict, type_dict,
                        cluster_dict, output_file=os.path.join(data_sample_dir_2hop, "LMET_test.txt"), mode="test")

        args = args_dict(dataset=f"{dataset}_sample_2hop", data_dir=data_folder)

        SEMdataset(args, "LMET_train.txt", e2id, r2id, t2id, c2id, 'train')
        SEMdataset(args, "LMET_test.txt", e2id, r2id, t2id, c2id, 'test')
        SEMdataset(args, "LMET_valid.txt", e2id, r2id, t2id, c2id, 'valid')
        
        print(".pkl files with sampled graph saved")
       
       # Copy other dataset files (no sampling)
        copy_other_kg_files(data_sample_dir, data_sample_dir_2hop)

if __name__ == '__main__':
    main()
