import networkx as nx
import random
import os
from sampling_utils import *
from utils import read_id

def main(dataset="FB15kET", sampled_folder="FB15kET_sample", generate_2hopsample=True):
    # Get data file paths
    project_path = os.getcwd()
    data_dir = os.path.join(project_path, "data", dataset)
    data_sample_dir = os.path.join(project_path, "data", f"{dataset}_sample")
    result_path = os.path.join(project_path, "data_entity_metrics")
    os.makedirs(data_sample_dir, exist_ok=True)

    # Load relationships
    G = load_graph_from_txt(os.path.join(data_dir, "KG_train.txt"))
    e2id = read_id(os.path.join(data_dir, 'entities.tsv'))

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


    # Filter train types with sampled entities in KG
    filter_types_data(os.path.join(data_sample_dir, "KG_train.txt"), os.path.join(data_dir, "ET_train.txt"),
                    os.path.join(data_sample_dir, "ET_train.txt"))

    # Filter valid types with sampled entities in KG
    filter_types_data(os.path.join(data_sample_dir, "KG_train.txt"), os.path.join(data_dir, "ET_valid.txt"),
                    os.path.join(data_sample_dir, "ET_valid.txt"))

    # Filter test types with sampled entities in KG
    filter_types_data(os.path.join(data_sample_dir, "KG_train.txt"), os.path.join(data_dir, "ET_test.txt"),
                    os.path.join(data_sample_dir, "ET_test.txt"))

    # Debug print
    print("ET files with sampled graph saved")
    
    # Create pkl files (unfiltered)
    kg_train_path = os.path.join(data_sample_dir, "KG_train.txt")

    # Get cluster id function
    if dataset == "FB15kET":
        get_clust_name_fonc = get_cluster_name_from_type_FB
    else:
        get_clust_name_fonc = get_cluster_name_from_type_YG

    # Convert to pkl
    triplets_list = convert_to_pkl(data_dir, kg_train_path, os.path.join(data_sample_dir,'ET_train.txt')
                                    , os.path.join(data_sample_dir,'LMET_train.pkl'), get_clust_name_fonc=get_clust_name_fonc)

    triplets_list = convert_to_pkl(data_dir, kg_train_path, os.path.join(data_sample_dir,'ET_train.txt')
                                    , os.path.join(data_sample_dir,'LMET_test.pkl'))

    triplets_list = convert_to_pkl(data_dir, kg_train_path, os.path.join(data_sample_dir,'ET_train.txt')
                                    , os.path.join(data_sample_dir,'LMET_valid.pkl'))

    # filter pkl files: Entities in train split set 
    df_et_train = pd.read_csv(os.path.join(data_sample_dir, "ET_train.txt"), sep='\t', header=None)
    ids = [e2id[entity] for entity in list(df_et_train[0].unique()) if entity in e2id]
    filter_pkl(os.path.join(data_sample_dir,'LMET_train.pkl'), entities=ids)

    # filter pkl files: Entities in valid split set 
    df_et_valid = pd.read_csv(os.path.join(data_sample_dir, "ET_valid.txt"), sep='\t', header=None)
    ids = [e2id[entity] for entity in list(df_et_valid[0].unique()) if entity in e2id]
    filter_pkl(os.path.join(data_sample_dir,'LMET_valid.pkl'), entities=ids)

    # filter pkl files: Entities in test split set
    df_et_test = pd.read_csv(os.path.join(data_dir, "ET_test.txt"), sep='\t', header=None)
    ids = [e2id[entity] for entity in list(df_et_test[0].unique()) if entity in e2id]
    filter_pkl(os.path.join(data_sample_dir,'LMET_test.pkl'), entities=ids)
    
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

        # Convert to pkl
        triplets_list = convert_to_pkl(data_dir, os.path.join(data_sample_dir_2hop,"KG_train.txt"),
                                       os.path.join(data_sample_dir_2hop,'ET_train.txt'),
                                       os.path.join(data_sample_dir_2hop,'LMET_train.pkl'), get_clust_name_fonc=get_clust_name_fonc)
        
        triplets_list = convert_to_pkl(data_dir, os.path.join(data_sample_dir_2hop,"KG_train.txt"),
                                os.path.join(data_sample_dir_2hop,'ET_train.txt'),
                                os.path.join(data_sample_dir_2hop,'LMET_test.pkl'), get_clust_name_fonc=get_clust_name_fonc)
        
        triplets_list = convert_to_pkl(data_dir, os.path.join(data_sample_dir_2hop,"KG_train.txt"),
                        os.path.join(data_sample_dir_2hop,'ET_train.txt'),
                        os.path.join(data_sample_dir_2hop,'LMET_valid.pkl'), get_clust_name_fonc=get_clust_name_fonc)

        # Copy other dataset files
        copy_other_kg_files(data_sample_dir, data_sample_dir_2hop)
        
        # filter pkl files: Entities in train split set 
        df_et_train = pd.read_csv(os.path.join(data_sample_dir_2hop, "ET_train.txt"), sep='\t', header=None)
        ids = [e2id[entity] for entity in list(df_et_train[0].unique()) if entity in e2id]
        filter_pkl(os.path.join(data_sample_dir_2hop,'LMET_train.pkl'), entities=ids)

        # filter pkl files: Entities in valid split set 
        df_et_valid = pd.read_csv(os.path.join(data_sample_dir_2hop, "ET_valid.txt"), sep='\t', header=None)
        ids = [e2id[entity] for entity in list(df_et_valid[0].unique()) if entity in e2id]
        filter_pkl(os.path.join(data_sample_dir_2hop,'LMET_valid.pkl'), entities=ids)

        # filter pkl files: Entities in test split set
        df_et_test = pd.read_csv(os.path.join(data_sample_dir_2hop, "ET_test.txt"), sep='\t', header=None)
        ids = [e2id[entity] for entity in list(df_et_test[0].unique()) if entity in e2id]
        filter_pkl(os.path.join(data_sample_dir_2hop,'LMET_test.pkl'), entities=ids)

        print("Created .pkl file with 2-hop neighbors")


if __name__ == '__main__':
    main()
