import networkx as nx
import random
import os
from sampling_utils import *
from utils import read_id
from dataloader import SEMdataset
import numpy as np
import json

dataset = "FB15kET"

project_path = os.getcwd()
data_folder = os.path.join(project_path, "data")
data_dir = os.path.join(project_path, "data", dataset)
data_sample_dir = os.path.join(project_path, "data", f"{dataset}_sample")

print("Generate sample graph with 2-hop neighbors")
data_sample_dir_2hop = os.path.join(project_path, "data", f"{dataset}_sample_2hop_sentences")
os.makedirs(data_sample_dir_2hop, exist_ok=True)

# Chargement des dictionnaires à partir des fichiers TSV
entite_dict = load_tsv(os.path.join(data_sample_dir, "entities.tsv"))
relation_dict = load_tsv(os.path.join(data_sample_dir_2hop, "relations.tsv"))
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
# with open(os.path.join(data_sample_dir_2hop,"relation2remove.json"), "r") as f:
#     r2move = json.load(f)

# # Construct LMET_train.txt files for dataloading
# construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict,
#                 output_file=os.path.join(data_sample_dir_2hop, "LMET_train.txt"), mode="train", kg_dict2=r2hop, kg_remove=r2move)
# # Construct LMET_valid.txt files for dataloading
# construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict,
#                 output_file=os.path.join(data_sample_dir_2hop, "LMET_valid.txt"), mode="train", kg_dict2=r2hop, kg_remove=r2move)
# # Construct LMET_test.txt files for dataloading
# construct_output(kg_dict, et_train_dict, et_valid_dict, et_test_dict, entite_dict, relation_dict, type_dict,
#                 cluster_dict, output_file=os.path.join(data_sample_dir_2hop, "LMET_test.txt"), mode="test", kg_dict2=r2hop, kg_remove=r2move)

# Construct LMET_train.txt files for dataloading
construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict,
                output_file=os.path.join(data_sample_dir_2hop, "LMET_train.txt"), mode="train", kg_dict2=r2hop)
# Construct LMET_valid.txt files for dataloading
construct_output(kg_dict, et_train_dict, et_valid_dict, {}, entite_dict, relation_dict, type_dict, cluster_dict,
                output_file=os.path.join(data_sample_dir_2hop, "LMET_valid.txt"), mode="train", kg_dict2=r2hop)
# Construct LMET_test.txt files for dataloading
construct_output(kg_dict, et_train_dict, et_valid_dict, et_test_dict, entite_dict, relation_dict, type_dict,
                cluster_dict, output_file=os.path.join(data_sample_dir_2hop, "LMET_test.txt"), mode="test", kg_dict2=r2hop)


# Load dictionaries entity -> id
e2id = read_id(os.path.join(data_sample_dir, 'entities.tsv'))
r2id = read_id(os.path.join(data_sample_dir_2hop, 'relations.tsv'))
t2id = read_id(os.path.join(data_sample_dir, 'types.tsv'))
c2id = read_id(os.path.join(data_sample_dir, 'clusters.tsv'))

args = args_dict(dataset=f"{dataset}_sample_2hop_sentences", data_dir=data_folder)

SEMdataset(args, "LMET_train.txt", e2id, r2id, t2id, c2id, 'train')
SEMdataset(args, "LMET_test.txt", e2id, r2id, t2id, c2id, 'test')
SEMdataset(args, "LMET_valid.txt", e2id, r2id, t2id, c2id, 'valid')

# Remove unnecessary .txt files
os.remove(os.path.join(data_sample_dir_2hop, "LMET_train.txt"))
os.remove(os.path.join(data_sample_dir_2hop, "LMET_valid.txt"))
os.remove(os.path.join(data_sample_dir_2hop, "LMET_test.txt"))

print(".pkl files with sampled graph saved")