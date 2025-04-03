import pandas as pd
import os
from pathlib import Path
import json
from utils import *
from mhsk_utils import *
import math
from tqdm import tqdm

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

if torch.cuda.is_available():
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')  # Forces GPU

print(torch.cuda.is_available())

# Specify locations for loading and saving data
project_folder = os.getcwd()
data_path = os.path.join(project_folder, "data", "FB15kET_sample")
result_folder = os.path.join(project_folder, "data_entity_metrics")

# Load Ids and descriptions
e2id = read_id(os.path.join(data_path, 'entities.tsv'))
r2id = read_id(os.path.join(data_path, 'relations.tsv'))
t2id = read_id(os.path.join(data_path, 'types.tsv'))
c2id = read_id(os.path.join(data_path, 'clusters.tsv'))

e2desc, e2text = read_entity_wiki(os.path.join(data_path, 'entity_wiki.json'), e2id, 'hybrid')
r2text = read_rel_context(os.path.join(data_path, 'relation2text.txt'), r2id)
t2desc = read_type_context(os.path.join(data_path, 'hier_type_desc.txt'), t2id)

# Load KG data
df_triples = pd.read_csv(os.path.join(data_path, "KG_train.txt"), sep='\t', header=None)
df_train = pd.read_csv(os.path.join(data_path, "ET_train.txt"), sep='\t', header=None)

with open(os.path.join(data_path, 'entity_wiki.json'), "r") as f:
    entity_labels = json.load(f)

# Recompute coherence metrics for entities
if not os.path.exists(os.path.join(result_folder, "entity_metrics_sample.csv")):
    recompute_similarity(df_triples, df_train, r2text, r2id, e2desc, e2id, t2desc, t2id, result_folder)

# Load entity coherence metrics
e_coherence = pd.read_csv(os.path.join(result_folder, "entity_metrics_sample.csv"))

# Filter only entities with bad metrics
df_entity_wrong = e_coherence[(e_coherence["kg_degree"] <= 9) & (e_coherence["et_degree"] <= 5)]
df_entity_wrong = df_entity_wrong.drop_duplicates()
entity_low_degree = df_entity_wrong['entity'].to_list()

# Output
output_2hop_kg = os.path.join(result_folder, "KG_train_2hop_sample.txt")
output_2hop_et = os.path.join(result_folder, "ET_train_2hop_sample.txt")

# write on both files 2-hop information
with open(output_2hop_kg, "w") as f1, open(output_2hop_et, "w") as f2:
    for entity in tqdm(entity_low_degree, total=len(entity_low_degree), desc="Processing entities", unit="entity"):
        
        # Current sentences
        kg_entity_text = kg_sentences(df_triples, entity, r2text, r2id, e2desc, e2id)
        et_train_sentences = et_sentences(df_train, entity, t2desc, t2id)
        entity_sentences = kg_entity_text + et_train_sentences

        # Number 2-hop relations and types added
        n_rel = int(round(7 * (1 - len(kg_entity_text) / 9)) + 3)
        n_type = int(round(4 * (1 - len(et_train_sentences) / 5)) + 2)
        
        # 2-hop neighbor sentences
        kg_2hop, kg_2hop_sentences = two_hop_neighbors(df_triples, entity, r2text, r2id, e2desc, e2id)
        types_2hop, et_txt_2hop = two_hop_types(df_triples, df_train, entity, t2desc, t2id)

        # 2-hop kg neighbor and type with highest average similarity score
        kg_top_2hop = max_sim_2hop(entity_sentences, kg_2hop_sentences, kg_2hop, n_rel)
        et_top_2hop = max_sim_2hop(entity_sentences, et_txt_2hop, types_2hop, n_type)

        # Write best results in new files for additional information
        for relation, entity2, direction in kg_top_2hop:
            if direction == '-':
                line_r = f"{entity}\t{relation}\t{entity2}\n"
            elif direction == 'inv':
                line_r = f"{entity2}\t{relation}\t{entity}\n"

            f1.write(line_r)

        for type in et_top_2hop:
            line_e = f"{entity}\t{type}\n"
            f2.write(line_e)
