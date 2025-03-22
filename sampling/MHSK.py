import pandas as pd
import os
from pathlib import Path
import json
from utils import *
from mhsk_utils import *
import math
from tqdm import tqdm

# Specify locations for loading and saving data
project_folder = Path(__file__).parent.parent
data_path = os.path.join(project_folder, "data", "FB15kET")
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

# # Recompute coherence metrics for entities
# recompute_similarity(df_triples, df_train, r2text, r2id, e2desc, e2id, t2desc, t2id, result_folder)

# Load metrics for entities (computed in main_analysis.py)
if not os.path.exists(os.path.join(result_folder, "entity_metrics.csv")):
    df_stats_correct = pd.read_csv(os.path.join(result_folder, "data_metrics_good_cl.csv"))
    df_stats_bad = pd.read_csv(os.path.join(result_folder, "data_metrics_bad_cl.csv"))
    e_coherence = pd.concat([df_stats_correct, df_stats_bad], ignore_index=True)
    e_coherence = clean_coherence_statsdf(e_coherence, entity_labels)
    e_coherence.to_csv("entity_metrics.csv", index=False)

# Load entity coherence metrics
e_coherence = pd.read_csv(os.path.join(result_folder, "entity_metrics.csv"))

# Filter coherence for 
coherence_stats = e_coherence.drop(columns=['entity']).describe().loc[['mean', 'std']]

# Global mean KG and ET similarity
kg_sim_global = coherence_stats.loc['mean', 'kg_sim_mu']
et_sim_global = coherence_stats.loc['mean', 'et_sim_mu']

# Lowerbound for metrics for entities to be considered 'bad'
kg_sim_lb = coherence_stats.loc['mean', 'kg_sim_mu'] - coherence_stats.loc['std', 'kg_sim_mu']*0.5
et_sim_lb = coherence_stats.loc['mean', 'et_sim_mu'] - coherence_stats.loc['std', 'et_sim_mu']*1.5

# Filter only entities with bad metrics
df_entity_wrong = e_coherence[(e_coherence["kg_sim_mu"] < kg_sim_lb) |
                                    (e_coherence["et_sim_mu"] < et_sim_lb)]

# Acquire entity list
e_low_coherence = df_entity_wrong['entity'].to_list()
e_low_coherence = list(set(e_low_coherence))

# Output
output_2hop_kg = os.path.join(result_folder, "KG_train_2hop.txt")
output_2hop_et = os.path.join(result_folder, "ET_train_2hop.txt")

# write on both files 2-hop information
with open(output_2hop_kg, "w") as f1, open(output_2hop_et, "w") as f2:
    for entity in tqdm(e_low_coherence, desc="Processing Entities", unit="entity"):
        # Compute mean cosine similarity of KG sentences
        kg_entity_text = kg_sentences(df_triples, entity, r2text, r2id, e2desc, e2id)
        if len(kg_entity_text) > 1:
            base_sim_kg = compute_mean_similarity(kg_entity_text)
        else:
            base_sim_kg = kg_sim_global

        # Compute mean cosine similarity of ET sentences
        et_train_sentences = et_sentences(df_train, entity, t2desc, t2id)
        if len(et_train_sentences) > 1:
            base_sim_et = compute_mean_similarity(et_train_sentences)
        else:
            base_sim_et = et_sim_global
        
        # 2-hop neighbor of entity
        neighbors_2hop = two_hop_neighbors(df_triples, entity)
        types_2hop = two_hop_types(df_triples, df_train, entity)

        # Number sample relation
        n_rel = int(math.ceil(len(kg_entity_text)*0.1))
        n_type = int(math.ceil(len(et_train_sentences)*0.15))

        # Low similarity score
        r_sim_low = (base_sim_kg < kg_sim_lb)
        et_sim_low = (base_sim_et < et_sim_lb)
        
        # Initiate curent best
        best_sim_kg = base_sim_kg
        best_sim_et = base_sim_et
        best_2hop_kg = []
        best_2hop_et = []

        i = 0
        while (i < 30):
            # Sample relations and types and construct their sentences
            sampled_r_txt, sampled_t_txt, sampled_kg_2hop, sampled_et_2hop = sample_2hop_sentences(neighbors_2hop, types_2hop,
                                                                            n_rel, n_type, e2desc,
                                                                            e2id, r2text, r2id, t2desc, t2id,
                                                                            sample_r=r_sim_low, sample_t=et_sim_low)

            # New sentences for current entity
            mh_kg_sentences = kg_entity_text + sampled_r_txt
            mh_et_sentences = et_train_sentences + sampled_t_txt

            if r_sim_low and et_sim_low:
                new_sim_kg = compute_mean_similarity(mh_kg_sentences)
                new_sim_et = compute_mean_similarity(mh_et_sentences)

                # Update best sample for improving similarity score
                if new_sim_kg > best_sim_kg:
                    best_2hop_kg = sampled_kg_2hop
                    best_sim_kg = new_sim_kg
                if new_sim_et > best_sim_et:
                    best_2hop_et = sampled_et_2hop
                    best_sim_et = new_sim_et

                # If current similarity above global average, stop searching
                if best_sim_kg >= kg_sim_global and best_sim_et >= et_sim_global:
                    break 

            elif r_sim_low and not et_sim_low:
                new_sim_kg = compute_mean_similarity(mh_kg_sentences)

                # Update best sample for improving similarity score
                if new_sim_kg > best_sim_kg:
                    best_2hop_kg = sampled_kg_2hop
                    best_sim_kg = new_sim_kg

                # If current similarity above global average, stop searching
                if best_sim_kg > kg_sim_global:
                    break 
                
            else:
                new_sim_et = compute_mean_similarity(mh_et_sentences)
                
                # Update best sample for improving similarity score
                if new_sim_et > best_sim_et:
                    best_2hop_et = sampled_et_2hop 
                    best_sim_et = new_sim_et

                # If current similarity above global average, stop searching
                if best_sim_et > et_sim_global:
                    break 

            i += 1

        # Write best results in new files for additional information
        for relation, entity2, direction in best_2hop_kg:
            if direction == '-':
                line_r = f"{entity} {relation} {entity2}\n"
            elif direction == 'inv':
                line_r = f"{entity2} {relation} {entity}\n"

            f1.write(line_r)
    
        for type in best_2hop_et:
            line_e = f"{entity} {type}\n"
            f2.write(line_e)
