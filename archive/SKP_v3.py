import pandas as pd
import os
import json
from pathlib import Path
import json
from utils import *
from mhsk_utils import *
import math
from tqdm import tqdm
from sampling_utils import *

"""
COMPLETE SEMANTIC 2-HOP CHANGES OF SEMANTIC KNOWLEDGE PROCESS - SKP V3
"""

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

if torch.cuda.is_available():
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')  # Forces GPU

print(torch.cuda.is_available())

# Specify locations for loading and saving data
project_folder = os.getcwd()
data_path = os.path.join(project_folder, "data", "FB15kET_sample")
result_folder = os.path.join(project_folder, "data_entity_metrics")
data_2hop_path = os.path.join(project_folder, "data", "FB15kET_sample_2hop_sentences")
os.makedirs(data_2hop_path, exist_ok=True)

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
e_coherence = pd.read_csv(os.path.join(result_folder, "entity_metrics_sample_new.csv"))

# Entities with degree considered too low
lowerb_quantiles = e_coherence[['kg_degree', 'et_degree']].quantile(0.1)
low_kg_degree = lowerb_quantiles.loc['kg_degree']
low_et_degree = lowerb_quantiles.loc['et_degree']

# Filter only entities with bad metrics
entity_low_degree_df = e_coherence[(e_coherence["kg_degree"] <= low_kg_degree) \
                              & (e_coherence["et_degree"] <= low_et_degree)]
entity_low_degree = entity_low_degree_df['entity'].to_list()

entity_kg_2hop = defaultdict(list)
entity_et_2hop = []

for entity in tqdm(entity_low_degree, total=len(entity_low_degree), desc="Processing entities", unit="entity"):
    
    # Current sentences
    kg_entity_text, _ = kg_sentences(df_triples, entity, r2text, r2id, e2desc, e2id)
    et_train_sentences, _ = et_sentences(df_train, entity, t2desc, t2id)
    entity_sentences = kg_entity_text + et_train_sentences

    # Number 2-hop relations and types added
    n_rel = int(round(low_kg_degree * (1 - len(kg_entity_text) / (low_kg_degree+1))) + 2)
    n_type = int(round(low_et_degree*2* (1 - len(et_train_sentences) / (low_et_degree+1))) + 2)
    
    # 2-hop neighbor sentences
    kg_2hop, kg_2hop_sentences = two_hop_neighbors_2(df_triples, entity, r2text, r2id, e2desc, e2id)
    types_2hop, et_txt_2hop = two_hop_types(df_triples, df_train, entity, t2desc, t2id)

    # 2-hop kg neighbor and type with highest average similarity score
    kg_top_2hop = max_sim_2hop(entity_sentences, kg_2hop_sentences, kg_2hop, n_rel)
    et_top_2hop = max_sim_2hop(entity_sentences, et_txt_2hop, types_2hop, n_type, kg=False)

    # Store best results for additional information
    for relation, entity2, direction in kg_top_2hop:
        if direction == '-':
            entity_kg_2hop[entity].append(f"{relation} {entity2}")
        else:
            entity_kg_2hop[entity].append(f"inv-{relation} {entity2}")

    for et_type in et_top_2hop:
        entity_et_2hop.append((entity, et_type))

    new_composed_relations = set()
    for relation, _, _ in kg_top_2hop:
        if '.' in relation and relation not in r2id:
            new_composed_relations.add(relation)

    update_relation_tsv(os.path.join(data_2hop_path, 'relations.tsv'), new_composed_relations, r2id)

# Save
with open(os.path.join(data_2hop_path, 'relation2hop.json'), "w") as f:
    json.dump(entity_kg_2hop, f, indent=4)

# # Entities with degree considered too high
# upperb_quantiles = e_coherence[['kg_degree', 'et_degree']].quantile(0.90)
# high_kg_degree = upperb_quantiles.loc['kg_degree']
# high_et_degree = upperb_quantiles.loc['et_degree']

# # Filter only entities with bad metrics
# entity_high_degree_df = e_coherence[(e_coherence["kg_degree"] > high_kg_degree) \
#                               & (e_coherence["et_degree"] > high_et_degree)]
# entity_high_degree = entity_high_degree_df['entity'].to_list()

# # removed results dataframe
# kg_train_removed_df = pd.DataFrame()
# et_train_removed_df = pd.DataFrame()
# entity_kg_remove = defaultdict(list)

# for entity in tqdm(entity_high_degree, total=len(entity_high_degree), desc="Processing entities", unit="entity"):

#     # Current sentences
#     kg_entity_text, neighbors = kg_sentences(df_triples, entity, r2text, r2id, e2desc, e2id, filter=False)
#     et_train_sentences, et_train = et_sentences(df_train, entity, t2desc, t2id)
#     entity_sentences = kg_entity_text + et_train_sentences

#     # Remove noisy relationships and types
#     n_kg_remove = int(math.ceil(len(kg_entity_text)*0.1))
#     n_et_remove = int(math.ceil(len(et_train_sentences)*0.1))

#     # Remove noisy neighbors through similarity score
#     kg_train_removed, et_train_removed = remove_noisy_neighbors(kg_entity_text, neighbors, et_train_sentences, et_train, n_kg_remove, n_et_remove)
#     # Save removed relationships and types
#     rel_removed = relationship_removed(kg_train_removed, entity)
#     entity_kg_remove[entity] = rel_removed

#     # Store removed results
#     et_train_removed_df = pd.concat([et_train_removed_df, et_train_removed], axis=0).reset_index(drop=True)

# Save
# with open(os.path.join(data_2hop_path, 'relation2remove.json'), "w") as f:
#     json.dump(entity_kg_remove, f, indent=4)

# Update KG_train and ET_train without noise relations
# kg_train_new = df_triples.merge(kg_train_removed_df, on=[0, 1, 2], how='left', indicator=True)
# kg_train_new = kg_train_new[kg_train_new['_merge'] == 'left_only'].drop(columns=['_merge'])
# et_train_new = df_train.merge(et_train_removed_df, how='left', indicator=True)
# et_train_new = et_train_new[et_train_new['_merge'] == 'left_only'].drop(columns=['_merge'])

# Convert 2-hop additions in dataframe
kg_train_2hop = pd.DataFrame(entity_kg_2hop, columns=[0,1,2])
et_train_2hop = pd.DataFrame(entity_et_2hop, columns=[0,1])

# Final processed train files
# kg_train_processed = pd.concat([kg_train_new, kg_train_2hop], axis=0).reset_index(drop=True)
# et_train_processed = pd.concat([et_train_new, et_train_2hop], axis=0).reset_index(drop=True)
# et_train_processed.to_csv(os.path.join(data_2hop_path, 'ET_train.txt'), sep='\t', header=None, index=False)

# Same but without the deletion
kg_train_processed = pd.concat([df_triples, kg_train_2hop], axis=0).reset_index(drop=True)
kg_train_processed.to_csv(os.path.join(data_2hop_path, 'KG_train.txt'), sep='\t', header=None, index=False)
et_train_processed = pd.concat([df_train, et_train_2hop], axis=0).reset_index(drop=True)
et_train_processed.to_csv(os.path.join(data_2hop_path, 'ET_train.txt'), sep='\t', header=None, index=False)
generate_relation2text(os.path.join(data_2hop_path, 'relations.tsv'), os.path.join(data_2hop_path, 'relation2text.txt'))