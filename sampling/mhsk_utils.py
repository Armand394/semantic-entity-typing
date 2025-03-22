import pandas as pd
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_coherence_statsdf(e_coherence, entity_labels):
    # Create a reverse mapping from label + description back to ID
    reverse_entity_labels = {
        f"{v.get('label', k)} {v.get('description', '')}": k 
        for k, v in entity_labels.items()
    }

    # Restore original entity IDs in the "object" column
    e_coherence["entity"] = e_coherence["entity"].map(lambda x: reverse_entity_labels.get(x, x))

    # Rename columns for clarity
    e_coherence = e_coherence.rename(columns={ 'triple_self_coherence_mean': 'kg_sim_mu',
            'triple_self_coherence_std': 'kg_sim_sig',
            'type_self_coherence_mean': 'et_sim_mu',
            'type_self_coherence_std': 'et_sim_sigma'})

    # Important features and their impact
    determining_features = { 'kg_sim_mu': 1,
            'kg_sim_sig': 1,
            'et_sim_mu': 1,
            'et_sim_sigma': -1}

    # Take only important columns of metrics
    e_coherence = e_coherence[['entity'] + list(determining_features.keys())]

    return e_coherence

def kg_sentences(df_triples, entity, r2text, r2id, e2desc, e2id):
    o, r, s = df_triples.columns

    # Textual value outcoming neighbors
    outgoing_neighbors = df_triples[df_triples[o] == entity].reset_index(drop=True)
    # Textual value relation
    outgoing_neighbors[r] = outgoing_neighbors[r].map(lambda rel: r2text[r2id[rel]])
    # Textual value subject
    outgoing_neighbors[s] = outgoing_neighbors[s].map(lambda e: e2desc[e2id[e]])

    # Textual value incoming neighbors
    ingoing_neighbors = df_triples[df_triples[s] == entity].reset_index(drop=True)
    # Textual value relation
    ingoing_neighbors[r] = ingoing_neighbors[r].map(lambda rel: r2text[r2id[rel]])
    # Textual value object
    ingoing_neighbors[o] = ingoing_neighbors[o].map(lambda e: e2desc[e2id[e]])

    # Construct sentences
    outgoing_sentences = (outgoing_neighbors[r].astype(str) + " " + outgoing_neighbors[s].astype(str)).tolist()
    ingoing_sentences = (ingoing_neighbors[o].astype(str) + " " + ingoing_neighbors[r].astype(str)).tolist()
    
    return outgoing_sentences + ingoing_sentences

def et_sentences(df_train, entity, t2desc, t2id):
    o, t = df_train.columns
    et_train = df_train[df_train[o] == entity].reset_index(drop=True)
    et_train[t] = et_train[t].map(lambda typ: t2desc[t2id[typ]])
    et_train[t] = et_train[t].str.replace(" [SEP] ", " ")
    et_train[t] = "has type " + et_train[t]

    return et_train[t].to_list()

def compute_mean_similarity(sentences):
    # Encode sentences to get an
    embedding_matrix = model.encode(sentences, convert_to_numpy=True)

    # Compute cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(embedding_matrix)


    # Extract upper triangle without diagonal
    triu_indices = np.triu_indices_from(cosine_sim_matrix, k=1)
    cosine_sim_values = cosine_sim_matrix[triu_indices]

    # Compute mean cosine similarity
    return np.mean(cosine_sim_values)

def two_hop_neighbors(df_triples, entity):
    # Retrieve columns
    e_col, r_col, s_col = df_triples.columns

    # Step 1: Construct adjacency lists for incoming & outgoing relations
    outgoing_map = defaultdict(set)
    incoming_map = defaultdict(set)

    # Fill adjacency maps with
    for obj, rel, subj in zip(df_triples[e_col], df_triples[r_col], df_triples[s_col]):
        outgoing_map[obj].add((subj, rel))
        incoming_map[subj].add((obj, rel))

    neighbors = set()

    # CASE 1: (object_x --> mid --> object_z)
    for mid, r1 in outgoing_map[entity]:  
        for obj_z, r2 in outgoing_map.get(mid, set()):  
            neighbors.add((r2, obj_z, '-'))

    # CASE 2: (object_x <-- mid <-- object_z)
    for mid, r1 in incoming_map[entity]:  
        for obj_z, r2 in incoming_map.get(mid, set()):  
            neighbors.add((r2, obj_z, 'inv'))

    # CASE 3: (object_x --> mid <-- object_z)
    for mid, r1 in outgoing_map[entity]:  
        for obj_z, r2 in incoming_map.get(mid, set()):  
            neighbors.add((r2, obj_z, 'inv'))

    # CASE 4: (object_x <-- mid --> object_z)
    for mid, r1 in incoming_map[entity]:
        for obj_z, r2 in outgoing_map.get(mid, set()):  
            neighbors.add((r2, obj_z, '-'))

    return list(neighbors)


def two_hop_types(df_triples, df_train, entity):
    # Retreive columns
    o, r, s = df_triples.columns
    o_t, t = df_train.columns
    # Incomining and outcoming neighbors
    outgoing_neighbors = df_triples[df_triples[o] == entity].reset_index(drop=True)
    ingoing_neighbors = df_triples[df_triples[s] == entity].reset_index(drop=True)
    # Unique neighbor entities
    neighbors = set(outgoing_neighbors[s]) | set(ingoing_neighbors[o])
    # 2-hop types
    return df_train[df_train[o_t].isin(neighbors)][t].to_list()


def sample_2hop_sentences(neighbors_2hop, types_2hop, n_rel, n_type, e2desc, e2id, r2text, r2id, t2desc, t2id, sample_r=True, sample_t=True):
    
    if sample_r and not sample_t:
        # Sample 2-hop from KG and ET
        sampled_neighbors_2hop = random.sample(neighbors_2hop, min(n_rel, len(neighbors_2hop)))

        # Create KG sampled sentences
        sampled_r_sentences = []
        for r, e, dir in sampled_neighbors_2hop:
            if dir == 'inv':
                sampled_r_sentences.append(f"{e2desc[e2id[e]]} {r2text[r2id[r]]}")
            else:
                sampled_r_sentences.append(f"{r2text[r2id[r]]} {e2desc[e2id[e]]}")    
        
        return sampled_r_sentences, [], sampled_neighbors_2hop, []
    
    elif not sample_r and sample_t:
        # Sample 2-hop from ET
        sampled_types_2hop = random.sample(types_2hop, min(n_type, len(types_2hop)))   
        # Create ET sampled sentences
        sampled_t_sentence = []
        for t in sampled_types_2hop:
            sampled_t_sentence.append(f"has type {t2desc[t2id[t]].replace(" [SEP] ", " ")}")

        return [], sampled_t_sentence, [], sampled_types_2hop
    else:
        # Sample 2-hop from KG and ET
        sampled_neighbors_2hop = random.sample(neighbors_2hop, min(n_rel, len(neighbors_2hop)))
        sampled_types_2hop = random.sample(types_2hop, min(n_type, len(types_2hop)))

        # Create KG sampled sentences
        sampled_r_sentences = []
        for r, e, dir in sampled_neighbors_2hop:
            if dir == 'inv':
                sampled_r_sentences.append(f"{e2desc[e2id[e]]} {r2text[r2id[r]]}")
            else:
                sampled_r_sentences.append(f"{r2text[r2id[r]]} {e2desc[e2id[e]]}")

        # Create ET sampled sentences
        sampled_t_sentence = []
        for t in sampled_types_2hop:
            sampled_t_sentence.append(f"has type {t2desc[t2id[t]].replace(" [SEP] ", " ")}")
        
        return sampled_r_sentences, sampled_t_sentence, sampled_neighbors_2hop, sampled_types_2hop








def recompute_similarity(df_triples, df_train, r2text, r2id, e2desc, e2id, t2desc, t2id, result_folder):
    """
    Only computes mean similarity for kg triples and training types (does not compute other metrics). For detailed
    metrics computation see main_analysis.py.
    """
    #Compute coherence metrics
    entities_kg = e2id.keys()
    metrics = []

    i = 0
    for entity in entities_kg:
        # Compute mean cosine similarity of KG sentences
        kg_entity_text = kg_sentences(df_triples, entity, r2text, r2id, e2desc, e2id)
        if len(kg_entity_text) > 1:
            base_sim_kg = compute_mean_similarity(kg_entity_text)
        else:
            base_sim_kg = None

        # Compute mean cosine similarity of ET sentences
        et_train_sentences = et_sentences(df_train, entity, t2desc, t2id)
        if len(et_train_sentences) > 1:
            base_sim_et = compute_mean_similarity(et_train_sentences)
        else:
            base_sim_et = None

        if i % 100 == 0:
            print(i)

        i += 1
        metrics.append((entity, base_sim_kg, base_sim_et))

    e_coherence = pd.DataFrame(metrics, columns=['entity', 'kg_sim_mu', 'et_sim_mu'])

    e_coherence.to_csv(os.path.join(result_folder, "entity_metrics.csv"), index=False)


