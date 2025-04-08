import pandas as pd
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
import math

if torch.cuda.is_available():
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')  # Forces GPU
else:
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


def recompute_similarity(df_triples, df_train, r2text, r2id, e2desc, e2id, t2desc, t2id, result_folder):
    """
    Only computes mean similarity for kg triples and training types (does not compute other metrics). For detailed
    metrics computation see main_analysis.py.
    """
    #Compute coherence metrics
    entities_kg = set(df_triples[0].unique()).union(set(df_triples[2].unique()))
    metrics = []

    i = 0
    for entity in tqdm(entities_kg, desc="Computing entity metrics", unit="Entity"):
        # Compute mean cosine similarity of KG sentences
        kg_entity_text, _ = kg_sentences(df_triples, entity, r2text, r2id, e2desc, e2id, filter=False)
        base_sim_kg = compute_mean_similarity(kg_entity_text)

        # Compute mean cosine similarity of ET sentences
        et_train_sentences, _ = et_sentences(df_train, entity, t2desc, t2id)
        base_sim_et = compute_mean_similarity(et_train_sentences)
        
        # Degree of entity
        degree = len(kg_entity_text) + len(et_train_sentences)
        
        # Average text length entity
        sentences = kg_entity_text + et_train_sentences
        avg_length = sum(len(s) for s in sentences) / len(sentences)

        # type-kg ratio, degrees
        kg_degree = len(kg_entity_text)
        et_degree = len(et_train_sentences)
        ratio = len(et_train_sentences) / len(kg_entity_text)

        metrics.append((entity, base_sim_kg, base_sim_et, degree, avg_length, kg_degree,
                        et_degree, ratio))

    e_coherence = pd.DataFrame(metrics, columns=['entity', 'kg_sim_mu', 'et_sim_mu', 'degree',
                                                'avg_txt_length', 'kg_degree', 'et_degree','type_kg_ratio'])

    e_coherence.to_csv(os.path.join(result_folder, "entity_metrics_sample.csv"), index=False)

def compute_mean_similarity(sentences):
    if len(sentences) < 2:
        return 0.5 

    # Encode sentences to get an
    embedding_matrix = model.encode(sentences, convert_to_numpy=True, batch_size=200)

    # Compute cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(embedding_matrix)
    
    # Extract upper triangle without diagonal
    triu_indices = np.triu_indices_from(cosine_sim_matrix, k=1)
    cosine_sim_values = cosine_sim_matrix[triu_indices]

    # Compute mean cosine similarity
    return np.mean(cosine_sim_values)

def kg_sentences(df_triples, entity, r2text, r2id, e2desc, e2id, filter=True):
    # Filter out the relashionships with low semantic information
    bad_r = ["/common/annotation_category/annotations./common/webpage/topic",
            "/common/topic/webpage./common/webpage/category"]
    
    o, r, s = df_triples.columns

    # Neigbor dataframe
    outgoing_neighbors = df_triples[df_triples[o] == entity].reset_index(drop=True)
    ingoing_neighbors = df_triples[df_triples[s] == entity].reset_index(drop=True)
    neighbors = pd.concat([outgoing_neighbors, ingoing_neighbors], axis=0).reset_index(drop=True)

    # Filter relationships with low semantic information
    if filter:
        outgoing_neighbors =  outgoing_neighbors[~(outgoing_neighbors[r].isin(bad_r))]
    # Textual value relation
    outgoing_neighbors[r] = outgoing_neighbors[r].map(lambda rel: r2text[r2id[rel]])
    # Textual value subject
    outgoing_neighbors[s] = outgoing_neighbors[s].map(lambda e: e2desc[e2id[e]])

    # Filter relationships with low semantic information
    if filter:
        ingoing_neighbors =  ingoing_neighbors[~(ingoing_neighbors[r].isin(bad_r))]
    # Textual value relation
    ingoing_neighbors[r] = ingoing_neighbors[r].map(lambda rel: r2text[r2id[rel]])
    # Textual value object
    ingoing_neighbors[o] = ingoing_neighbors[o].map(lambda e: e2desc[e2id[e]])

    # Construct sentences
    outgoing_sentences = (outgoing_neighbors[r].astype(str) + " " + outgoing_neighbors[s].astype(str)).tolist()
    ingoing_sentences = (ingoing_neighbors[o].astype(str) + " " + ingoing_neighbors[r].astype(str)).tolist()
    sentences = outgoing_sentences + ingoing_sentences
    return sentences, neighbors

def et_sentences(df_train, entity, t2desc, t2id):
    o, t = df_train.columns
    et_train = df_train[df_train[o] == entity].reset_index(drop=True)
    et_train_filtered = et_train.copy(deep=True)
    et_train.loc[:, t] = et_train[t].map(lambda typ: t2desc[t2id[typ]])
    et_train.loc[:, t] = et_train[t].str.replace(" [SEP] ", " ", regex=False)
    et_train.loc[:, t] = "has type " + et_train[t]

    return et_train[t].to_list(), et_train_filtered

def two_hop_neighbors(df_triples, entity, r2text, r2id, e2desc, e2id):
    # Filter out the relashionships with low semantic information
    bad_r = ["/common/annotation_category/annotations./common/webpage/topic",
            "/common/topic/webpage./common/webpage/category"]
    
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
    sentences = set()

    # CASE 1: (object_x --> mid --> object_z)
    for mid, r1 in outgoing_map[entity]:  
        for obj_z, r2 in outgoing_map.get(mid, set()):
            if r2 in bad_r:
                continue
            neighbors.add((r2, obj_z, '-'))
            sentences.add(f"{r2text[r2id[r2]]} {e2desc[e2id[obj_z]]}")

    # CASE 2: (object_x <-- mid <-- object_z)
    for mid, r1 in incoming_map[entity]:  
        for obj_z, r2 in incoming_map.get(mid, set()):
            if r2 in bad_r:
                continue
            neighbors.add((r2, obj_z, 'inv'))
            sentences.add(f"{e2desc[e2id[obj_z]]} {r2text[r2id[r2]]}")

    # CASE 3: (object_x --> mid <-- object_z)
    for mid, r1 in outgoing_map[entity]:  
        for obj_z, r2 in incoming_map.get(mid, set()): 
            if r2 in bad_r:
                continue
            neighbors.add((r2, obj_z, 'inv'))
            sentences.add(f"{e2desc[e2id[obj_z]]} {r2text[r2id[r2]]}")

    # CASE 4: (object_x <-- mid --> object_z)
    for mid, r1 in incoming_map[entity]:
        for obj_z, r2 in outgoing_map.get(mid, set()):  
            if r2 in bad_r:
                continue
            neighbors.add((r2, obj_z, '-'))
            sentences.add(f"{r2text[r2id[r2]]} {e2desc[e2id[obj_z]]}")

    return list(neighbors), list(sentences)


def two_hop_types(df_triples, df_train, entity, t2desc, t2id):
    # Retreive columns
    o, r, s = df_triples.columns
    o_t, t = df_train.columns
    # Incomining and outcoming neighbors
    outgoing_neighbors = df_triples[df_triples[o] == entity].reset_index(drop=True)
    ingoing_neighbors = df_triples[df_triples[s] == entity].reset_index(drop=True)
    # Unique neighbor entities
    neighbors = set(outgoing_neighbors[s]) | set(ingoing_neighbors[o])
    # 2-hop types
    train_types_2hop = df_train[df_train[o_t].isin(neighbors)]
    train_types_2hop = train_types_2hop.drop_duplicates(subset=[t])
    types_2hop = train_types_2hop[t].to_list()
    # Sentences 2-hop types
    train_types_2hop.loc[:, t] = train_types_2hop[t].map(lambda typ: t2desc[t2id[typ]])
    train_types_2hop.loc[:, t] = train_types_2hop[t].str.replace(" [SEP] ", " ", regex=False)
    train_types_2hop.loc[:, t] = "has type " + train_types_2hop[t]
    sentences_2hop_type = train_types_2hop[t].to_list()

    return types_2hop, sentences_2hop_type


def max_sim_2hop(entity_1hop_txt, entity_2hop_txt, hop2_ids, n_best, kg=True):
    
    if len(hop2_ids) <= n_best:
        return hop2_ids

    # Encode KG sentences to get embeddings
    embeddings_1hop = model.encode(entity_1hop_txt, convert_to_numpy=True, batch_size=400)
    embeddings_2hop = model.encode(entity_2hop_txt, convert_to_numpy=True, batch_size=400)
    
    # Compute 2-hop neighbors with most similarity
    cos_sim_matrix = cosine_similarity(embeddings_1hop, embeddings_2hop)
    similarities_2hop = np.mean(cos_sim_matrix, axis=0)

    # Select best 2-hop kg neighbors
    if kg:
        top_2hop_neighbors = select_unique_kg_neighbors(similarities_2hop, hop2_ids, n_best)
    else:
        top_n_relations = list(np.argsort(similarities_2hop)[-n_best:][::-1])
        top_2hop_neighbors = [hop2_ids[i] for i in top_n_relations]

    return top_2hop_neighbors


def select_unique_kg_neighbors(similarities_2hop, hop2_ids, n_best):
    
    top_2hop_neighbors = []
    relations_added = set()
    top_selected = 0

    # Get the sorted indices descending by similarity
    sorted_relations = list(np.argsort(similarities_2hop)[::-1])

    for idx in sorted_relations:
        relation, entity, direction = hop2_ids[idx]
        if (relation in relations_added):
            continue
        relations_added.add(relation)
        top_2hop_neighbors.append((relation, entity, direction))
        top_selected += 1
        if top_selected == n_best:
            break

    return top_2hop_neighbors

def remove_noisy_neighbors(kg_entity_text, neighbors, et_train_sentences, et_train, n_kg_remove, n_et_remove):
    # Encode KG sentences to get embeddings
    kg_embeddings = model.encode(kg_entity_text, convert_to_numpy=True, batch_size=400)
    et_embeddings = model.encode(et_train_sentences, convert_to_numpy=True, batch_size=400)

    # Compute embeddings with most similarity
    cos_sim_matrix = cosine_similarity(kg_embeddings)
    similarities_kg_rels = np.mean(cos_sim_matrix, axis=0)
    cos_sim_matrix = cosine_similarity(et_embeddings)
    similarities_et_rels = np.mean(cos_sim_matrix, axis=0)

    # Sort by lowest similarity
    kg_removed = np.argsort(similarities_kg_rels)[:n_kg_remove]
    et_removed = np.argsort(similarities_et_rels)[:n_et_remove]

    # Filter removed relationships and types
    kg_train_removed = neighbors.iloc[kg_removed].reset_index(drop=True)
    et_train_removed = et_train.iloc[et_removed].reset_index(drop=True)

    return kg_train_removed, et_train_removed

def relationship_removed(kg_train_removed, entity):
    # Case where entity is entity1
    out_1 = kg_train_removed[kg_train_removed[0] == entity]
    out_1 = (out_1[1] + ' ' + out_1[2]).tolist()

    # Case where entity is entity2
    out_2 = kg_train_removed[kg_train_removed[2] == entity]
    out_2 = ( 'inv-' + out_2[1] + ' ' + out_2[0]).tolist()

    return out_1 + out_2

def plot_entity_metrics_distribution(e_coherence, result_folder):
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Plot distribution of kg_sim_mu
    sns.histplot(e_coherence['kg_sim_mu'], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title("Distribution of kg_sim_mu")
    axes[0].set_xlabel("kg_sim_mu")

    # Plot distribution of et_sim_mu
    sns.histplot(e_coherence['et_sim_mu'], kde=True, ax=axes[1], color='lightgreen')
    axes[1].set_title("Distribution of et_sim_mu")
    axes[1].set_xlabel("et_sim_mu")

    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, "dsitribution_metrics"))
    plt.close()


def save_entity_kg_2hop(entity_kg_2hop, output_tsv_path):
    """
    Sauvegarde les relations de type (entité, relation, entité2) dans un fichier TSV :
    entité<TAB>relation,entité2<TAB>relation,entité3...
    """
    ent_rel_dict = defaultdict(list)

    for head, rel, tail, direction in entity_kg_2hop:
        if head != tail: 
            ent_rel_dict[head].append(f"{rel},{tail},{direction}")

    with open(output_tsv_path, "w", encoding="utf-8") as f:
        for ent, rel_list in ent_rel_dict.items():
            f.write(f"{ent}\t" + "\t".join(rel_list) + "\n")