import pandas as pd
import os
import json
from pathlib import Path
import json
from utils import *
from skp_utils import *
import math
from tqdm import tqdm
from sampling_utils import *
import shutil

def main(dataset: str, augment: str, augment_2hop: bool, supress_noise: bool, augment_all: bool):

    # Raise error if incorrect augmentation method chosen
    if augment not in ["full", "selective", "concat"]:
        raise ValueError("Choose between full, selective, and concat augmentation methods.")

    # Specify locations for loading and saving data
    project_folder = os.getcwd()
    general_data_path = os.path.join(project_folder, "data")
    data_path = os.path.join(project_folder, "data", dataset)

    # Define augmented data
    if augment_2hop and supress_noise:
        augment_dataset = f"{dataset}_{augment}_as"
    elif augment_2hop:
        augment_dataset = f"{dataset}_{augment}_a"
    else:
        augment_dataset = f"{dataset}_{augment}_s"
    
    data_augment_path = os.path.join(project_folder, "data", augment_dataset)
    os.makedirs(data_augment_path, exist_ok=True)

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
    df_types = pd.read_csv(os.path.join(data_path, "ET_train.txt"), sep='\t', header=None)

    with open(os.path.join(data_path, 'entity_wiki.json'), "r") as f:
        entity_labels = json.load(f)

    # Recompute coherence metrics for entities
    if not os.path.exists(os.path.join(data_path, f"{dataset}_features.csv")):
        recompute_similarity(df_triples, df_types, r2text, r2id, e2desc, e2id, t2desc, t2id, data_path, dataset)

    # Load entity coherence metrics
    e_coherence = pd.read_csv(os.path.join(data_path, f"{dataset}_features.csv"))

    # Processed data
    df_types_processed = df_types.copy(deep=True)
    df_triples_processed = df_triples.copy(deep=True)

    if augment_2hop:
        # Entities with degree considered too low
        lowerb_quantiles = e_coherence[['kg_degree', 'et_degree']].quantile(0.1)
        low_kg_degree = lowerb_quantiles.loc['kg_degree']
        low_et_degree = lowerb_quantiles.loc['et_degree']

        # Filter only entities with bad metrics
        entity_low_degree_df = e_coherence[(e_coherence["kg_degree"] <= low_kg_degree) \
                                    & (e_coherence["et_degree"] <= low_et_degree)]
        entity_low_degree = entity_low_degree_df['entity'].to_list()

        # If we augment all entities, we add r2-hop information for all
        if augment_all:
            entity_low_degree = e_coherence["entity"].to_list()

        entity_kg_2hop = defaultdict(list)
        entity_et_2hop = []

        for entity in tqdm(entity_low_degree, total=len(entity_low_degree), desc="Processing entities", unit="entity"):
            
            # Current sentences
            kg_entity_text, _ = kg_sentences(df_triples, entity, r2text, r2id, e2desc, e2id)
            et_train_sentences, _ = et_sentences(df_types, entity, t2desc, t2id)
            entity_sentences = kg_entity_text + et_train_sentences

            # Number 2-hop relations
            kg_degree_min = min(len(kg_entity_text), low_kg_degree)
            n_rel = int(round(low_kg_degree * (1 - kg_degree_min / low_kg_degree)) + 2)

            # Add slightly increasing 2-hop information for high degree rel (when consider all entities)
            if len(kg_entity_text) > low_kg_degree:
                # Method 1 - log scale
                # bonus = math.log(len(kg_entity_text) + 1, 2) - 1
                # Method 2 - constant addition
                n_rel = int(round(n_rel + 1))

            # Number of types added
            et_degree_min = min(len(et_train_sentences), low_et_degree)
            n_type = int(round(low_et_degree*2* (1 - et_degree_min/ low_et_degree+1)) + 2)
            
            # 2-hop neighbor sentences
            if augment == 'concat':
                kg_2hop, kg_2hop_sentences = two_hop_neighbors_concat(df_triples, entity, r2text, r2id, e2desc, e2id)
            else:
                kg_2hop, kg_2hop_sentences = two_hop_neighbors(df_triples, entity, r2text, r2id, e2desc, e2id)
            
            types_2hop, et_txt_2hop = two_hop_types(df_triples, df_types, entity, t2desc, t2id)

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

        # Save
        with open(os.path.join(data_augment_path, 'relation2hop.json'), "w") as f:
            json.dump(entity_kg_2hop, f, indent=4)

        # Update ET_train with 2-hop types
        et_train_2hop = pd.DataFrame(entity_et_2hop, columns=[0,1])
        df_types_processed = pd.concat([df_types_processed, et_train_2hop], axis=0).reset_index(drop=True)


    if supress_noise:
        # Entities with degree considered too high
        upperb_quantiles = e_coherence[['kg_degree', 'et_degree']].quantile(0.80)
        high_kg_degree = upperb_quantiles.loc['kg_degree']
        high_et_degree = upperb_quantiles.loc['et_degree']

        # Filter only entities with bad metrics
        entity_high_degree_df = e_coherence[(e_coherence["kg_degree"] > high_kg_degree) \
                                    & (e_coherence["et_degree"] > high_et_degree)]
        entity_high_degree = entity_high_degree_df['entity'].to_list()

        # removed results dataframe
        kg_train_removed_df = pd.DataFrame()
        et_train_removed_df = pd.DataFrame()
        entity_kg_remove = defaultdict(list)

        for entity in tqdm(entity_high_degree, total=len(entity_high_degree), desc="Processing entities", unit="entity"):

            # Current sentences
            kg_entity_text, neighbors = kg_sentences(df_triples, entity, r2text, r2id, e2desc, e2id, filter=False)
            et_train_sentences, et_train = et_sentences(df_types, entity, t2desc, t2id)
            entity_sentences = kg_entity_text + et_train_sentences

            # Remove noisy relationships and types
            n_kg_remove = int(math.ceil(len(kg_entity_text)*0.050))
            n_et_remove = int(math.ceil(len(et_train_sentences)*0.050))

            # Remove noisy neighbors through similarity score
            kg_train_removed, et_train_removed = remove_noisy_neighbors(kg_entity_text, neighbors, et_train_sentences, et_train, n_kg_remove, n_et_remove)
            # Save removed relationships and types
            rel_removed = relationship_removed(kg_train_removed, entity)
            entity_kg_remove[entity] = rel_removed

            # Store removed results
            et_train_removed_df = pd.concat([et_train_removed_df, et_train_removed], axis=0).reset_index(drop=True)

        # Save
        with open(os.path.join(data_augment_path, 'relation2remove.json'), "w") as f:
            json.dump(entity_kg_remove, f, indent=4)

        # Update ET_train with supressed data
        df_types_processed = df_types_processed.merge(et_train_removed_df, how='left', indicator=True)
        df_types_processed = df_types_processed[df_types_processed['_merge'] == 'left_only'].drop(columns=['_merge'])
    

    # Save empty augmentation files if no specific augmentation
    if not supress_noise:
        with open(os.path.join(data_augment_path, 'relation2remove.json'), "w") as f:
            json.dump({}, f, indent=4)

    if not augment_2hop:
        with open(os.path.join(data_augment_path, 'relation2hop.json'), "w") as f:
            json.dump({}, f, indent=4)

    # Save processed types dataframe
    df_types_processed.to_csv(os.path.join(data_augment_path, 'ET_train.txt'), sep='\t', header=None, index=False)

    # Load processing dictionaries
    with open(os.path.join(data_augment_path,"relation2hop.json"), "r") as f:
        r2hop = json.load(f)
    with open(os.path.join(data_augment_path,"relation2remove.json"), "r") as f:
        r2move = json.load(f)

    # Full augmentation relationships (direct)
    if augment == 'full':

        # Convert into dataframe
        triples_2hop, _ = convert_dict2df(r2hop)

        # Added 2-hop triples to KG_train
        df_triples_processed = pd.concat([df_triples_processed, triples_2hop], axis=0).reset_index(drop=True)

        # Convert into dataframe
        triples_noise, _ = convert_dict2df(r2move)

        # Remove the noise relationships
        df_triples_processed = df_triples_processed.merge(triples_noise, on=[0, 1, 2], how='left', indicator=True)
        df_triples_processed = df_triples_processed[df_triples_processed['_merge'] == 'left_only'].drop(columns=['_merge'])

        # Save final KG
        df_triples_processed.to_csv(os.path.join(data_augment_path, 'KG_train.txt'), sep='\t', header=None, index=False)

        # Generate all remaining files with updated kg
        # ( --> DO NOT: add dictionaries, the relationships were directly altered in KG_train)
        generate_full_augmented_data(general_data_path, data_path, data_augment_path, dataset=augment_dataset,
                                        augment='full', r2hop={}, r2move={})

    # Selective augmentation relationships (in data building)
    elif augment == 'selective':
        generate_full_augmented_data(general_data_path, data_path, data_augment_path, dataset=augment_dataset, 
                                        augment='selective', r2hop=r2hop, r2move=r2move)

    
    # Concatenating augmentation
    elif augment == 'concat':
        
        # Copy original relation-id file to modify with concat r
        shutil.copy2(os.path.join(data_path, 'relations.tsv'), os.path.join(data_augment_path, 'relations.tsv'))

        # Convert into dataframe
        triples_2hop, concat_r_new = convert_dict2df(r2hop, concat=True)

        # Update id file, and text description of concatted relationships
        update_relation_tsv(os.path.join(data_augment_path, 'relations.tsv'), concat_r_new, r2id)
        generate_relation2text(os.path.join(data_augment_path, 'relations.tsv'), os.path.join(data_augment_path, 'relation2text.txt'))

        # Added 2-hop triples to KG_train
        df_triples_processed = pd.concat([df_triples_processed, triples_2hop], axis=0).reset_index(drop=True)

        # Convert into dataframe
        triples_noise, _ = convert_dict2df(r2move)

        # Remove the noise relationships
        df_triples_processed = df_triples_processed.merge(triples_noise, on=[0, 1, 2], how='left', indicator=True)
        df_triples_processed = df_triples_processed[df_triples_processed['_merge'] == 'left_only'].drop(columns=['_merge'])

        # Save final KG
        df_triples_processed.to_csv(os.path.join(data_augment_path, 'KG_train.txt'), sep='\t', header=None, index=False)

        # Generate all remaining files with updated kg
        # ( --> DO NOT: add dictionaries, the relationships were directly altered in KG_train)
        generate_full_augmented_data(general_data_path, data_path, data_augment_path, dataset=augment_dataset,
                                        augment='concat', r2hop={}, r2move={})


if __name__ == "__main__":
    main(dataset="FB15kET_sample", augment='concat', augment_2hop=True, supress_noise=False, augment_all=False)