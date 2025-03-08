import pandas as pd


def preprocess_dataframes(df_triples, df_train, entity_labels ):
    
    # Add text in dataframes for triples in KG 
    df_triples = df_triples.rename(columns={df_triples.columns[0]: 'object', 
                                            df_triples.columns[1]: 'relation', 
                                            df_triples.columns[2]: 'subject'})

    # Adjust columns names for types in KG for clarity
    df_train = df_train.rename(columns={df_train.columns[0]: 'object', df_train.columns[1]: 'type'})

    # Convert entity ids into text label
    df_triples["object"] = df_triples["object"].map(
    lambda x: f"{entity_labels.get(x, {}).get('label', x)} {entity_labels.get(x, {}).get('description', '')}")

    df_triples.loc[:, "subject"] = df_triples["subject"].map(
    lambda x: f"{entity_labels.get(x, {}).get('label', x)} {entity_labels.get(x, {}).get('description', '')}")
    
    df_train.loc[:, "object"] = df_train["object"].map(
    lambda x: f"{entity_labels.get(x, {}).get('label', x)} {entity_labels.get(x, {}).get('description', '')}")

    return df_triples, df_train


def convert_type_df_to_text(df_type_text, df_type, entity_labels):
    
    # convert id object into string
    df_type.loc[:, "object"] = df_type["object"].map(lambda x: entity_labels.get(x, {}).get("label", x))

    # Text of types
    df_type_text = df_type_text.rename(columns={df_type_text.columns[0]: 'type', df_type_text.columns[1]: 'text'})
    # Join text and types to acquire text for each type in set
    df_types_kg = df_type.merge(df_type_text, left_on='type', right_on='type', suffixes=('_1', '_2'))
    
    # Remove [SEP] here as we only look at the semantic of actual text
    df_types_kg["text"] = df_types_kg["text"].str.replace("[SEP]", "", regex=False)
    # Add has type as prefix for type text
    df_types_kg["text"] = "has type " + df_types_kg['text']
    # Remove extra spaces
    df_types_kg['text'] = df_types_kg['text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    #Return only object and text column
    return df_types_kg[['object', 'text']]


def convert_entity_text(df_triples):
    
    # Create outgoing text column (relation, subject) per entity
    outgoing_texts = df_triples.groupby("object").apply(lambda x: set(zip(x["relation"], x["subject"])))

    # Create incoming text column (object, relation) per entity
    incoming_texts = df_triples.groupby("subject").apply(lambda x: set(zip(x["object"], x["relation"])))

    # Ensure every entity appears
    all_entities = set(df_triples["object"]).union(set(df_triples["subject"]))
    entity_texts = pd.DataFrame({"object": list(all_entities)})

    # Apply outgoing and incoming texts independently
    entity_texts["text"] = entity_texts["object"].apply(
        lambda x: sorted(outgoing_texts.get(x, set()) | incoming_texts.get(x, set()))
    )

    # Explode the text column so each entity appears on multiple rows
    entity_texts = entity_texts.explode("text")

    # Convert tuple values into pure text
    entity_texts["text"] = entity_texts["text"].apply(lambda x: f"{x[0]} {x[1]}" if isinstance(x, tuple) else str(x))

    return entity_texts


def process_rank_df(df_rank, entity_labels):
    # Adjust columns and columns names
    df_rank[['type', 'rank']] = df_rank[1].str.split(' ', n=1, expand=True)
    df_rank = df_rank.drop(columns=[1])
    df_rank = df_rank.rename(columns={ 0 : 'entity'})
    # Format columns
    df_rank['rank'] = df_rank['rank'].astype(int)
    df_rank.loc[:, "entity"] = df_rank["entity"].map(lambda x: entity_labels.get(x, {}).get("label", x))

    return df_rank


def rename_columns(df):

    return df.rename(columns={
        'avg_ttr': 'TTR (avg)',
        'triple_train_coherence_mean': '(r-tr) (mean)',
        'triple_train_coherence_std': '(r-tr) (std)',
        'triple_self_coherence_mean': 'r (AC) (mean)',
        'triple_self_coherence_std': 'r (AC) (std)',
        'type_self_coherence_mean': 'tr (AC) (mean)',
        'type_self_coherence_std': 'tr (AC) (std)',
        'direct_neighbors': 'dir_n',
        'type_triple_ratio': 'tr-r ratio',
        'avg_triple_length': 'r (length) (avg)',
        'avg_type_length': 'tr (length) (avg)',
    })