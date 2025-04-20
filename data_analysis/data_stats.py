from pathlib import Path
import pandas as pd
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import networkx as nx
import json

def run_stats_analysis(data_folder, dataset, folder_result):

    df_entities, df_relations, df_types, df_triples, df_train, df_test, df_validate = load_data(data_folder)

    # Descriptive statistics
    nentities = df_entities.nunique()[0]
    nrelations = df_relations.nunique()[0]
    ntypes = df_types.nunique()[0]
    ntriples = df_triples.count()[0]
    ntrain = df_train.count()[0]
    ntest = df_test.count()[0]
    nvalidate = df_validate.count()[0]

    # Create a dictionary of metrics
    stats = {
        "Entities": nentities,
        "Relations": nrelations,
        "Types": ntypes,
        "Triples": ntriples,
        "Train": ntrain,
        "Valid": nvalidate,
        "Test": ntest
    }

    # Convert to DataFrame with 'Metric' and 'Value' columns
    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])

    # Save as TSV
    stats_df.to_csv(os.path.join(folder_result, f"{dataset}_general_stat.tsv"), sep='\t', index=False)

    # Rename columns for clarity
    df_triples = df_triples.rename(columns={df_triples.columns[0]: 'object', 
                                            df_triples.columns[1]: 'relation', 
                                            df_triples.columns[2]: 'subject'})

    df_train = df_train.rename(columns={df_train.columns[0]: 'entity', df_train.columns[1]: 'type'})
    df_test = df_test.rename(columns={df_test.columns[0]: 'entity', df_test.columns[1]: 'type'})
    
    # Compute outgoing and incoming degree
    outgoing_counts = df_triples.groupby('object')['subject'].nunique()
    incoming_counts = df_triples.groupby('subject')['object'].nunique()
    
    # Sum the two counts for of 1-hop neighbors
    total_neighbors = outgoing_counts.add(incoming_counts, fill_value=0)  # Adds both, treating NaNs as 0

    # Compute descriptive statistics for the total 1-hop neighbors
    descriptive_statistics = total_neighbors.describe().to_frame().T
    descriptive_statistics.index = ["1-hop neighbors"]

    # Direct types
    type_counts = df_train.groupby(df_train.columns[0])[df_train.columns[1]].nunique()
    
    # Compute descriptive statistics for direct types
    direct_types = type_counts.describe().to_frame().T
    direct_types.index = ["1-hop Types"]
    descriptive_statistics = pd.concat([descriptive_statistics, direct_types], axis=0)

    # Retrieve columns
    e_col, r_col, s_col = df_triples.columns

    # Step 1: Construct adjacency lists for incoming & outgoing relations
    outgoing_map = defaultdict(set)
    incoming_map = defaultdict(set) 

    # Fill adjacency maps (avoiding duplicate storage)
    for obj, subj in zip(df_triples[e_col], df_triples[s_col]):
        outgoing_map[obj].add(subj)
        incoming_map[subj].add(obj)

    # Step 2: Compute unique 2-hop neighbors per entity
    two_hop_neighbors = defaultdict(set)

    for entity in set(outgoing_map.keys()).union(set(incoming_map.keys())):
        # CASE 1: (object_x --> object_y --> object_z)
        for mid in outgoing_map[entity]:  # object_x -> mid
            two_hop_neighbors[entity].update(outgoing_map[mid])  # mid -> object_z

        # CASE 2: (object_x <-- object_y <-- object_z)
        for mid in incoming_map[entity]:  # object_x <- mid
            two_hop_neighbors[entity].update(incoming_map[mid])  # mid <- object_z

        # CASE 3: (object_x --> object_y <-- object_z)
        for mid in outgoing_map[entity]:  # object_x -> mid
            two_hop_neighbors[entity].update(incoming_map[mid])  # object_z -> mid

        # CASE 4: (object_x <-- object_y --> object_z)
        for mid in incoming_map[entity]:  # object_x <- mid
            two_hop_neighbors[entity].update(outgoing_map[mid])  # mid -> object_z

    # Ccount unique 2-hop neighbors per entity
    unique_2hop_neighbors = pd.Series({key: len(value) for key, value in two_hop_neighbors.items()})

    # Compute descriptive statistics for unique 2-hop neighbors
    hop2_stats = unique_2hop_neighbors.describe().to_frame().T
    hop2_stats.index = ["2-hop neighbors"]

    # Merge with previous statistics
    descriptive_statistics = pd.concat([descriptive_statistics, hop2_stats], axis=0)

    # Rename columns for joining
    df_train = df_train.rename(columns={df_train.columns[0]: 'entity', df_train.columns[1]: 'type'})

    # Extract column names
    e_col, r_col, s_col = df_triples.columns 
    entity_col, type_col = df_train.columns

    # Get types of direct neighbors
    df_1hop_types = df_triples.merge(df_train, left_on=s_col, right_on=entity_col)

    # Count unique types per entity (object)
    type_counts_1hop = df_1hop_types.groupby(e_col)[type_col].nunique()

    # Compute descriptive statistics for 1-hop neighbors
    hop1_t_sats = type_counts_1hop.describe().to_frame().T
    hop1_t_sats.index = ["2-hop types"]
    descriptive_statistics = pd.concat([descriptive_statistics, hop1_t_sats], axis=0)

    # Save results
    file_result = os.path.join(folder_result, f"{dataset}_neighbor_stat.csv")
    descriptive_statistics.to_csv(file_result, index=True, header=True)  



def plot_sample_graph(data_folder, result_folder, fig_name, FB=True):
    
    # Figure plot location
    figure_result = os.path.join(result_folder, "figures", fig_name + ".png")

    # Load data
    df_entities, df_relations, df_types, df_triples, df_train, df_test, df_validate = load_data(data_folder)

    # Rename columns for clarity
    df_triples = df_triples.rename(columns={df_triples.columns[0]: 'object', 
                                            df_triples.columns[1]: 'relation', 
                                            df_triples.columns[2]: 'subject'})

    df_train = df_train.rename(columns={df_train.columns[0]: 'entity', df_train.columns[1]: 'type'})
    df_test = df_test.rename(columns={df_test.columns[0]: 'entity', df_test.columns[1]: 'type'})

    # Acquire sample entity for example graph
    entity_example = select_sample_entity(df_triples, df_train, df_test)
    filtered_triples, filtered_train, filtered_test = filter_dataframes_by_entity(entity_example, df_triples, df_train, df_test)

    # Save result
    plot_entity_graph(entity_example, filtered_triples, filtered_train, filtered_test, figure_result, data_folder, FB=FB)


def select_sample_entity(df_triples, df_train, df_test):
    # Compute number of neighbors per entity
    neighbor_counts = df_triples.groupby('object')['subject'].nunique()

    # Compute number of types per entity
    type_counts = df_train.groupby('entity')['type'].nunique()

    # Compute number of missing types per entity from df_test
    missing_types_counts = df_test.groupby('entity')['type'].nunique()

    # Merge all counts into a single DataFrame
    df_stats = pd.DataFrame({'neighbors': neighbor_counts, 'types': type_counts, 'missing_types': missing_types_counts}).fillna(0)

    # Filter entities satisfying the conditions
    filtered_entities = df_stats[
        (df_stats['neighbors'].between(3, 4)) & 
        (df_stats['types'].between(1, 2)) & 
        (df_stats['missing_types'].between(1, 1))
    ]

    # Select entity 
    if not filtered_entities.empty:
        selected_entity = filtered_entities.index[1] 
        return str(selected_entity)
    else:
        return None


def filter_dataframes_by_entity(entity_example, df_triples, df_train, df_test):

    # Filter df_triples where the entity appears as 'object'
    filtered_df_triples = df_triples[df_triples['object'] == entity_example]

    # Filter df_train where the entity appears
    filtered_df_train = df_train[df_train['entity'] == entity_example]

    # Filter df_test where the entity appears
    filtered_df_test = df_test[df_test['entity'] == entity_example]

    return filtered_df_triples, filtered_df_train, filtered_df_test

def plot_entity_graph(entity_example, filtered_triples, filtered_train, filtered_test, figure_result, datafolder, FB=True):
    
    # Create a directed graph
    G = nx.DiGraph()

    # Initialize color map for nodes
    node_colors = {}
    
    #json file
    entity_labels_file = os.path.join(datafolder, "entity_wiki.json")
    
    if FB == True:
        with open(entity_labels_file, "r") as f:
            entity_labels = json.load(f)

        # Replace IDs with labels by extracting the "label" field from the dictionary
        filtered_triples.loc[:, "object"] = filtered_triples["object"].map(lambda x: entity_labels.get(x, {}).get("label", x))
        filtered_triples.loc[:, "subject"] = filtered_triples["subject"].map(lambda x: entity_labels.get(x, {}).get("label", x))

        entity_example = entity_labels[entity_example]["label"]

    # Add central entity
    G.add_node(entity_example)
    node_colors[entity_example] = "red"  # Central entity

    # Add edges and classify nodes
    for _, row in filtered_triples.iterrows():
        G.add_edge(entity_example, row["subject"], label=row["relation"])
        if row["subject"] not in node_colors:
            node_colors[row["subject"]] = "lightblue"  # Neighbors

    for _, row in filtered_train.iterrows():
        G.add_edge(entity_example, row["type"], label="has_type")
        if row["type"] not in node_colors:
            node_colors[row["type"]] = "green"  # Known types

    for _, row in filtered_test.iterrows():
        G.add_edge(entity_example, row["type"], label="has_type")
        if row["type"] not in node_colors:
            node_colors[row["type"]] = "orange"  # Missing types

    # Assign colors based on the node classification
    color_map = [node_colors.get(node, "gray") for node in G.nodes()]

    # Plot the graph
    plt.figure(figsize=(9,6))
    pos = nx.spring_layout(G, k=0.1)

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color=color_map, edge_color="gray", node_size=2000, font_size=8)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    nx.kamada_kawai_layout(G)

    # Add legend
    legend_labels = {
        "red": "Central Entity",
        "lightblue": "Neighbors",
        "green": "Known Types",
        "orange": "Missing Types"
    }

    for color, label in legend_labels.items():
        plt.scatter([], [], color=color, label=label)

    plt.legend()

    # Show the graph
    plt.title(f"Graph for Entity: {entity_example}")
    plt.savefig(figure_result)

def load_data(data_folder):
    # Specify files 1
    entities_path = os.path.join(data_folder, "entities.tsv")
    relations_path = os.path.join(data_folder, "relations.tsv")
    types_path = os.path.join(data_folder, "types.tsv")
    triples_path = os.path.join(data_folder, "KG_train.txt")
    train_path = os.path.join(data_folder, "ET_train.txt")
    test_path = os.path.join(data_folder, "ET_test.txt")
    validate_path = os.path.join(data_folder, "ET_valid.txt")

    # Open files
    df_entities = pd.read_csv(entities_path, sep='\t', header=None)
    df_relations = pd.read_csv(relations_path, sep='\t', header=None)
    df_types = pd.read_csv(types_path, sep='\t', header=None)
    df_triples = pd.read_csv(triples_path, sep='\t', header=None)
    df_train = pd.read_csv(train_path, sep='\t', header=None)
    df_test = pd.read_csv(test_path, sep='\t', header=None)
    df_validate = pd.read_csv(validate_path, sep='\t', header=None)

    return df_entities, df_relations, df_types, df_triples, df_train, df_test, df_validate


