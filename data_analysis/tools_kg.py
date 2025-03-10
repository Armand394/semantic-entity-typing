# RImport necessary libraries after execution reset
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
import os
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import KMeans

# Apply global settings
plt.rcParams.update({
    'axes.spines.right': False,   # Enable right spine (solid)
    'axes.spines.top': False,     # Enable top spine (solid)
    'axes.grid': True,           # Enable grid
    'grid.alpha': 0.5,           # Make the grid transparent (adjust alpha)
    'xtick.direction': 'in',     # Tickmarks on x-axis (inside)
    'ytick.direction': 'in',     # Tickmarks on y-axis (inside)
    'grid.linestyle': '--',      # Dashed grid (can be changed)
    'axes.edgecolor': 'black',   # Ensure spines are visible
    'axes.linewidth': 1.2        # Make spines slightly thicker
})


# Load the SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Metric to compute token to type ratio
def token_to_type_ratio(text):
    """
    Computes the lexical diversity of a sentence.
    It measures how varied the vocabulary is within a given text.
    """
    words = text.lower().split()
    unique_words = set(words)
    return len(unique_words) / len(words) if len(words) > 0 else 0


# Compute coherence of train sentences for entity
def entity_train_coherence(sentences):

    # If only one sentence indecisive so return 0.5
    if len(sentences) < 2:
        return 0.5, 1.0

    # Encode sentences
    embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute simalirity between sentences
    similarity_scores = [
        util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
        for i in range(len(sentences)) for j in range(i + 1, len(sentences))
    ]

    # Compute mean and standard deviation
    mean_coherence = np.mean(similarity_scores) if similarity_scores else 0.0
    std_coherence = np.std(similarity_scores) if similarity_scores else 0.0

    return mean_coherence, std_coherence


def coherence_sets_sentences(sentence_set1, sentence_set2):

    if not sentence_set1 or not sentence_set2:
        return 0.0, 0.0

    # Encode train and test sentences
    train_embeddings = model.encode(sentence_set1, convert_to_tensor=True)
    test_embeddings = model.encode(sentence_set2, convert_to_tensor=True)

    # Compute cosine similarities between each train sentence and each test sentence
    similarity_scores = [
        util.pytorch_cos_sim(train_embeddings[i], test_embeddings[j]).item()
        for i in range(len(sentence_set1)) for j in range(len(sentence_set2))
    ]

    # Compute mean and standard deviation
    mean_coherence = np.mean(similarity_scores) if similarity_scores else 0.0
    std_coherence = np.std(similarity_scores) if similarity_scores else 0.0

    return mean_coherence, std_coherence

# Average character length of sentences for an entity 
def average_char_length(sentences):
    # total sentences
    num_sentences = len(sentences)
    # Average length
    avg_triple_length = sum(len(text) for text in sentences) / num_sentences if num_sentences > 0 else 0

    return avg_triple_length


# Compute Metrics Statistics Function
def compute_metrics_statistics(entities, df_KG, df_triples_text, df_train_text):

    # Filter df_KG to only include entities in training set
    df_filtered = df_KG[df_KG['object'].isin(entities)].copy()
    
    # Compute Token-to-Type Ratio (TTR) for each row
    df_filtered['ttr'] = df_filtered['text'].apply(token_to_type_ratio)

    df_stats = df_filtered.groupby('object').agg({
        'ttr': ['mean'],
    }).reset_index()

    # Rename columns
    df_stats.columns = ['entity', 'avg_ttr']

    entity_coherence_scores = []

    for entity, group in df_filtered.groupby('object'):
        # All training sentences for entity
        kg_train_sentences = group['text'].tolist()
        # Triple-based sentences for the entity
        triple_sentences = df_triples_text[df_triples_text['object'] == entity]['text'].tolist()
        # Type-based sentences for the entity
        train_type_sentences = df_train_text[df_train_text['object'] == entity]['text'].tolist()

        # Mean and standard deviation of coherence between train triples and train types
        coherence_train_mean, coherence_train_std = coherence_sets_sentences(triple_sentences, train_type_sentences)

        # Mean and standard deviation of coherence within entity’s train triples
        coherence_triples_mean, coherence_triples_std = entity_train_coherence(triple_sentences)

        # Mean and standard deviation of coherence within entity’s train types
        coherence_types_mean, coherence_types_std = entity_train_coherence(train_type_sentences)

        # Number of direct neighbors
        ndirect_neighbors = len(kg_train_sentences)

        # Ratio of triples vs. types in train data
        num_triples = len(triple_sentences)
        num_types = len(train_type_sentences)
        type_triple_ratio = num_types / num_triples if num_triples > 0 else 0 

        # Compute average character length 
        avg_triple_length = average_char_length(triple_sentences)
        avg_type_length = average_char_length(train_type_sentences)

        # Append results
        entity_coherence_scores.append((entity, coherence_train_mean, coherence_train_std,
                                        coherence_triples_mean, coherence_triples_std,
                                        coherence_types_mean, coherence_types_std,
                                        ndirect_neighbors, type_triple_ratio,
                                        avg_triple_length, avg_type_length))

    # Convert results to DataFrame
    df_coherence = pd.DataFrame(entity_coherence_scores, columns=[
        'entity', 'triple_train_coherence_mean', 'triple_train_coherence_std',
        'triple_self_coherence_mean', 'triple_self_coherence_std',
        'type_self_coherence_mean', 'type_self_coherence_std',
        'direct_neighbors', 'type_triple_ratio',
        'avg_triple_length','avg_type_length'])

    # Merge with df_stats to include entity coherence
    df_final = df_stats.merge(df_coherence, on='entity', how='left')

    return df_final


# creates list of entities with correct and wrong classification
def classification_entities(df_rank, top_hit=10, undersampling=False):
    # Acquire entities where type classification result were poor
    df_bad_classif = df_rank[df_rank['rank'] > top_hit]
    df_corr_classif = df_rank[df_rank['rank'] <= top_hit]

    # Identify entities that appear in both dataframes
    common_entities = set(df_bad_classif['entity']).intersection(set(df_corr_classif['entity']))

    # Remove these entities from df_corr_classif
    df_corr_classif = df_corr_classif[df_corr_classif['entity'] != common_entities]

    # List of unique entities with wrong and correct ranks
    correct_entities = list(df_corr_classif['entity'].unique())
    wrong_entities = list(df_bad_classif['entity'].unique())

    if undersampling:
        correct_entities = undersample_correct_entities(correct_entities, wrong_entities)

    return list(correct_entities), wrong_entities

def undersample_correct_entities(correct_entities, wrong_entities, seed=42):

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Determine the target size (same as wrong_entities)
    target_size = len(wrong_entities)

    # Randomly sample from correct_entities to match the size of wrong_entities
    undersampled_correct_entities = np.random.choice(correct_entities, size=target_size, replace=False)

    return undersampled_correct_entities

def descriptive_stats_features(df_final, result_folder):
    # Compute descriptive statistics separately for y=0 and y=1
    df_descriptive_stats_y0 = df_final[df_final['y'] == 0].describe()
    df_descriptive_stats_y1 = df_final[df_final['y'] == 1].describe()

    # Rename columns to indicate class (y=0 or y=1)
    df_descriptive_stats_y0.columns = [f"{col} (0)" for col in df_descriptive_stats_y0.columns]
    df_descriptive_stats_y1.columns = [f"{col} (1)" for col in df_descriptive_stats_y1.columns]

    # Concatenate both statistics DataFrames side by side
    df_combined_stats = pd.concat([df_descriptive_stats_y0, df_descriptive_stats_y1], axis=1)
    df_combined_stats = df_combined_stats.round(3)

    # Save to CSV file
    output_file = os.path.join(result_folder,"descriptive_statistics_by_class.csv")
    df_combined_stats.to_csv(output_file, index=True)

def test_mean_difference(df, variable, target_column='y'):

    group_0 = df[df[target_column] == 0][variable]
    group_1 = df[df[target_column] == 1][variable]

    # Use t-test (Welch's if variances differ)
    t_stat, p_val = ttest_ind(group_0, group_1, equal_var=False)
    stat = t_stat
    test_used = "Welch's t-test"

    return variable, test_used, stat, p_val


def apply_statistical_tests_and_save(df, output_file):

    results = []

    # Select only feature columns (excluding 'y' column if present)
    feature_columns = df.columns[df.columns != 'y'] 
       
    for column in feature_columns:
        variable, test_used, stat, p_val = test_mean_difference(df, column)
        results.append((variable, test_used, stat, p_val))

    # Convert to DataFrame and save to CSV
    df_results = pd.DataFrame(results, columns=['Variable', 'Test Used', 'Statistic', 'p-value'])
    df_results.to_csv(output_file, index=False)

    return df_results


def compute_mean_embeddings(df_KG_train_text, model_name="all-MiniLM-L6-v2"):

    # Load SBERT model
    model = SentenceTransformer(model_name)

    # Dictionary to store mean embeddings
    entity_embeddings = {}

    # Group text descriptions by entity
    grouped_texts = df_KG_train_text.groupby("object")["text"].apply(list)

    # Iterate over each entity and compute mean embedding
    for entity, sentences in grouped_texts.items():
        # Compute embeddings for each sentence
        embeddings = model.encode(sentences, convert_to_tensor=True)
        
        # Compute mean embedding vector
        mean_embedding = embeddings.mean(dim=0).cpu().numpy()

        # Store in dictionary
        entity_embeddings[entity] = mean_embedding

    return entity_embeddings


def find_optimal_clusters(entity_embeddings, max_clusters=15, random_state=42, result_folder=None):

    entities = list(entity_embeddings.keys())
    embeddings = np.array(list(entity_embeddings.values()))

    inertia = []
    cluster_range = range(5, max_clusters + 1)
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(embeddings)
        inertia.append(kmeans.inertia_)  # Inertia is the sum of squared distances to cluster centers

    # Plot Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, inertia, marker='o', linestyle='-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.savefig(os.path.join(result_folder, "clustering_k_inertia.png"))
    plt.close()

    # Find the "elbow point" manually (or use automatic detection methods)
    optimal_k = cluster_range[np.argmin(np.diff(inertia, 2))]  # Second derivative heuristic

    return optimal_k


def cluster_entities(entity_embeddings, num_clusters=10, random_state=42):

    # Extract entity names and corresponding embeddings
    entities = list(entity_embeddings.keys())
    embeddings = np.array(list(entity_embeddings.values()))

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Organize entities by cluster label
    clustered_entities = {i: [] for i in range(num_clusters)}
    for entity, label in zip(entities, labels):
        clustered_entities[label].append(entity)

    return clustered_entities


def label_clusters_with_nearest_words(clustered_entities, entity_embeddings, word_list):

    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Encode the entire word list into embedding space
    word_embeddings = model.encode(word_list)

    cluster_labels = {}

    for cluster, entities in clustered_entities.items():
        # Get entity embeddings for this cluster
        cluster_vectors = [entity_embeddings[e] for e in entities if e in entity_embeddings]

        if not cluster_vectors:
            cluster_labels[cluster] = "Unknown"
            continue

        # Compute mean embedding for the cluster
        mean_cluster_embedding = np.mean(cluster_vectors, axis=0).reshape(1, -1)

        # Compute cosine similarity with the word embeddings
        similarities = util.pytorch_cos_sim(mean_cluster_embedding, word_embeddings).flatten()

        # Find the most similar word
        best_word = word_list[np.argmax(similarities)]
        cluster_labels[cluster] = best_word

    return cluster_labels