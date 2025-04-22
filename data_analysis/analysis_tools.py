import os
import tqdm
from collections import Counter
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, skew, normaltest


if torch.cuda.is_available():
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')  # Forces GPU
else:
    model = SentenceTransformer("all-MiniLM-L6-v2")

def recompute_similarity(df_triples, df_train, r2text, r2id, e2desc, e2id, t2desc, t2id, result_folder):
    """
    Only computes mean similarity for kg triples and training types (does not compute other metrics). For detailed
    metrics computation see main_analysis.py.
    """
    #Compute coherence metrics
    entities_kg = set(df_triples[0].unique()).union(set(df_triples[2].unique()))
    metrics = []

    # List of sentences & types with low coherence
    rlow = []
    elow = []

    for entity in tqdm(entities_kg, desc="Computing entity metrics", unit="Entity"):
        # Compute mean cosine similarity of KG sentences
        kg_entity_text, kg_neighbors = kg_sentences(df_triples, entity, r2text, r2id, e2desc, e2id, filter=False)
        base_sim_kg, rlow_sim_index = compute_mean_similarity(kg_entity_text)

        # Compute mean cosine similarity of ET sentences
        et_train_sentences, et_entity = et_sentences(df_train, entity, t2desc, t2id)
        base_sim_et, elow_sim_index = compute_mean_similarity(et_train_sentences)

        # Retreive relationship and type with low score
        rlow_sim = kg_neighbors[1].iloc[rlow_sim_index].to_list()
        elow_sim = et_entity[1].iloc[elow_sim_index].to_list()
        rlow.extend(rlow_sim)
        elow.extend(elow_sim)

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

    # Convert entity metrics in dataframe and save
    e_coherence = pd.DataFrame(metrics, columns=['entity', 'kg_sim_mu', 'et_sim_mu', 'degree',
                                                'avg_txt_length', 'kg_degree', 'et_degree','type_kg_ratio'])
    e_coherence.to_csv(os.path.join(result_folder, "entity_metrics.csv"), index=False)

    # Count the occurrences of relationships and types
    rlow_counts = Counter(rlow)
    elow_counts = Counter(elow)

    # Convert counts to DataFrame, then sort by count in descending order
    rlow_df = pd.DataFrame.from_dict(rlow_counts, orient='index', columns=['count']).reset_index()
    rlow_df.columns = ['relationship', 'count']
    rlow_df = rlow_df.sort_values(by='count', ascending=False)
    rlow_df.to_csv(os.path.join(result_folder, "relationships_noise.csv"), index=False)

    elow_df = pd.DataFrame.from_dict(elow_counts, orient='index', columns=['count']).reset_index()
    elow_df.columns = ['type', 'count']
    elow_df = elow_df.sort_values(by='count', ascending=False)
    elow_df.to_csv(os.path.join(result_folder, "types_noise.csv"), index=False)


def compute_mean_similarity(sentences):
    if len(sentences) < 2:
        return 0.5, np.array([])

    # Encode sentences to get an
    embedding_matrix = model.encode(sentences, convert_to_numpy=True, batch_size=200)

    # Compute cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(embedding_matrix)
    mean_coherence = np.mean(cosine_sim_matrix, axis=1)
    # Extract upper triangle without diagonal
    triu_indices = np.triu_indices_from(cosine_sim_matrix, k=1)
    cosine_sim_values = cosine_sim_matrix[triu_indices]

    # Compute mean cosine similarity and return low similarity sentences
    return np.mean(cosine_sim_values), np.argsort(mean_coherence)[:5]


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


def avg_rank_entity(rank_df):
    # Process dataframe
    rank_df[[1,2]] = rank_df[1].str.split(' ', expand=True)
    rank_df[2] = rank_df[2].astype(float)

    # Average rank values per entity
    rank_filter_df = rank_df.groupby(0)[2].mean().reset_index()
    rank_filter_df.columns = ['entity', 'rank_value']

    return rank_filter_df


def analyze_skewness(df, columns, result_folder, table_name, alpha=0.05):
    results = []

    for col in columns:
        data = df[col].dropna()
        sk = skew(data)
        try:
            stat, p = normaltest(data)
        except Exception:
            stat, p = np.nan, np.nan  # In case sample size is too small or constant data

        skewed = abs(sk) > 1  # Rule of thumb: > 1 or < -1 is highly skewed
        significant = p < alpha  # Normality test p-value < alpha means significant deviation

        results.append({
            "Metric": col,
            "Skewness": sk,
            "NormalTest_pval": p,
            "HighlySkewed": skewed,
            "StatisticallyNonNormal": significant
        })

    result_df = pd.DataFrame(results)

    # Save to CSV
    output_path = os.path.join(result_folder, f"{table_name}.csv")
    result_df.to_csv(output_path, index=False)


def export_metric_summary_table(df_all, df_low, df_high, columns, output_folder, file_name, use_median=False):
    """
    Computes average (or median) of selected metrics for y=0 and y=1
    across all, low-degree, and high-degree entities.
    Saves the result as a CSV table.
    """
    os.makedirs(output_folder, exist_ok=True)
    method = "median" if use_median else "mean"

    summary_data = {}

    for col in columns:
        summary_data[col] = {
            f"All (y=0)": df_all[df_all.y == 0][col].median() if use_median else df_all[df_all.y == 0][col].mean(),
            f"All (y=1)": df_all[df_all.y == 1][col].median() if use_median else df_all[df_all.y == 1][col].mean(),
            f"Low (y=0)": df_low[df_low.y == 0][col].median() if use_median else df_low[df_low.y == 0][col].mean(),
            f"Low (y=1)": df_low[df_low.y == 1][col].median() if use_median else df_low[df_low.y == 1][col].mean(),
            f"High (y=0)": df_high[df_high.y == 0][col].median() if use_median else df_high[df_high.y == 0][col].mean(),
            f"High (y=1)": df_high[df_high.y == 1][col].median() if use_median else df_high[df_high.y == 1][col].mean(),
        }

    result_df = pd.DataFrame.from_dict(summary_data, orient='index')
    result_df.index.name = "Metric"
    
    output_path = os.path.join(output_folder, f"{file_name}.csv")
    result_df.to_csv(output_path)


def logistic_regression_fit(entity_metrics_preds, fit_name, result_folder):
    # Select all columns except the first one
    X = entity_metrics_preds.iloc[:,:-1]
    y = entity_metrics_preds["y"]

    # Add a constant term for the intercept
    X = sm.add_constant(X)

    # Logistic Regression
    model = sm.Logit(y, X).fit()

    # save model summary
    output_file = os.path.join(result_folder, f"lr_summary_{fit_name}.txt")
    with open(output_file, "w") as f:
        f.write(model.summary().as_text())


def stat_test_metric_distribution(df, result_folder, file_name):

    results = []

    # Select only feature columns (excluding 'y' column if present)
    feature_columns = df.columns[df.columns != 'y'] 
       
    for column in feature_columns:
        variable, test_used, stat, p_val = test_mean_difference(df, column)
        results.append((variable, test_used, stat, p_val))

    # Convert to DataFrame and save to CSV
    df_results = pd.DataFrame(results, columns=['Variable', 'Test Used', 'Statistic', 'p-value'])
    df_results.to_csv(os.path.join(result_folder, f"stat_test_{file_name}.csv"), index=False)


def test_mean_difference(df, variable, target_column='y'):

    group_0 = df[df[target_column] == 0][variable]
    group_1 = df[df[target_column] == 1][variable]

    # Use t-test (Welch's if variances differ)
    t_stat, p_val = mannwhitneyu(group_0, group_1, alternative='two-sided')
    stat = t_stat
    test_used = "Mann-Whitney U t-test"

    return variable, test_used, stat, p_val