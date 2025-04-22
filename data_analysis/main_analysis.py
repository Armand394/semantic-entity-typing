import pandas as pd
import os
import json
from pathlib import Path
import json
from utils import *
from analysis_tools import *
from visualisations import *
from data_stats import *

# Specify locations for loading and saving data
project_folder = os.getcwd()
data_path = os.path.join(project_folder, "data", "FB15kET")
data_path_YG = os.path.join(project_folder, "data", "YAGO43kET")
result_folder = os.path.join(project_folder, "results")
figure_folder = os.path.join(project_folder, "results", "figures")

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
rank_df = pd.read_csv(os.path.join(data_path, "rank_FB15kET_original.txt"), header=None, sep='\t')

with open(os.path.join(data_path, 'entity_wiki.json'), "r") as f:
    entity_labels = json.load(f)

# General descripive statistics of Knowledge Graph
if not os.path.exists(os.path.join(result_folder, "FB15kET_neighbor_stat.csv")):
    run_stats_analysis(data_path, "FB15kET", result_folder)
    run_stats_analysis(data_path_YG, "YAGO43kET", result_folder)

plot_sample_graph(data_path, result_folder, "sample_FB")
plot_sample_graph(data_path_YG, result_folder, "sample_YG", FB=False)

# Recompute metrics for entities (--> efficient if GPU available)
if not os.path.exists(os.path.join(result_folder, "entity_metrics.csv")):
    recompute_similarity(df_triples, df_train, r2text, r2id, e2desc, e2id, t2desc, t2id, result_folder)

# Merge on entity
entity_metrics = pd.read_csv(os.path.join(result_folder, "entity_metrics.csv"))

# Process rank dataframe for average rank prediction of entity
rank_avg_df = avg_rank_entity(rank_df)

# Merge average rank of entities on metrics data
entity_metrics = pd.merge(entity_metrics, rank_avg_df, on='entity', how='left')
entity_metrics_preds = entity_metrics.copy()
entity_metrics_preds = entity_metrics_preds.dropna()

# ====== Percentage of types with rank above thresholds ======
rank_df.columns = ['entity', 'type', 'rank']
plot_percentages_ranks(rank_df, figure_folder)

# ===== Pair plot degree - similarity =====
plot_similarity_metrics_pairplots(entity_metrics_preds, figure_folder, "pairplot_degree_similarity")

# ====== Pair plot rank - metrics ======
plot_rank_value_pairplots(entity_metrics_preds, figure_folder, "pairplot_rank_metrics")

# ===== Average rank quantiles bins - 3 important metrics =====
rank_by_metric_barplot(entity_metrics_preds, figure_folder)

# ===== Transform and update columns and dataframes ======

# Create 'correct' prediction feature for entity
entity_metrics_preds.loc[:, 'y'] = (entity_metrics_preds["rank_value"] == 1).astype(int)
entity_metrics_low = entity_metrics_preds[entity_metrics_preds['degree'] < 15].copy(deep=True)
entity_metrics_high = entity_metrics_preds[entity_metrics_preds['degree'] > 135].copy(deep=True)

# Make copy of dataframe for scaling, fittings and plotting
metric_cols = list(entity_metrics_preds.columns[1:-2])
metric_cols.remove('degree')

# Analyze skewness of metrics before transforming
analyze_skewness(entity_metrics_preds, metric_cols, result_folder, "test_skewness_metrics", alpha=0.05)

# Apply log transform
entity_metrics_preds[metric_cols] = np.log1p(entity_metrics_preds[metric_cols])
entity_metrics_low[metric_cols]   = np.log1p(entity_metrics_low[metric_cols])
entity_metrics_high[metric_cols]  = np.log1p(entity_metrics_high[metric_cols])

# ====== Create boxplot for metrics - disitribution difference correct and wrong ======

# Boxplots
box_plot_metrics(entity_metrics_low, metric_cols, 'pred_metrics_low', figure_folder)
box_plot_metrics(entity_metrics_high, metric_cols, 'pred_metrics_high', figure_folder)
box_plot_metrics(entity_metrics_preds, metric_cols, 'pred_metrics', figure_folder)

# summary table
export_metric_summary_table(
    df_all=entity_metrics_preds,
    df_low=entity_metrics_low,
    df_high=entity_metrics_high,
    columns=metric_cols,
    output_folder=result_folder,
    file_name="metric_summary_by_class_and_degree",
    use_median=False  # Set to True to use medians instead of means
)

# ====== Logistic Regression fit ======
metric_cols.append('y')

entity_metrics_preds = entity_metrics_preds[metric_cols]
entity_metrics_low = entity_metrics_low[metric_cols]
entity_metrics_high = entity_metrics_high[metric_cols]

logistic_regression_fit(entity_metrics_preds, "all", result_folder)
logistic_regression_fit(entity_metrics_low, "low", result_folder)
logistic_regression_fit(entity_metrics_high, "high", result_folder)


# ====== Statistical-test distribution ======
stat_test_metric_distribution(entity_metrics_preds, result_folder, "all")
stat_test_metric_distribution(entity_metrics_low, result_folder, "low")
stat_test_metric_distribution(entity_metrics_high, result_folder, "high")