from tools_kg import *
from data_cleaning import *
from data_visualization import *
from analysis import load_data
from pathlib import Path
import os
import json
import pandas as pd
import statsmodels.api as sm
from wordfreq import top_n_list
from sklearn.metrics import accuracy_score

# Specify locations for loading and saving data
project_folder = Path(__file__).parent.parent
data_folder = os.path.join(project_folder, "Data")
result_folder = os.path.join(project_folder, "Results", "Rank Performance")
figures_folder = os.path.join(project_folder, "Results", "Rank Performance", "Figures")

# Specify data folder
data_folder_FB = os.path.join(data_folder, "FB15kET")
data_folder_YG = os.path.join(data_folder, "YAGO43kET")

# wiki FB
entity_labels_FB = os.path.join(data_folder_FB, 'entity_wiki.json')

with open(entity_labels_FB, "r") as f:
    entity_labels = json.load(f)

# Load all data
df_entities, df_relations, df_types, df_triples, df_train, df_test, df_validate = load_data(data_folder_FB)
df_type_text = pd.read_csv(os.path.join(data_folder_FB, 'hier_type_desc.txt'), sep='\t', header=None)
df_rank = pd.read_csv(os.path.join(data_folder_FB, "rank_FB15kET_original.txt"), sep='\t', header=None)

# preprocess rank dataframe for easy manipulation
df_rank = process_rank_df(df_rank, entity_labels)

# Plot percentage of types with rank above thresholds
plot_percentages_ranks(df_rank, figures_folder)

# Preprocess triples and train types
df_triples, df_train = preprocess_dataframes(df_triples, df_train, entity_labels)

# Retreive entities with good and bad predictions
correct_entities, wrong_entities = classification_entities(df_rank, top_hit=1, undersampling=False)

# Plot top relations for good and bad
plot_top_relationships(correct_entities, wrong_entities, df_triples, result_folder=figures_folder)
plot_top_types(correct_entities, wrong_entities, df_train, result_folder=figures_folder)

# Text of relations
relation_mapping = pd.read_csv(os.path.join(data_folder_FB, 'relation2text.txt'), sep="\t", header=None, names=["relation_id", "relation_text"])
relation_dict = dict(zip(relation_mapping["relation_id"], relation_mapping["relation_text"]))
df_triples['relation'] = df_triples['relation'].map(relation_dict).fillna(df_triples['relation'])

# Create text for outgoing and incoming arcs of each entity
df_triples_text = convert_entity_text(df_triples)

# Text of types
df_train_type_txt = convert_type_df_to_text(df_type_text, df_train)

# Final dataframe for KG train text data
df_KG_train_text = pd.concat([df_triples_text, df_train_type_txt])

# Retreive entities with good and bad predictions
correct_entities, wrong_entities = classification_entities(df_rank, top_hit=1, undersampling=True)

# Define file paths
correct_file_path = os.path.join(result_folder, "data_metrics_good_cl.csv")
bad_file_path = os.path.join(result_folder, "data_metrics_bad_cl.csv")

# Check if the file for correctly classified entities exists
if os.path.exists(correct_file_path):
    print('correct classif - loading existing metrics')
    df_stats_correct = pd.read_csv(correct_file_path)
else:
    print('correct classif - start compute metrics')
    df_stats_correct = compute_metrics_statistics(correct_entities, df_KG_train_text,
                                                  df_triples_text, df_train_type_txt)
    df_stats_correct.to_csv(correct_file_path, index=False)

# Check if the file for wrongfully classified entities exists
if os.path.exists(bad_file_path):
    print('bad classif - loading existing metrics')
    df_stats_bad = pd.read_csv(bad_file_path)
else:
    print('bad classif - start compute metrics')
    df_stats_bad = compute_metrics_statistics(wrong_entities, df_KG_train_text,
                                              df_triples_text, df_train_type_txt)
    df_stats_bad.to_csv(bad_file_path, index=False)


# Rename columns
df_stats_correct = rename_columns(df_stats_correct)
df_stats_bad = rename_columns(df_stats_bad)

plot_metrics_classif(df_stats_correct, df_stats_bad, "(r-tr) (mean)", "tr (AC) (mean)", figures_folder,
                     xlabel=r"RTcC_{\mu}", ylabel=r"TC_{\mu}")

plot_metrics_classif(df_stats_correct, df_stats_bad, "(r-tr) (mean)", "TTR (avg)", figures_folder,
                     xlabel=r"RTcC_{\mu}", ylabel=r"TTT_{\mu}")

plot_metrics_classif(df_stats_correct, df_stats_bad, "r (AC) (std)", "(r-tr) (std)", figures_folder,
                     xlabel=r"RC_{\sigma}", ylabel=r"RTcC_{\sigma}")

# Add the classification column
df_stats_correct["y"] = 1
df_stats_bad["y"] = 0 

# Dataframe with final features
df_final = pd.concat([df_stats_correct, df_stats_bad], ignore_index=True)
df_final = df_final.drop(columns=["entity"])

# Compute descriptive statistics features
descriptive_stats_features(df_final, result_folder)

# Select all columns except the first one
X = df_final.iloc[:,:-1]
y = df_final["y"]

# Add a constant term for the intercept
X = sm.add_constant(X)

# Logistic Regression
model = sm.Logit(y, X).fit()

# Make predictions (probabilities)
y_pred_prob = model.predict(X)

# Convert probabilities to class labels (threshold = 0.5)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Compute accuracy
accuracy = accuracy_score(y, y_pred)
print(accuracy)

# Print and save model summary
print(model.summary())
output_file = os.path.join(result_folder,"logistic_regression_summary.txt")
with open(output_file, "w") as f:
    f.write(model.summary().as_text())


# Do the test of differences in distribution between correct and wrong for variables
output_tests = os.path.join(result_folder, "stat_test_distribution.csv")
df_test_results = apply_statistical_tests_and_save(df_final, output_tests)

print('start clustering')
# Find mean embedding for each entity
entity_embeddings = compute_mean_embeddings(df_KG_train_text)
# Find clusters
optimal_k = find_optimal_clusters(entity_embeddings, max_clusters=500, result_folder=figures_folder)
print('optimal cluster', optimal_k)
# Cluster entities
clustered_entities = cluster_entities(entity_embeddings, num_clusters=optimal_k)
# Label clusters
word_list = top_n_list("en", 1000)
cluster_labels = label_clusters_with_nearest_words(clustered_entities, entity_embeddings, word_list)

# Plot clusters
frequent_clusters_plot(cluster_labels, clustered_entities, correct_entities, wrong_entities, result_folder=figures_folder)