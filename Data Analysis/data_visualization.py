import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np


def plot_percentages_ranks(df_rank, result_folder):
    # Compute percentage of types with rank above thresholds
    thresholds = [1, 3, 10, 100, 1000]
    percentages = [(df_rank[df_rank['rank'] > t].shape[0] / df_rank.shape[0]) * 100 for t in thresholds]

    # Generate shades of blue using Matplotlib's colormap
    colors = plt.cm.Blues(np.linspace(0.4, 1, len(thresholds))) 

    # Create bar plot
    plt.figure(figsize=(6, 4))
    bars = plt.bar([str(t) for t in thresholds], percentages, color=colors)

    # Add labels on top of the bars
    for bar, percentage in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{percentage:.1f}%", 
                ha='center', va='bottom', fontsize=10)

    # Labels and title
    plt.xlabel(" > @hit")
    plt.ylabel("%")

    # Figure plot location
    figure_result = os.path.join(result_folder, "distribution_ranks.png")

    # Display the plot
    plt.savefig(figure_result)
    plt.close()


def plot_metrics_classif(df_stats_correct, df_stats_bad, metric1, metric2, result_folder, xlabel=None, ylabel=None):
    # Initialize figure
    plt.figure(figsize=(8, 6))

    # Plot KDE distributions focusing on core density
    sns.kdeplot(x=df_stats_correct[metric1], y=df_stats_correct[metric2], cmap="Blues", alpha=0.8, fill=False, levels=5)
    sns.kdeplot(x=df_stats_bad[metric1], y=df_stats_bad[metric2], cmap="Reds", alpha=0.65, fill=False, levels=5)

    # Set axis labels with LaTeX formatting
    plt.xlabel(rf"${xlabel}$" if xlabel else metric1)
    plt.ylabel(rf"${ylabel}$" if ylabel else metric2)

    # Define legend handles
    legend_handles = [mpatches.Patch(facecolor="blue", alpha=0.8, label="Correct"),
                      mpatches.Patch(facecolor="red", alpha=0.65, label="Wrong")]

    # Add legend
    plt.legend(handles=legend_handles, loc="upper right")

    # Figure plot location
    figure_result = os.path.join(result_folder, f"classification_{metric1}_{metric2}_results.png")

    # Save and close plot
    plt.savefig(figure_result)
    plt.close()


def plot_top_relationships(entities_good_topn, entities_bad_topn, df_triples, result_folder):

    # Filter df_triples for good and bad classified entities
    df_good_topn = df_triples[df_triples['object'].isin(entities_good_topn)]
    df_bad_topn = df_triples[df_triples['object'].isin(entities_bad_topn)]

    # Count the top 10 most frequent relations for both groups
    top_relations_good = df_good_topn['relation'].value_counts().head(10)
    top_relations_bad = df_bad_topn['relation'].value_counts().head(10)

    # Identify common relations between the two sets
    common_relations = set(top_relations_good.index).intersection(set(top_relations_bad.index))

    # Function to assign colors
    def assign_colors(labels, common_clusters, base_color):
        return ["plum" if label in common_clusters else base_color for label in labels]

    # Assign colors for good and bad clusters
    colors_good = assign_colors(list(top_relations_good.index), common_relations, base_color='seagreen')
    colors_bad = assign_colors(list(top_relations_bad.index), common_relations, base_color='tomato')

    # Determine which colors are used for legend
    legend_colors_good = set(colors_good)
    legend_colors_bad = set(colors_bad)

    # Create patches for the legend (only for colors used in the plot)
    legend_patches_good = []
    if "seagreen" in legend_colors_good:
        legend_patches_good.append(mpatches.Patch(color="seagreen", label="Correct"))
    if "plum" in legend_colors_good:
        legend_patches_good.append(mpatches.Patch(color="plum", label="Common"))

    legend_patches_bad = []
    if "tomato" in legend_colors_bad:
        legend_patches_bad.append(mpatches.Patch(color="tomato", label="Incorrect"))
    if "plum" in legend_colors_bad:
        legend_patches_bad.append(mpatches.Patch(color="plum", label="Common"))

    # Plot for top 10 relations in top 2 classified entities
    plt.figure(figsize=(10, 6))
    plt.barh(top_relations_good.index[::-1], top_relations_good.values[::-1], color=colors_good[::-1], alpha=0.7)
    plt.xlabel("Frequency", fontsize=10)
    # plt.ylabel("Relation", fontsize=10)
    plt.yticks(fontsize=10)
    if legend_patches_good:
        plt.legend(handles=legend_patches_good, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, "relationships_good.png"))
    plt.close()

    # Plot for top 10 relations in entities outside top 100
    plt.figure(figsize=(10, 6))
    plt.barh(top_relations_bad.index[::-1], top_relations_bad.values[::-1], color=colors_bad[::-1], alpha=0.7)
    plt.xlabel("Frequency", fontsize=10)
    # plt.ylabel("Relation", fontsize=10)
    plt.yticks(fontsize=10)
    if legend_patches_bad:
        plt.legend(handles=legend_patches_bad, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, "relationships_bad.png"))
    plt.close()


def plot_top_types(entities_good_topn, entities_bad_topn, df_type_train, result_folder):

    # Filter df_type_train for good and bad classified entities
    df_good_topn = df_type_train[df_type_train['object'].isin(entities_good_topn)]
    df_bad_topn = df_type_train[df_type_train['object'].isin(entities_bad_topn)]

    # Count the top 10 most frequent types for both groups
    top_types_good = df_good_topn['type'].value_counts().head(10)
    top_types_bad = df_bad_topn['type'].value_counts().head(10)

    # Identify common types between the two sets
    common_types = set(top_types_good.index).intersection(set(top_types_bad.index))

    # Function to assign colors
    def assign_colors(labels, common_types, base_color):
        return ["plum" if label in common_types else base_color for label in labels]

    # Assign colors for good and bad clusters
    colors_good = assign_colors(list(top_types_good.index), common_types, base_color='seagreen')
    colors_bad = assign_colors(list(top_types_bad.index), common_types, base_color='tomato')

    # Determine which colors are used for legend
    legend_colors_good = set(colors_good)
    legend_colors_bad = set(colors_bad)

    # Create patches for the legend
    legend_patches_good = []
    if "seagreen" in legend_colors_good:
        legend_patches_good.append(mpatches.Patch(color="seagreen", label="Correct"))
    if "plum" in legend_colors_good:
        legend_patches_good.append(mpatches.Patch(color="plum", label="Common"))

    legend_patches_bad = []
    if "tomato" in legend_colors_bad:
        legend_patches_bad.append(mpatches.Patch(color="tomato", label="Incorrect"))
    if "plum" in legend_colors_bad:
        legend_patches_bad.append(mpatches.Patch(color="plum", label="Common"))


    # Plot for top 10 types in correctly classified entities
    plt.figure(figsize=(10, 6))
    plt.barh(top_types_good.index[::-1], top_types_good.values[::-1], color=colors_good[::-1], alpha=0.7)
    plt.xlabel("Frequency", fontsize=10)
    # plt.ylabel("Type", fontsize=10)
    plt.yticks(fontsize=10)
    if legend_patches_good:
        plt.legend(handles=legend_patches_good, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, "types_good_cl.png"))
    plt.close()

    # Plot for top 10 types in incorrectly classified entities
    plt.figure(figsize=(10, 6))
    plt.barh(top_types_bad.index[::-1], top_types_bad.values[::-1], color=colors_bad[::-1], alpha=0.7)
    plt.xlabel("Frequency", fontsize=10)
    # plt.ylabel("Type", fontsize=10)
    plt.yticks(fontsize=10)
    if legend_patches_bad:
        plt.legend(handles=legend_patches_bad, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, "types_bad_cl.png"))
    plt.close()

def frequent_clusters_plot(cluster_labels, clustered_entities, entities_good_topn, entities_bad_topn, top_n=5, result_folder=None):

    # Count occurrences of clusters for correctly and wrongly classified entities
    good_cluster_counts = {}
    bad_cluster_counts = {}

    for cluster, entities in clustered_entities.items():
        label = cluster_labels.get(cluster, "Unknown")
        cluster_size = len(entities)  # Total number of entities in this cluster

        if cluster_size > 0:
            # Count how many entities in this cluster belong to good or bad classifications
            good_entities_in_cluster = set(entities).intersection(entities_good_topn)
            bad_entities_in_cluster = set(entities).intersection(entities_bad_topn)

            # Normalize by cluster size
            good_count = len(good_entities_in_cluster) / cluster_size
            bad_count = len(bad_entities_in_cluster) / cluster_size

            if good_count > 0:
                good_cluster_counts[label] = good_count

            if bad_count > 0:
                bad_cluster_counts[label] = bad_count

    # Get the most frequent normalized clusters
    top_good_clusters = sorted(good_cluster_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_bad_clusters = sorted(bad_cluster_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Extract data for plotting
    good_labels, good_values = zip(*top_good_clusters) if top_good_clusters else ([], [])
    bad_labels, bad_values = zip(*top_bad_clusters) if top_bad_clusters else ([], [])

    # Find common clusters
    common_clusters = set(good_labels) & set(bad_labels)

    # Function to assign colors
    def assign_colors(labels, common_clusters, base_color):
        return ["plum" if label in common_clusters else base_color for label in labels]

    # Assign colors for good and bad clusters
    colors_good = assign_colors(good_labels, common_clusters, base_color='seagreen')
    colors_bad = assign_colors(bad_labels, common_clusters, base_color='tomato')

    # Determine which colors are used for legend
    legend_colors_good = set(colors_good)
    legend_colors_bad = set(colors_bad)

    # Create patches for the legend (only for colors used in the plot)
    legend_patches_good = []
    if "seagreen" in legend_colors_good:
        legend_patches_good.append(mpatches.Patch(color="seagreen", label="Correct Classification"))
    if "plum" in legend_colors_good:
        legend_patches_good.append(mpatches.Patch(color="plum", label="Common Cluster"))

    legend_patches_bad = []
    if "tomato" in legend_colors_bad:
        legend_patches_bad.append(mpatches.Patch(color="tomato", label="Incorrect Classification"))
    if "plum" in legend_colors_bad:
        legend_patches_bad.append(mpatches.Patch(color="plum", label="Common Cluster"))

    # Plot correctly classified entities' clusters (normalized)
    plt.figure(figsize=(10, 6))
    plt.barh(good_labels[::-1], good_values[::-1], color=colors_good[::-1], alpha=0.7)
    plt.xlabel("Normalized Frequency", fontsize=10)
    plt.ylabel("Cluster Label", fontsize=10)
    if legend_patches_good:
        plt.legend(handles=legend_patches_good, fontsize=10)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, "top_clusters_good_normalized.png"))
    plt.close()

    # Plot wrongly classified entities' clusters (normalized)
    plt.figure(figsize=(10, 6))
    plt.barh(bad_labels[::-1], bad_values[::-1], color=colors_bad[::-1], alpha=0.7)
    plt.xlabel("Normalized Frequency", fontsize=10)
    plt.ylabel("Cluster Label", fontsize=10)
    plt.yticks(fontsize=9)
    if legend_patches_bad:
        plt.legend(handles=legend_patches_bad, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, "top_clusters_bad_normalized.png"))
    plt.close()
