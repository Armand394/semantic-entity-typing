import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Apply global settings
plt.rcParams.update({
    'axes.spines.right': False,   # Enable right spine (solid)
    'axes.spines.top': False,     # Enable top spine (solid)
    'axes.grid': True,           # Enable grid
    'grid.alpha': 0.4,           # Make the grid transparent (adjust alpha)
    'xtick.direction': 'out',     # Tickmarks on x-axis (inside)
    'ytick.direction': 'out',     # Tickmarks on y-axis (inside)
    'grid.linestyle': '--',      # Dashed grid (can be changed)
    'axes.edgecolor': 'black',   # Ensure spines are visible
    'axes.linewidth': 1.2,        # Make spines slightly thicker
    'axes.labelsize': 11
})

metric_latex_map = {
    "kg_sim_mu": r"$\mathrm{R}_{\mu}^{\cos}$",    # Relationship sentence coherence
    "et_sim_mu": r"$\mathrm{T}_{\mu}^{\cos}$",    # Type sentence coherence
    "degree":     r"$d$",                         # Total entity degree
    "avg_txt_length": r"$L_{\mathrm{e}}$",        # Avg. length of entity descriptions
    "kg_degree":  r"$d_{\mathrm{r}}$",            # Relationship graph degree
    "et_degree":  r"$d_{\mathrm{t}}$",            # Type graph degree
    "type_kg_ratio": r"$\rho_{t:r}$"              # Type-to-relationship ratio
}

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


def plot_similarity_metrics_pairplots(df, figure_folder, figure_name):
    # Define columns to plot against 'rank_value'
    x_cols = ['avg_txt_length', 'kg_degree', 'et_degree']
    y_col = ['kg_sim_mu', 'et_sim_mu']

    x_cols = ['avg_txt_length', 'kg_degree', 'et_degree']
    y_cols = ['kg_sim_mu', 'et_sim_mu']

    # Set up a grid of plots (rows = y metrics, cols = x metrics)
    nrows, ncols = len(y_cols), len(x_cols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)

    # Ensure axes is always a 2D array
    if nrows == 1:
        axes = [axes]
    if ncols == 1:
        axes = [[ax] for ax in axes]

    for i, y in enumerate(y_cols):
        for j, x in enumerate(x_cols):
            ax = axes[i][j]

            # Apply log transform for degree-based x axes
            if x in ['kg_degree', 'et_degree']:
                sns.scatterplot(x=np.log1p(df[x]), y=df[y], ax=ax)
                if i == 1:
                    ax.set_xlabel(f"log({metric_latex_map[x]})", fontsize=14)
                else:
                    ax.set_xlabel(" ")
            else:
                sns.scatterplot(x=df[x], y=df[y], ax=ax)
                if i == 1:
                    ax.set_xlabel(metric_latex_map[x], fontsize=14)
                else:
                    ax.set_xlabel(" ")

            # Set y-axis label only on first column
            if j == 0:
                ax.set_ylabel(metric_latex_map[y], fontsize=14)
            else:
                ax.set_ylabel("")

            # No title
            ax.set_title("")

    # Save the plot
    os.makedirs(figure_folder, exist_ok=True)
    plot_path = os.path.join(figure_folder, f"{figure_name}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_rank_value_pairplots(df, figure_folder, figure_name):
    # Define columns to plot against 'rank_value'
    x_cols = ['kg_sim_mu', 'et_sim_mu']
    y_col = 'rank_value'

    # Set up the plot
    num_cols = len(x_cols)
    fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(5 * num_cols, 5), constrained_layout=True)
    
    axes[0].set_ylabel('Rank')

    # Plot each subplot
    for i, col in enumerate(x_cols):
        ax = axes[i]
        if col in ['kg_degree', 'et_degree']:
            sns.scatterplot(x=np.log1p(df[col]), y=df[y_col], ax=ax)
            ax.set_xlabel(f"log({metric_latex_map[col]})", fontsize=14)
        else:
            sns.scatterplot(x=df[col], y=df[y_col], ax=ax)
            ax.set_xlabel(metric_latex_map[col], fontsize=14)

        # Only label y-axis on the first plot
        if i == 0:
            ax.set_ylabel('Rank prediction', fontsize=14)
        else:
            ax.set_ylabel('')

    # Save the plot
    plot_path = os.path.join(figure_folder, f"{figure_name}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()


def box_plot_metrics(metrics_df, columns, fig_name, figure_folder):

    figure = os.path.join(figure_folder, f"boxplot_{fig_name}.png")

    # Apply Min-Max scaling to the selected columns
    scaler = MinMaxScaler()
    metrics_df_scaled = metrics_df.copy()
    metrics_df_scaled[columns] = scaler.fit_transform(metrics_df_scaled[columns])

    # Melt the DataFrame to long format for seaborn
    df_melted = metrics_df_scaled.melt(
        id_vars='y',
        value_vars=columns,
        var_name='Metric',
        value_name='Value'
    )

    # Replace metric names with LaTeX labels
    df_melted['Metric'] = df_melted['Metric'].map(metric_latex_map).fillna(df_melted['Metric'])

    # Define custom colors for the classes
    custom_palette = {0: "firebrick", 1: "limegreen"}

    # Create the boxplot with custom colors
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Metric', y='Value', hue='y', data=df_melted, palette=custom_palette)

    # Improve plot aesthetics
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.legend(title="Class (y)", loc='upper right')
    plt.tight_layout()
    plt.savefig(figure)
    plt.close()


def rank_by_metric_barplot(entity_metrics, figure_folder):

    entity_metrics = entity_metrics.copy()

    # Bin variables to be tuned for plotting
    entity_metrics.loc[:,'et_degree_bin'] = pd.qcut(entity_metrics['et_degree'], q=10, duplicates='drop')
    entity_metrics.loc[:,'kg_degree_bin'] = pd.qcut(entity_metrics['kg_degree'], q=10, duplicates='drop')
    entity_metrics.loc[:,'avg_txt_length_bin'] = pd.qcut(entity_metrics['avg_txt_length'], q=10, duplicates='drop')

    # Average rank by quantile bin
    binned_types = entity_metrics.groupby('et_degree_bin', observed=True)['rank_value'].mean()
    binned_kg = entity_metrics.groupby('kg_degree_bin', observed=True)['rank_value'].mean()
    binned_length = entity_metrics.groupby('avg_txt_length_bin', observed=True)['rank_value'].mean()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # First plot
    axes[1].bar(range(len(binned_types)), binned_types.values, color='salmon')
    axes[1].set_xticks(range(len(binned_types)))
    axes[1].set_xticklabels(binned_types.index.astype(str), rotation=45, ha='right')
    axes[1].set_xlabel(metric_latex_map['et_degree'], fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)

    # Second plot
    axes[0].bar(range(len(binned_kg)), binned_kg.values, color='skyblue')
    axes[0].set_xticks(range(len(binned_kg)))
    axes[0].set_xticklabels(binned_kg.index.astype(str), rotation=45, ha='right')
    axes[0].set_ylabel('Average Rank', fontsize=14)
    axes[0].set_xlabel(metric_latex_map['kg_degree'], fontsize=14)

    # Third plot
    axes[2].bar(range(len(binned_length)), binned_length.values, color='mediumseagreen')
    axes[2].set_xticks(range(len(binned_length)))
    axes[2].set_xticklabels(binned_length.index.astype(str), rotation=45, ha='right')
    axes[2].set_xlabel(metric_latex_map['avg_txt_length'], fontsize=14)

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, "metrics_rank_distributions.png"))
    plt.close()