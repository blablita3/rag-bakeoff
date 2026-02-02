import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PALETTES = {
        "retrieval_method": {
            'baseline': '#aec7e8',
            'bm25': '#ffbb78',
            'dense': '#98df8a',
            'hybrid': '#ff9896'
        },
        "chunking_method": {
            'recursive': '#c5b0d5',
            'paragraph': "#CDCDCD"
        }
}

# Bar plots for comparing chunking methods
CHUNKING_PLOTS = ["answer_accuracy","source_accuracy"]

# Bar plots for comparing retrieval methods
RETRIEVAL_PLOTS = ["answer_accuracy", "source_accuracy", "coverage", "avg_latency"]

# Metrics to dispaly on tables
TABLE_METRICS = ["answer_accuracy", "source_accuracy", "coverage", "answer_std", "source_std", "avg_latency"]

# --- Helper Functions ---

def bar_plot(df, metric, x_col, saving_dir, filename, palette=None):
    """
    Builds a bar plot for a given metric (str) inside a df (DataFrame). Takes title (str) and
    a palette of colors (dict). Plot is saved to saving dir (path).
    """

    # Build directory if necessary
    saving_dir.mkdir(parents=True, exist_ok=True)

    # Build figure
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, hue=x_col, y=metric, palette=palette)

    # Set values
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel(metric.replace("_", " ").title())
    plt.tight_layout()

    # Save plot
    saving_path = saving_dir / filename
    plt.savefig(saving_path, bbox_inches="tight", dpi=300)
    print(f"Saved Bar Plot to {saving_path}")

    return None


def overview_table(main_df, table_metrics, saving_dir, filename, palettes=None):
    """
    Plots a table containing the metrics for a single chunking method. 
    Rows = retrieval methods, Columns = table_metrics (list(str))
    Metrics are found in main_df (DataFrame).
    Palettes (dict[dict[str]]) specifies the row colors.
    Table saved to figures directory (path).
    """

    # Build directory if necessary
    saving_dir.mkdir(parents=True, exist_ok=True)

    # Fill empty values
    df = main_df.fillna("-")

    # Prepare table values: retrieval_method as first column, then metrics
    table_values = df[["retrieval_method"] + table_metrics].values.tolist()

    # Create row colors based on retrieval_method
    row_colors = []
    for _, row in df.iterrows():
        rcolor = palettes.get("retrieval_method", {}).get(row["retrieval_method"], "#ffffff")
        colors = [rcolor] * (len(table_metrics) + 1)
        row_colors.append(colors)

    # Adjust figure height based on number of rows
    fig_height = len(df)
    fig, ax = plt.subplots(figsize=(12, fig_height), dpi=300)
    ax.axis("off")

    # Create table
    table = ax.table(cellText=table_values,
                     colLabels=["Retrieval Method"] + table_metrics,
                     cellLoc="center",
                     loc="center",
                     cellColours=row_colors)
                     # CHANGED: removed fixed colWidths

    # Tight cells width
    table.auto_set_column_width(col=list(range(len(table_metrics)+1)))

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(20)

    # Scale cells height
    table.scale(2, 3)

    plt.tight_layout()

    # Save figure
    saving_path = saving_dir / filename
    fig.savefig(saving_path, bbox_inches="tight", dpi=300, pad_inches=0)
    print(f"Saved Overview Table to {saving_path}")

    return None


# --- Main Figures Builder ---

def build_all_figures(metrics, main_chunking_method, saving_dir, table_metrics=TABLE_METRICS, chunking_plots=CHUNKING_PLOTS, retrieval_plots=RETRIEVAL_PLOTS, palettes=PALETTES):
    """
    Builds all figures, based on metrics (list(dict)).
    Retrieval methods plots focus a the main chunking method (str).
    Table metrics (list(str)) is the list of metrics the overview table will display.
    Chunking plots (list(tuple)) contains the (metric, title) pair to plot for comparing chunking methods.
    Retrieval plots (list(tuple)) contains the (metric, title, include_baseline) to plot for comparing retrieval methods.
    Palettes (dict(dict(str))) bring color.
    Results are saved to figures_dir path.
    """

    print("Building figures ... \n")
    df = pd.DataFrame(metrics)


    # For all chunking methods
    for method in df["chunking_method"].unique():

        # Chunking method df
        method_df = df[df["chunking_method"] == method].round(2)
        
        # Build overview table with all retrieval metrics
        filename = f"{method}_overview_table.png"
        overview_table(method_df, table_metrics, saving_dir, filename, palettes)


    # Group by chunking method, excluding the baseline metric (noise)
    chunking_df = df[df["retrieval_method"] != "baseline"]
    chunking_df = chunking_df.groupby("chunking_method")[["answer_accuracy", "source_accuracy"]].mean().reset_index().round(2)

    # Plots to compare chunking methods
    for metric in chunking_plots:

        filename = f"chunking_{metric}.png"

        bar_plot(
            chunking_df,
            metric=metric,
            x_col="chunking_method",
            saving_dir=saving_dir,
            filename=filename,
            palette=palettes.get("chunking_method", None),
        )


    # Group by baseline method
    main_df = df.groupby(["retrieval_method"])[table_metrics].mean().reset_index().round(2)

    # Overview retrieval metrics
    overview_table(main_df, table_metrics, saving_dir, "retrieval_overview_table", palettes)

    # Plots to comapre retrieval methods
    for metric in retrieval_plots:

        filename = f"{metric}.png"

        bar_plot(
            main_df,
            metric=metric,
            x_col="retrieval_method",
            saving_dir=saving_dir,
            filename=filename,
            palette=palettes.get("retrieval_method", None),
        )

    print(f"\nAll figures built. See them at {saving_dir} directory\n")

    return None