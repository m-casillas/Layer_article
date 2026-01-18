from scipy.stats import wilcoxon, mannwhitneyu
import seaborn as sns
from globalsENAS import *
from scipy.stats import pearsonr
from matplotlib.colors import ListedColormap
import os

def rank1_stats(rank1_folder = ''):
    columns = ["Crossover_type", "Mutation_type", "Accuracy", "FLOPs", "Num_Params"]
    dfs = []
    for file in os.listdir(rank1_folder):
        if file.endswith(".csv"):
            path = os.path.join(rank1_folder, file)
            df = pd.read_csv(path)
            df = df[columns]
            df["Operators"] = df["Crossover_type"] + "_" + df["Mutation_type"]
            dfs.append(df)

    df_total = pd.concat(dfs, ignore_index=True)
    ref_row = df_total[df_total["Crossover_type"] == "HHSE"]
    ref_acc = ref_row["Accuracy"].iloc[0]
    ref_flops = ref_row["FLOPs"].iloc[0]
    ref_params = ref_row["Num_Params"].iloc[0]

    df_total["Accuracy_diff_pct_vs_HHSE_HHSE"] = (df_total["Accuracy"] - ref_acc) / ref_acc*100
    df_total["FLOPs_diff_pct_vs_HHSE_HHSE"] = (df_total["FLOPs"] - ref_flops) / ref_flops*100
    df_total["Num_Params_diff_pct_vs_HHSE_HHSE"] = (df_total["Num_Params"] - ref_params) / ref_params*100
    df_total.to_csv(os.path.join(rank1_folder,'rank1_results.csv'), index=False)

    metrics = ["Accuracy", "FLOPs", "Num_Params"]
    for metric in metrics:
        df_sorted = df_total.sort_values(by=metric, ascending=False)
        plt.figure(figsize=(12, 6))
        plt.bar(df_sorted["Operators"], df_sorted[metric])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)
        plt.xlabel("Crossover_Mutation")
        plt.title(f"{metric} by Crossover-Mutation")
        plt.tight_layout()
        plt.savefig(os.path.join(rank1_folder, f"{metric}.png"))
        plt.close()

    pct_metrics = {
        "Accuracy_diff_pct_vs_HHSE_HHSE": "Accuracy % Diff vs HHSE_HHSE",
        "FLOPs_diff_pct_vs_HHSE_HHSE": "FLOPs % Diff vs HHSE_HHSE",
        "Num_Params_diff_pct_vs_HHSE_HHSE": "Num Params % Diff vs HHSE_HHSE"
    }

    for col, label in pct_metrics.items():
        df_sorted = df_total.sort_values(by=col, ascending=False)
        plt.figure(figsize=(12, 6))
        plt.bar(df_sorted["Operators"], df_sorted[col])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(label)
        plt.xlabel("Crossover_Mutation")
        plt.title(label)
        plt.tight_layout()
        plt.savefig(os.path.join(rank1_folder, f"{col}.png"))
        plt.close()

    return df_total

def build_go_matrix(report_folder, input_csv):
    #Matrix for Wilcoxon results
    # Load the pairwise test results
    df = pd.read_csv(input_csv)
    # Standardize labels
    df['H0 veredict'] = df['H0 veredict'].replace({
        'Rejected': 'Rejected',
        'Non-rejected': 'Not Rejected'
    })
    # Identify all unique GO conditions
    gos = sorted(set(df['GO1']).union(set(df['GO2'])))
    # Create matrix with text labels
    text_matrix = pd.DataFrame("—", index=gos, columns=gos)
    for g in gos:
        text_matrix.loc[g, g] = "Same"
    for _, row in df.iterrows():
        g1, g2 = row['GO1'], row['GO2']
        verdict = row['H0 veredict']
        text_matrix.loc[g1, g2] = verdict
        text_matrix.loc[g2, g1] = verdict
    # Save CSV
    matrixCSV_path = os.path.join(report_folder, f'GO_matrix.csv')
    text_matrix.to_csv(matrixCSV_path, index=True)
    print("CSV saved as GO_matrix.csv")
    # Build numeric matrix for PNG (0=Rejected red, 1=Not rejected green)
    numeric_matrix = pd.DataFrame(np.nan, index=gos, columns=gos)
    for g in gos:
        numeric_matrix.loc[g, g] = np.nan  # diagonal blank
    for _, row in df.iterrows():
        g1, g2 = row['GO1'], row['GO2']
        verdict = row['H0 veredict']
        numeric_matrix.loc[g1, g2] = 1 if verdict == "Not Rejected" else 0
        numeric_matrix.loc[g2, g1] = 1 if verdict == "Not Rejected" else 0
    return gos, numeric_matrix


def plot_go_matrix(gos, numeric_matrix, report_folder):
    cmap = ListedColormap(["red", "green"])
    plt.figure(figsize=(10, 8))
    plt.imshow(numeric_matrix, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.xticks(range(len(gos)), gos, rotation=90)
    plt.yticks(range(len(gos)), gos)
    plt.colorbar()
    plt.tight_layout()
    matrixPNG_path = os.path.join(report_folder, f'go_matrix.png')
    plt.savefig(matrixPNG_path, dpi=300)
    print("PNG saved as go_matrix.png")

def Wilcoxon_ENAS(experiment_folder):
    #Uses the Wilcoxon test for each pair of combinations of genetic operators. It compares the accuracy of the best architecture from the last generation.

    splitted_folder = os.path.join(experiment_folder, 'splitted')
    files = os.listdir(splitted_folder)
    pairs_of_files = list(combinations(files,2))
    report_columns = ['GO1', 'GO2', 'Wilcoxon Statistic', 'p-value', 'H0 veredict']
    df_report = pd.DataFrame(columns=report_columns)
    report_folder = os.path.join(experiment_folder, 'stats','Wilcoxon')
    ensure_folder_exists(report_folder)
    for file1, file2 in pairs_of_files:
        print(f"Wilcoxon test statistic: {file1} vs {file2}")
        file1_path = os.path.join(splitted_folder, file1)
        file2_path = os.path.join(splitted_folder, file2)
        # Read the CSV files
        df1 = pd.read_csv(file1_path, encoding='utf-8')
        df2 = pd.read_csv(file2_path, encoding='utf-8')

        #Get the best architecture from the last generation
        df1Best = df1[ (df1['arch_status'] == 'BEST') & (df1['Generation'] == df1['Generation'].max()) ]
        df2Best = df2[(df2['arch_status'] == 'BEST') & (df2['Generation'] == df2['Generation'].max()) ]

        data1 = df1Best['Accuracy']
        data2 = df2Best['Accuracy']

        #stat, p_value = wilcoxon(data1, data2)
        stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')

        filename1 = file1[:-4]
        filename2 = file2[:-4]
        
        if p_value < 0.05:
            result = "Rejected"
        else:
            result = "Non-rejected"
        df_report.loc[len(df_report)] = [filename1, filename2, stat, p_value, result]
        #df_report.to_csv(os.path.join(report_folder, f'{file1}_{file2}.csv'), index=False)
        print(f"Wilcoxon test statistic: {file1} vs {file2} complete\n")
    reportfile_path = os.path.join(report_folder, f'Wilcoxon_report.csv')
    df_report.to_csv(reportfile_path, index=False)
    gos, numeric_matrix = build_go_matrix(report_folder, reportfile_path)
    plot_go_matrix(gos, numeric_matrix, report_folder)

def correlation_matrix_folder(experiment_folder):
    splitted_folder = os.path.join(experiment_folder, 'splitted')
    ensure_folder_exists(splitted_folder)

    correlation_plot_folder = os.path.join(experiment_folder, 'stats', 'correlation')
    ensure_folder_exists(correlation_plot_folder)

    figs = []
    for filename in os.listdir(splitted_folder):
        file = os.path.join(splitted_folder, filename)
        print(f"Processing {file}...")
        df = pd.read_csv(file)

        dfBestGen = df[df['arch_status'] == 'BEST']
        if 'NONE_NONE' in filename:
            selected_cols = dfBestGen[config_tecnas.plot_archcolumns + config_tecnas.plot_GA_RANDOM]
        elif 'NONE_' in filename:
            selected_cols = dfBestGen[config_tecnas.plot_archcolumns + config_tecnas.plot_GA_NONECROSS]
        elif '_NONE' in filename:
            selected_cols = dfBestGen[config_tecnas.plot_archcolumns + config_tecnas.plot_GA_NONEMUT]
        else:
            selected_cols = dfBestGen[config_tecnas.plot_archcolumns + config_tecnas.plot_GAcolumns]

        corr_matrix = selected_cols.corr()
        np.fill_diagonal(corr_matrix.values, np.nan)

        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title(f"Correlation Matrix\n{filename[:-4]}")
        plt.tight_layout()
        out_path = os.path.join(correlation_plot_folder, f'{filename[:-4]}_correlation_matrix.png')
        plt.savefig(out_path, dpi=200)
        figs.append(fig)
        plt.close(fig)

    combined_path = os.path.join(correlation_plot_folder, 'all_correlation_matrices.png')
    n = len(figs)
    rows, cols = 4, 3
    final_fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

    axes = axes.flatten()
    for ax in axes[n:]:
        ax.axis('off')

    for i, fig in enumerate(figs):
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = plt.imread(buf)
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(combined_path, dpi=200)
    plt.close(final_fig)




def pairwise_scatterplots_folder(experiment_folder):
    splitted_folder = os.path.join(experiment_folder, 'splitted')
    ensure_folder_exists(splitted_folder)

    scatterplot_folder = os.path.join(experiment_folder, 'stats', 'scatterplots')
    ensure_folder_exists(scatterplot_folder)

    for filename in os.listdir(splitted_folder):
        file = os.path.join(splitted_folder, filename)
        print(f"Processing {file}...")
        df = pd.read_csv(file)
        dfBestGen = df[df['arch_status'] == 'BEST']

        selected_cols = ['Accuracy', 'FLOPs', 'Num_Params', 'HD_P1', 'HD_P2', 'HD_BM', 'Penalization']
        df_selected = dfBestGen[selected_cols]

        pairs = list(itertools.combinations(selected_cols, 2))

        for x, y in pairs:
            plt.figure(figsize=(5, 4))
            sns.scatterplot(data=df_selected, x=x, y=y)

            # Compute Pearson correlation
            correlation, _ = pearsonr(df_selected[x], df_selected[y])
            corr_str = f"{correlation:.4f}"

            # Update title with correlation
            plt.title(f'{y} vs {x} (r = {corr_str})')
            plt.tight_layout()

            # Build filename with correlation at the beginning
            plot_filename = f"{corr_str}_{filename[:-4]}_{x}_vs_{y}.png".replace(" ", "_")
            output_path = os.path.join(scatterplot_folder, plot_filename)

            print(f'Saving {output_path}')
            plt.savefig(output_path, dpi=200)
            plt.clf()
            plt.close()

        print("Done\n")

os.system("cls")
#path = r"C:\Users\xaero\OneDrive\ITESM DCC\Layer_article\results\experiment5"
#pairwise_scatterplots_folder(path)
#filename = 'archs_2025-05-13_00-14.csv'
#filepath = os.path.join(path, filename)
#correlation_matrix(filepath)
#Wilcoxon_ENAS(path)
#pairwise_scatterplots_folder(path)