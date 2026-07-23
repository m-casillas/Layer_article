# Refactored Plotter Class
# Generated from ChatGPT refactor request
import ast
import math
import os
import random
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from globalsENAS import *
from TECNAS import TECNAS

class Plotter:

    DEFAULT_FIGSIZE = (14, 8)
    DEFAULT_DPI = 300
    CROSSOVER_COLORS = {
        'SPC': '#1f77b4',  # blue
        'TPC': '#d62728',  # red
        'UC': '#2ca02c',   # green
        'SBX': '#9467bd',  # purple
        'NONE': '#ff7f0e',  # orange
        'HHSE': "#793939"   # cyan
    }
    MUTATION_COLORS = {
    'MPOLY': '#1f77b4',   # blue
    'MNUF': '#d62728',    # red
    'NONE': '#2ca02c',    # green
    'MSWAP': '#9467bd',   # purple
    'MPAR': '#ff7f0e',    # orange
    'MBFLIP': '#17becf',   # cyan
    'RANDOM': '#000000',
    'AOS': '#8c564b',
    'HAOS': '#ffd700'
    }

    WINDOW_COLORS = {
    'MPOLY': '#e377c2',        # pink
    'MNUF': '#bcbd22',         # olive
    'NONE': '#8c564b',         # brown
    'MSWAP': '#7f7f7f',        # gray
    'MPAR': '#393b79',         # indigo
    'MBFLIP': '#637939',       # dark olive green
    'RANDOML_NONE': '#525252' # dark gray
    }

    WINDOW_COLORS = {
    1:  '#1f77b4',  # cyan
    5:  '#ff7f0e',  # orange
    10: '#2ca02c',  # green
    20: '#d62728',  # red
    40: '#9467bd',  # purple
    50: '#8c564b',  # brown
    60: '#e377c2',  # pink
    80: '#17becf'   # blue
    }



    WINDOW_MARKERS =  {
    1: 'o',    # circle
    5: 's',    # square
    10: '^',   # triangle up
    20: 'D',   # diamond
    40: 'P',   # filled plus
    50: 'X',   # filled x
    60: '*',   # star
    80: 'p',   # pentagon
    }

    CROSSOVER_MARKERS = {'SPC': 'o', 'TPC': 's', 'UC': '^', 'SBX': 'D', 'NONE': '>', 'HHSE': '<'}
    CROSSOVER_LINESTYLES = {'SPC': '-',  'TPC': '--', 'UC': '-.', 'SBX': ':',  'NONE': (0, (3, 1, 1, 1))}
    NSGA2_LINES = [5, 25, 50, 100, 200, 300, 400]

    def __init__(self, experiment_folder):
        os.system('cls')
        self.experiment_folder = experiment_folder
        self.plot_folder = os.path.join(experiment_folder, 'plots')
        self.splitted_folder = os.path.join(experiment_folder, 'splitted')
        self.boxplots = os.path.join(self.plot_folder, 'boxplots')
        self.bestarchs_summaries_folder = os.path.join(experiment_folder, 'bestarchs_summaries')
        self.GAs_summaries_folder = os.path.join(experiment_folder, 'GAs_summaries')
        self.columns_arch = config_tecnas.plot_archcolumns
        self.columns_GA = config_tecnas.plot_GAcolumns
        #self._create_folders(self.experiment_folder, self.plot_folder,  self.splitted_folder,  self.boxplots,  self.bestarchs_summaries_folder, self.GAs_summaries_folder)
        sns.set_theme(style='whitegrid', context='talk', palette='tab10')

    @staticmethod
    def get_window_color(hex_color = '', Wnum = 0, min_W=0, max_W=100):
        from matplotlib.colors import to_rgb, to_hex
        base_rgb = np.array(to_rgb(hex_color))
        t = (Wnum - min_W) / (max_W - min_W)
        t = np.clip(t, 0, 1)
        # factor ranges from 1.3 (lighter) to 0.7 (darker)
        factor = 1.3 - 0.7 * t
        rgb = np.clip(base_rgb * factor, 0, 1)
        return to_hex(rgb)

    @staticmethod
    def _create_folders(*folders):
        for folder in folders:
            ensure_folder_exists(folder)

    @staticmethod
    def _load_csv(filepath):
        return pd.read_csv(filepath, encoding='utf-8')

    @staticmethod
    def _save_plot(plot_path, dpi=DEFAULT_DPI):
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close()

    @staticmethod
    def _compute_tendency(values, window_ratio=0.1):
        window = max(3, int(window_ratio * len(values)))
        return (pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy())

    @staticmethod
    def _compute_auc(values, generations):
        return np.trapz(values, generations)

    @staticmethod
    def _find_best_point(values, generations, mode='max'):
        idx = np.nanargmax(values) if mode == 'max' else np.nanargmin(values)
        return generations[idx], values[idx], idx

    @staticmethod
    def _format_large_number(value):
        if value > 1e4:
            sci = f'{value:.3e}'
            mantissa, exp = sci.split('e')
            return f'{mantissa}e{int(exp)}'
        return f'{value:.4f}'

    @staticmethod
    def _get_ticks(generations, step=20):
        start = int(min(generations))
        end = int(max(generations))

        return list(np.arange(start, end + 1, step))
 
    
    def _load_generation_status_data(self):
        all_data = {}
        for filename, filepath in self._iter_csv_files(self.splitted_folder):
            if 'generation_status' not in filename:
                continue
            print(f'Loading {filepath}')
            df_raw = self._load_csv(filepath)
            df_grouped = (df_raw.groupby('Generation').mean(numeric_only=True).reset_index())
            stem = filename[:-4]
            all_data[stem] = {'df': df_grouped, 'df0': df_raw,'generations': df_grouped['Generation'].values, 'search_name': df_raw['search_name'].iloc[0]}
        return all_data

    def _rank_operators_by_auc(self, all_data, column, mode):
        ranked = []
        for stem, data in all_data.items():
                values = data['df'][column].values
                auc = self._compute_auc(values, data['generations'])
                ranked.append((auc, stem, data))
        ranked.sort(key=lambda item: item[0], reverse=(mode == 'max'))
        return ranked
    
    def _plot_top_k_operators(self, ax, top_k,column, mode, plot_raw_data=False, nsga2_window=False, slice_generations=False, use_random_colors=False):
        #Load all search names to check if there is any window method and the number: W20
        W_values = [] #Save them for plotting vertical lines.
        W = ''
        starts_at = 0 #Starting generation for plotting.
        W_type = None
        for _, _, data in top_k:
            parts = data['search_name'].split('_')
            for part in parts:
                if part.startswith('W'):
                        value = int(part[1:])
                        W_values.append(value*config_tecnas.GENERATIONS/100.0) #Convert to generation number for vertical line plotting.
                        if W_type is None or value > W_type:
                            W_type = value
                            
        for rank, (auc_val, _, data) in enumerate(top_k):
            parts = data['search_name'].split('_')
            crossover = parts[0]
            mutation  = parts[1]
            W_current = parts[2] if len(parts) > 2 else '' #Use it for coloring Window methods. 
            if ((mutation == 'NONE' and 'mut' in column) or (crossover == 'NONE' and 'cross' in column)):
                continue
            style = self._get_visual_style(crossover, mutation, int(W_current[1:]) if W_current != '' else -1)
            #print(W_current, style)
            values = data['df'][column].values
            generations = data['generations']
            if slice_generations:
                starts_at = int(W_type)/100.0
                start_idx = int( len(generations) * starts_at )
                generations = generations[start_idx:]
                values = values[start_idx:]
            tendency = self._compute_tendency(values)
            y = values if plot_raw_data else tendency
            best_gen, best_val, _ = self._find_best_point(y,generations, mode )
            color = random.choice(plt.rcParams['axes.prop_cycle'].by_key()['color']) \
            if use_random_colors else style['color']
  
            sns.lineplot(
                x=generations,
                y=y,
                ax=ax,
                color=color,
                linestyle=style['linestyle'],
                linewidth=1.8,
                alpha=0.9,
                marker=style['marker'],
                markersize=5,
                markeredgecolor='black',
                markeredgewidth=0.5,
                markevery=max(1, len(generations) // 12),
                label=(f'[#{rank + 1}] ' f'AUC={auc_val:.2e} | ' f'{data["search_name"]}' )
            )
            ax.scatter( best_gen, best_val, color=style['color'], s=120, marker='X', edgecolors='black',linewidths=0.5,zorder=5 )
            if nsga2_window:
                if slice_generations:
                    ax.axvline(x=W_type*config_tecnas.GENERATIONS/100.0, color='red', linestyle=':',linewidth=0.7, alpha=0.9)
                else:
                    for line in W_values:
                        ax.axvline(x=line, color='red', linestyle=':',linewidth=0.7, alpha=0.9)
                
            
                    
        return starts_at

    def _iter_csv_files(self, folder):
        for filename in os.listdir(folder):
            if filename.endswith('.csv'):
                yield filename, os.path.join(folder, filename)

    def _apply_axis_style(self, ax, xlabel='Generation', ylabel='', ticks=None):
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        if ticks is not None:
            ax.set_xticks(ticks)
        ax.tick_params(axis='x', labelsize=7, rotation=90)
        ax.tick_params(axis='y', labelsize=7)
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.grid(True, linestyle='--', alpha=0.3)

    def _get_visual_style(self, crossover, mutation, W_num = -1):
        
        if crossover == 'NONE' and mutation == 'NONE':
            return {'color': 'black',  'marker': 'X', 'linestyle': '-'}
      
        if W_num != -1:
            #Use this for comparing same windows, same mutation, different crossovers.
            return {'color': self.CROSSOVER_COLORS.get(crossover, 'black'), 'marker': self.WINDOW_MARKERS .get(W_num, 'o'), 'linestyle': self.CROSSOVER_LINESTYLES.get(crossover, '-')}
            #return {'color': Plotter.get_window_color(hex_color = self.WINDOW_COLORS.get(W_num, 'black'), Wnum = W_num), 'marker': self.WINDOW_MARKERS .get(W_num, 'o'), 'linestyle': self.CROSSOVER_LINESTYLES.get(crossover, '-')}
        else:
            return {'color': self.MUTATION_COLORS.get(mutation, 'black'),'marker': self.CROSSOVER_MARKERS.get(crossover, 'o'), 'linestyle': self.CROSSOVER_LINESTYLES.get(crossover, '-')}
        
        

    
    def get_median_oneCombination(self, df, cross, mut):
        print(f'Processing median: {cross} - {mut}')
        last_generation = df['Generation'].max()
        df_filtered = df[ (df['Crossover_type'] == cross) & (df['Mutation_type'] == mut) & (df['arch_status'] == 'BEST')]
        df_last_gen = df_filtered[ df_filtered['Generation'] == last_generation ]
        df_sorted = df_last_gen.sort_values( by='Accuracy').reset_index(drop=True)
        median_row = df_sorted.iloc[len(df_sorted) // 2]
        median_exec = int(median_row['Execution'])
        median_trajectory = ( df_filtered[ df_filtered['Execution'] == median_exec]['Accuracy'].tolist())
        return {f'{cross}-{mut}': median_trajectory}

    

    def plot_measures_from_folder(self, plot_bestarchs=True):
        """
        Reads summarized CSV files and creates seaborn barplots
        for comparing crossover-mutation strategies.
        """
        self.barplot_folder = os.path.join(self.plot_folder, 'summarized_measures')
        if plot_bestarchs:
            self.to_be_plotted_folder = self.bestarchs_summaries_folder
            columns = self.columns_arch
        else:
            self.to_be_plotted_folder = self.GAs_summaries_folder
            columns = self.columns_GA
        ensure_folder_exists(self.barplot_folder)
        ensure_folder_exists(self.to_be_plotted_folder)
        for column in columns:
            plot_data = []
            for filename, filepath in self._iter_csv_files(self.to_be_plotted_folder):
                print(f'Plotting measures: Processing {filename}')
                df = self._load_csv(filepath)
                df_mean = df[df['Execution'] == 'Mean']
                value = df_mean[column].astype(float).iloc[0]
                strategy = filename.replace('_bestarch_summary.csv', '') if plot_bestarchs else filename.replace('_GA_summary.csv', '')
                strategy = strategy.replace('_generation_status', '')
                plot_data.append({'strategy': strategy, 'value': value, 'group': 'HHSE' if 'HHSE' in strategy else 'Normal'   })
                print(f'Processing {filename} done')
            plot_df = pd.DataFrame(plot_data)
            plot_df = plot_df.sort_values(by='value', ascending=False)
            plt.figure(figsize=(12, 6))
            plot_df['mutation'] = plot_df['strategy'].apply(lambda s: 'RANDOM_NONE' if s == 'NONE_NONE' else s.split('_')[1]) #Get mutation for color bar.
            palette = {strategy: Plotter.MUTATION_COLORS.get(mutation, 'gray')  for strategy, mutation in zip(plot_df['strategy'], plot_df['mutation'])}
            ax = sns.barplot(data=plot_df, x='strategy', y='value', dodge=False, palette=palette)
            plt.xlabel('Search strategy', fontsize=6)
            plt.ylabel(column, fontsize=6)
            plt.title(f'{config_tecnas.plot_convergency_generation_status_title_names[column]}\nfor all {config_tecnas.EXECUTIONS} executions in the last generation per Strategy', fontsize=10)
            plt.xticks(rotation=90, ha='right', fontsize=6)
            plt.yticks(fontsize=6)
            min_val = plot_df['value'].min()
            max_val = plot_df['value'].max()
            margin = (max_val - min_val) * 0.05 if max_val != min_val else 0.1
            if np.isfinite(min_val) and np.isfinite(max_val):
                plt.ylim(min_val - margin, max_val + margin)
            for patch in ax.patches:
                height = patch.get_height()
                if np.isnan(height):
                    continue
                if height > 1e4:
                    sci = f'{height:.2e}'
                    mantissa, exp = sci.split('e')
                    label = f'{mantissa}e{int(exp)}'
                else:
                    label = f'{height:.4f}'
                ax.text(patch.get_x() + patch.get_width() / 2, height + margin * 0.15, label, ha='center', va='bottom', fontsize=5, rotation=90)
            offset = ax.yaxis.get_offset_text()
            offset.set_fontsize(5)
            plt.tight_layout()
            plot_path = os.path.join(self.barplot_folder, f'{column}.png')
            print(f'Saving {plot_path}')
            self._save_plot(plot_path, dpi=200)
            print('Done\n')
    def boxplot_from_folder(self):
        columns = config_tecnas.plot_convergency_generation_status_columns
        for column in columns:
            plot_data = []
            for filename, filepath in self._iter_csv_files(self.splitted_folder):
                print(f'Processing file: {filepath}')
                df = self._load_csv(filepath)
                go_str = filename[:-4].replace('_generation_status', '')
                temp_df = pd.DataFrame({'value': df[column], 'strategy': go_str  })
                plot_data.append(temp_df)
            plot_df = pd.concat(plot_data, ignore_index=True)
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=plot_df, x='strategy', y='value', linewidth=1.2, fliersize=2)
            #sns.stripplot(data=plot_df,x='strategy', y='value', color='black', alpha=0.3,  size=2)
            #sns.violinplot(data=plot_df, x='strategy', y='value')
            
            plt.title(f'Boxplot of {config_tecnas.plot_convergency_generation_status_title_names[column]}', fontsize = 10)
            plt.ylabel(column, fontsize = 10)
            plt.xlabel('Strategy', fontsize = 10)
            plt.xticks(rotation=90, ha='right', fontsize=6)
            plt.yticks(fontsize=6)
            plt.tight_layout()
            plot_path = os.path.join(self.boxplots, f'boxplot_{column}.png')
            self._save_plot(plot_path, dpi=200)
            print(f'Boxplot boxplot_{column}.png saved successfully!\n')
    def plot_convergence_exec(self, show_mean=True):
        output_folder = os.path.join(self.plot_folder, 'convergence_executions')
        self._create_folders(output_folder)
        for filename, filepath in self._iter_csv_files(self.splitted_folder):
            print(f'\nProcessing convergence: {filename}')
            df = self._load_csv(filepath)
            exec_list = sorted(df['Execution'].unique())
            generations = sorted(df['Generation'].unique())
            for column in config_tecnas.plot_convergency_generation_status_columns:
                if column not in df.columns:
                    continue
                fig, ax = plt.subplots( figsize=self.DEFAULT_FIGSIZE )
                all_exec_values = []
                for exec_id in exec_list:
                    df_exec = df[df['Execution'] == exec_id]
                    values = [ ( df_exec[ df_exec['Generation'] == gen][column].max()  if len(df_exec[df_exec['Generation'] == gen]) > 0 else np.nan) for gen in generations]
                    all_exec_values.append(values)
                    sns.lineplot(x=generations,  y=values,  ax=ax,  linestyle='--', linewidth=1,  alpha=0.5)
                if show_mean:
                    mean_values = np.nanmean(np.array(all_exec_values), axis=0 )
                    sns.lineplot(x=generations, y=mean_values, ax=ax, color='darkblue', linewidth=3, linestyle='--', label='Mean')
                GO_str = filename[:-4].replace('_generation_status', '')
                ax.set_title(f'{GO_str} Convergence of {config_tecnas.plot_convergency_generation_status_title_names[column]}\n{config_tecnas.EXECUTIONS} executions and mean trajectory', fontsize=10, pad=20)
                self._apply_axis_style(ax, ylabel=column, ticks=self._get_ticks(generations))
                ax.legend(loc='best', fontsize = 6)
                filename_folder = os.path.join(output_folder, GO_str)
                ensure_folder_exists(filename_folder)
                output_name = (f'{GO_str}_convergence_{column}.png' )
                plot_path = os.path.join(filename_folder, output_name)
                self._save_plot(plot_path)
                print(f'Saving: {output_name}')

    def plot_convergence_exec_grouped(self, show_mean=True):
        columns = config_tecnas.plot_convergency_generation_status_columns
        arch_columns = ['mean_accuracy', 'mean_flops', 'mean_params', 'best_acc', 'FLOPs_bestarch', 'Params_bestarch']
        search_columns = [ c for c in columns.keys() if c not in arch_columns]
        grouped_configs = [  ('archs_performance.png', arch_columns),  ('search_performance.png', search_columns) ]
        output_folder = os.path.join(self.plot_folder, 'convergence_executions_grouped')
        ensure_folder_exists(output_folder)
        all_data = {}
        for filename, filepath in self._iter_csv_files(self.splitted_folder):
            print(f'Loading convergence data: {filename}')
            df = self._load_csv(filepath)
            go_str = filename[:-4].replace('_generation_status', '')
            all_data[go_str] = { 'df': df, 'executions': sorted(df['Execution'].unique()), 'generations': sorted(df['Generation'].unique())     }
        for strategy_name, data in all_data.items():
            strategy_folder = os.path.join(output_folder, strategy_name)
            ensure_folder_exists(strategy_folder)
            for fig_filename, cols_group in grouped_configs:
                valid_columns = [ column for column in cols_group if column in data['df'].columns]
                nrows = 2
                ncols = math.ceil(len(valid_columns) / nrows)
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows), sharex=True)
                axes = np.array(axes).reshape(-1)
                subplot_labels = 'abcdefghijklmnopqrstuvwxyz'
                for ax, column in zip(axes, valid_columns):
                    df = data['df']
                    exec_list = data['executions']
                    generations = data['generations']
                    all_exec_values = []
                    for exec_id in exec_list:
                        df_exec = df[df['Execution'] == exec_id]
                        mode = columns[column][1]
                        values = []
                        for gen in generations:
                            df_gen = df_exec[df_exec['Generation'] == gen]
                            if len(df_gen) > 0:
                                if mode == 'max':
                                    value = df_gen[column].max()
                                else:
                                    value = df_gen[column].min()
                            else:
                                value = np.nan
                            values.append(value)
                        all_exec_values.append(values)
                        sns.lineplot(x=generations, y=values, ax=ax, linewidth=0.7, alpha=0.15, legend=False)
                    if show_mean:
                        mean_values = np.nanmean(np.array(all_exec_values), axis=0)
                        mutation = strategy_name.split('_')[1]
                        color = self.MUTATION_COLORS.get(mutation, 'black')
                        if strategy_name == 'NONE_NONE':
                            color = 'black'
                        sns.lineplot(x=generations, y=mean_values, ax=ax, color=color, linewidth=2.5, label=strategy_name)
                    plot_idx = valid_columns.index(column)
                    label = subplot_labels[plot_idx]
                    ax.set_title(f'{label}) {config_tecnas.plot_convergency_generation_status_title_names[column]}', fontsize=10, pad=15)
                    #ax.set_title(config_tecnas.plot_convergency_generation_status_title_names[column], fontsize=8, pad=15)
                    self._apply_axis_style(ax, ylabel=column, ticks=self._get_ticks(generations))
                    ax.legend(loc='best', fontsize=4, frameon=True, framealpha=0.3)
                for idx in range(len(valid_columns), len(axes)):
                    axes[idx].axis('off')
                #plt.tight_layout(rect=[0, 0, 0.82, 1])
                plt.tight_layout()
                figure_path = os.path.join(strategy_folder, f'{strategy_name}_{fig_filename}')
                plt.savefig(figure_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f'Grouped convergence plot saved -> {figure_path}')

    def plot_acc_loss_arch(self):
        output_folder = os.path.join(self.plot_folder,'convergence_individual_archs')
        self._create_folders(output_folder)
        for filename, filepath in self._iter_csv_files(self.experiment_folder):
            df = self._load_csv(filepath)
            for arch in df['ID'].unique():
                print(f'Processing architecture: {arch}')
                df_arch = df[df['ID'] == arch]
                epochs = list( range(1, df_arch['Epochs'].iloc[0] + 1 ))
                acc_history = ast.literal_eval( df_arch['Accuracy_history'].iloc[0])
                loss_history = ast.literal_eval(df_arch['Loss_history'].iloc[0])
                plt.figure(figsize=(10, 6))
                plt.plot( epochs,  acc_history, label='Accuracy' )
                plt.plot( epochs, loss_history, label='Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy / Loss')
                plt.title(f'Architecture {arch}')
                plt.legend()
                plot_path = os.path.join(output_folder, f'{arch}.png')
                self._save_plot(plot_path)

    def plot_grouped_generation_status(self, nsga2_window=False, slice_generations = False, use_random_colors=False, plot_raw_data=False, window_tend=5, k_best=35):
        columns = config_tecnas.plot_convergency_generation_status_columns
        arch_columns = ['mean_accuracy','mean_flops', 'mean_params', 'best_acc','FLOPs_bestarch', 'Params_bestarch' ]
        search_columns = [ c for c in columns.keys() if c not in arch_columns ]
        grouped_configs = [  ('archs_performance.png', arch_columns), ('search_performance.png', search_columns) ]
        self.generation_status_folder = os.path.join(self.plot_folder, 'generation_status'   )
        ensure_folder_exists(self.generation_status_folder)
        all_data = self._load_generation_status_data()
        #print(all_data[list(all_data.keys())[0]]['df'].columns)
        
        for fig_filename, cols_group in grouped_configs:
            valid_columns = [column for column in cols_group  if any(column in data['df'].columns for data in all_data.values() )]
            nrows = 2
            ncols = math.ceil(len(valid_columns) / nrows)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows), sharex=False)
            axes = np.array(axes).reshape(-1)
            subplot_labels = 'abcdefghijklmnopqrstuvwxyz'

            for ax, column in zip(axes, valid_columns):
                mode = columns[column][1]
                ranked_data = self._rank_operators_by_auc(all_data, column, mode)
                top_k_data = ranked_data[:k_best]
                starts_at = self._plot_top_k_operators(
                    ax=ax,
                    top_k=top_k_data,
                    column=column,
                    mode=mode,
                    plot_raw_data=plot_raw_data,
                    nsga2_window=nsga2_window,
                    slice_generations=slice_generations,
                    use_random_colors=use_random_colors
                )
                plot_idx = valid_columns.index(column)
                label = subplot_labels[plot_idx]
                ax.set_title(f'{label}) {config_tecnas.plot_convergency_generation_status_title_names[column]}', fontsize=10, pad=15)
                if slice_generations: #Adjust the generations fow window method
                    all_generations = np.concatenate([data['generations'][int(len(data['generations']) * starts_at):] for _, _, data in top_k_data ])
                else:
                    all_generations = np.concatenate([data['generations']for _, _, data in top_k_data ])
                #ax.set_title(config_tecnas.plot_convergency_generation_status_title_names[column], fontsize=8, pad=15)
                self._apply_axis_style(ax, ylabel=column, ticks=self._get_ticks(all_generations) )
                ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
                ax.legend(loc='best', fontsize=6, frameon=True, framealpha=0.3)

            for idx in range(len(valid_columns), len(axes)):
                axes[idx].axis('off')
            plt.tight_layout(rect=[0, 0, 0.82, 1])
            folder_name = f'top{k_best}_grouped_'
            folder_name += '_window' if nsga2_window else ''
            folder_name += '_slicedGenerations' if slice_generations else ''
            grouped_folder = os.path.join(self.generation_status_folder, folder_name)
            ensure_folder_exists(grouped_folder)
            figure_path = os.path.join(grouped_folder, fig_filename)
            plt.savefig(figure_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'Grouped plot saved -> {fig_filename}')

    def plot_generation_status(self,individual_plots=False, topK_plots = False, nsga2_window=False,use_random_colors=False,plot_raw_data=False,window_tend=5, k_best=35):
        columns = config_tecnas.plot_convergency_generation_status_columns
        self.generation_status_folder = os.path.join(self.plot_folder, 'generation_status' )
        self._create_folders(self.generation_status_folder)
        all_data = self._load_generation_status_data()

        if topK_plots:
            for column, (col_type, mode) in columns.items():
                ranked_data = self._rank_operators_by_auc(all_data, column, mode)
                top_k_data = ranked_data[:k_best]
                fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
                self._plot_top_k_operators(ax=ax, top_k=top_k_data, column=column, mode=mode, plot_raw_data=plot_raw_data, nsga2_window=nsga2_window, use_random_colors=use_random_colors)
                ax.set_title(f'Operators by AUC\n{config_tecnas.plot_convergency_generation_status_title_names[column]}', fontsize=16, pad=20)
                all_generations = np.concatenate([data['generations']  for _, _, data in top_k_data])
                self._apply_axis_style(ax,  ylabel=column, ticks=self._get_ticks(all_generations) )
                ax.legend( loc='best', fontsize=9, frameon=True, framealpha=0.3)
                plt.tight_layout()
                output_folder = os.path.join(self.generation_status_folder, f'top{k_best}_by_auc' )
                ensure_folder_exists(output_folder)
                plot_path = os.path.join(output_folder,f'top{k_best}_{column}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight' )
                plt.close()
                print(f'Plot saved -> top{k_best}_{column}.png')
        if individual_plots:
            for stem, data in all_data.items():
                df = data['df']
                generations = data['generations']
                for column, (col_type, mode) in columns.items():
                    fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
                    values = df[column].values
                    tendency = self._compute_tendency(values, window_ratio=0.1)
                    best_gen, best_val, _ = self._find_best_point(tendency,generations,mode)
                    sns.lineplot(
                        x=generations,
                        y=values,
                        ax=ax,
                        marker='o',
                        color='blue',
                        markersize=5,
                        markeredgecolor='black',
                        markeredgewidth=0.5,
                        linewidth=1.5,
                        alpha=0.7
                    )
                    sns.lineplot(x=generations, y=tendency, ax=ax,color='black', linestyle='--', linewidth=2.5)
                    ax.scatter(best_gen, best_val, color='red', marker='X',s=120,zorder=5)
                    ax.axvline(best_gen, linestyle='--',color='black',linewidth=0.5)
                    ax.axhline(best_val, linestyle='--', color='black', linewidth=0.5 )
                    # ==================================================
                    # NSGA2 WINDOWS
                    # ==================================================
                    if nsga2_window:
                        for line in self.NSGA2_LINES:
                            ax.axvline(x=line,color='red',linestyle=':',linewidth=1,alpha=0.9)
                    ax.set_title(f'{data["search_name"]} - {column} per Generation')
                    self._apply_axis_style(ax, ylabel=column, ticks=self._get_ticks(generations))
                    plt.tight_layout()

                    # ==================================================
                    # SAVE BY SEARCH NAME
                    # ==================================================
                    folder_by_name = os.path.join(self.generation_status_folder,stem)
                    ensure_folder_exists(folder_by_name)
                    plot_by_name = os.path.join(folder_by_name, f'{stem}_{column}.png')
                    plt.savefig(plot_by_name,dpi=300,bbox_inches='tight')
                    # ==================================================
                    # SAVE BY METRIC
                    # ==================================================
                    folder_by_metric = os.path.join(self.generation_status_folder,column)
                    ensure_folder_exists(folder_by_metric)
                    plot_by_metric = os.path.join(folder_by_metric,f'{stem}_{column}.png')
                    plt.savefig(plot_by_metric,dpi=300,bbox_inches='tight')
                    plt.close()
                    print(f'Saved -> {stem}_{column}.png')
    # =============================================================================== ALL SUBFOLDERS ============================================================================
    def plot_grouped_generation_status_all_subfolders(self,   root_folder,  nsga2_window=False, slice_generations=False, use_random_colors=False,  plot_raw_data=False,  window_tend=5,  k_best=35  ):
        #self.experiment_folder = experiment_folder
        #self.plot_folder = os.path.join(experiment_folder, 'plots')
        #self.splitted_folder = os.path.join(experiment_folder, 'splitted')
        #self.plot_folder = os.path.join(experiment_folder, 'plots')
        original_folder = self.splitted_folder
        splitted_name = os.path.basename(original_folder)
        for entry in os.scandir(root_folder):
            splitted_path = os.path.join(entry.path, splitted_name)
            if not os.path.isdir(splitted_path):
                print(f"Skipping {entry.name}: {splitted_name}' folder not found" )
                continue
            print(f"Processing {splitted_path}")
            self.splitted_folder = splitted_path
            self.plot_folder = os.path.join(entry.path, 'plots')
            self.plot_grouped_generation_status(nsga2_window=nsga2_window, slice_generations=slice_generations, use_random_colors=use_random_colors, plot_raw_data=plot_raw_data, window_tend=window_tend, k_best=k_best)
        