from globalsENAS import *
from utilitiesENAS import *
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


#gen_list = [{'INP':28}, {'CONV':[32,3]}, {'POOLMAX':2}, {'CONV':[64,3]}, {'POOLMAX':2}, {'FLATTEN':None}, {'DENSE':[64,'relu']}, {'DENSE':[10,'softmax']}]
#genotype = Genotype('L', 'IV', gen_list)
#arch = Architecture('X', 0, 0, 0, 0, 1, genotype)

#
class PlotterENAS:


    def plot_bars_matplotlib(self, reportCSV_list, column_list):

        # Columns to plot
        columns = column_list
        x = np.arange(len(columns))  # the label locations
        # Read the data from the files and store the values
        values_list = []
        labels = []
        for idx, file in enumerate(reportCSV_list):
            df = pd.read_csv(file)
            values = df.iloc[0][columns].values.astype(float)
            values_list.append(values)
            #labels.append(f'File{idx+1}')
            label = determine_label_filename(file)
            labels.append(label)
        N = len(reportCSV_list)
        width = 0.8 / N  # total bar width is 0.8, divide equally among files
        offsets = np.linspace(-0.4 + width/2, 0.4 - width/2, N)
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size as needed

        for i in range(N):
            ax.bar(x + offsets[i], values_list[i], width, label=labels[i])
        # Add labels, title, and custom x-axis tick labels
        ax.set_ylabel('Values')
        ax.set_title('Performance measures grouped by\nmutation operator (or random search)')
        ax.set_xticks(x)
        ax.set_xticklabels(columns)
        ax.legend()
        last_name = create_last_name(column_list)

        path_fig = os.path.join(path_figures, f'{self.tecNASobj.filename}_bars_{last_name}.png')
        plt.savefig(path_fig)
        plt.show()
        print(f'Figure saved in {path_fig}')
        plt.clf()
        plt.close()

    def plot_grouped_boxplots(self, reportCSV_list, column_list):
        # Create a list to store data for each column
        column_data = {column: [] for column in column_list}

        # Read the data from each file and store values for each column
        for file in reportCSV_list:
            df = pd.read_csv(file)
            for column in column_list:
                if column in df.columns:
                    column_data[column].append(df[column].dropna().values)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Number of columns and files
        num_columns = len(column_list)
        num_files = len(reportCSV_list)

        # Set width for each boxplot group and calculate offsets
        width = 0.6#0.6
        offsets = np.linspace(-width/3, width/3, num_files)

        # Use a colormap to generate distinct colors for each boxplot
        color_map = cm.get_cmap('Set1', num_files)

        # Plot each column as a grouped boxplot and store patch handles for legend
        legend_handles = []
        for i, column in enumerate(column_list):
            for j in range(num_files):
                pos = i + offsets[j]
                color = color_map(j)  # Get a distinct color for each file
                box = ax.boxplot(column_data[column][j], positions=[pos], widths=width / (num_files + 1),
                                patch_artist=True, showfliers=False,
                                boxprops=dict(facecolor=color, color='black'),
                                medianprops=dict(color='black'))
                if i == 0:  # Add to legend only once per file
                    labelPlot = determine_label_filename(reportCSV_list[j])
                    legend_handles.append(plt.Line2D([0], [0], color=color, lw=4, label=labelPlot))
                    #legend_handles.append(plt.Line2D([0], [0], color=color, lw=4, label=os.path.basename(reportCSV_list[j])))

        # Customize the plot
        ax.set_xticks(range(num_columns))
        ax.set_xticklabels(column_list)
        ax.set_title('Boxplots of Performance Measures')
        ax.set_ylabel('Values')
        ax.legend(handles=legend_handles, loc='best')

        # Save and display the figure
        last_name = create_last_name(column_list)
        path_fig = os.path.join(path_figures, f'{self.tecNASobj.filename}_boxes_{last_name}.png')
        plt.savefig(path_fig)
        plt.show()
        print(f'Figure saved in {path_fig}')
        plt.clf()
        plt.close()

    def plot_accuracy_loss(self, history):
        # Plotting the training and validation accuracy and loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()
        path_fig = os.path.join(path_figures, f'{self.filename}.png')
        plt.savefig(path_fig)
        print(f'Figure saved in {path_fig}')
        plt.close()

    def plot_accuracy_loss_histories(self, plot_type):
        plt.figure(figsize=(10, 6))
        acc_loss_history = self.accuracy_histories if plot_type == 'A' else self.loss_histories
        title = 'Validation Accuracy Histories' if plot_type == 'A' else 'Training Loss Histories'
        ylabel = 'Accuracy' if plot_type == 'A' else 'Loss'
        # Plot each network's validation accuracy history
        epoch_list = [str(e) for e in range(1, EPOCHS + 1)]
        for i, history in enumerate(acc_loss_history):
            plt.plot(epoch_list, history, label=f'Execution {i+1}')
        # Add labels and legend
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        title_aux = determine_label_filename(self.filename)
        title = f'{title} - {title_aux}'
        plt.title(title)
        plt.legend()
        plt.grid(True)
        path_fig = os.path.join(path_figures, f'{self.tecNASobj.filename}_{ylabel}_exec.png')
        plt.savefig(path_fig)
        plt.show()
        print(f'Figure saved in {path_fig}')
        plt.clf()
        plt.close()

    def plot_accuracy_loss_fromFiles(self, reportFileNamesLst, plot_type):
        #Get acc/loss from the MEDIAN REPORTS per EPOCH and plot them

        plt.figure(figsize=(10, 6))  # Set the figure size
        value_name = 'Best_accuracy' if plot_type == 'A' else 'Loss'
        for file in reportFileNamesLst:
            # Read the CSV file into a DataFrame
            pathfile = os.path.join(path_results, f'{file}.csv')
            data = pd.read_csv(pathfile)
            plt.plot(data['Epochs'], data[value_name], label=determine_label_filename(file))
        plt.xlabel('EPOCHS')
        plt.ylabel(value_name)
        plt.title(f'{value_name} per Epoch for Median architectures')
        plt.legend()
        path_fig = os.path.join(path_figures, f'{self.tecNASobj.filename}_{value_name}_medians.png')
        plt.savefig(path_fig)
        plt.show()
        print(f'Figure saved in {path_fig}')
        plt.clf()
        plt.close()

    def create_grouped_plots(self, filenames, bar_columns, boxplot_columns):
        reportCSV_list = []
        for filename in filenames:
            path = os.path.join(path_results, f'{filename}.csv')
            reportCSV_list.append(path)

        self.plot_bars_matplotlib(reportCSV_list, bar_columns)
        self.plot_grouped_boxplots(reportCSV_list, boxplot_columns)

  
    def __init__(self, tecNASobj):
        self.tecNASobj = tecNASobj
        pass