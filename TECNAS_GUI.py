
import FreeSimpleGUI as sg
from file_handler import split_all_csvs_in_folder, merge_csv_files, ensure_folder_exists
from ReportENAS import ReportENAS
from PlotterENAS import Plotter
import os
def split(experiment_folder):
    input_folder = experiment_folder
    output_folder = os.path.join(experiment_folder, 'splitted')
    ensure_folder_exists(input_folder)
    ensure_folder_exists(output_folder)
    split_all_csvs_in_folder(input_folder, output_folder)

def merge(experiment_folder):
    input_folder = experiment_folder
    output_folder = os.path.join(experiment_folder, 'merged')
    ensure_folder_exists(input_folder)
    ensure_folder_exists(output_folder)
    merge_csv_files(input_folder, output_folder, output_file='merged.csv')

def plot_medians(experiment_folder):
    #Plot median measures
    input_folder = experiment_folder
    output_folder = os.path.join(experiment_folder, 'plots')
    ensure_folder_exists(input_folder)
    ensure_folder_exists(output_folder)
    plotter = Plotter(input_folder)
    plotter.plot_medians_from_folder(output_folder)

def summarize_GA_folder(experiment_folder):
        #Summarize GAs performance per execution
        splitted_files_path = os.path.join(experiment_folder, 'splitted')
        GA_summaries_path = os.path.join(experiment_folder, 'GAs_summaries')
        reporter = ReportENAS()
        ensure_folder_exists(splitted_files_path)
        ensure_folder_exists(GA_summaries_path)
        reporter.summarize_GA_report(splitted_files_path, GA_summaries_path)

def summarize_bestarchs_folder(experiment_folder):
        #Summarize best architectures performance per execution
        splitted_files_path = os.path.join(experiment_folder, 'splitted')
        best_archs_summaries_path = os.path.join(experiment_folder, 'bestarchs_summaries')
        reporter = ReportENAS()
        ensure_folder_exists(splitted_files_path)
        ensure_folder_exists(best_archs_summaries_path)
        reporter.summarize_bestarchs_report(splitted_files_path, best_archs_summaries_path)

def splitFrame():
    '''layout = [[sg.Text('Input Folder'), sg.InputText(key='split_input_folder'), sg.FolderBrowse(initial_folder=os.getcwd())]'''
    layout = [[sg.Text('Experiment Folder'), sg.InputText(default_input_path, key='split_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path)],
              [sg.Button('Split Folder', key = 'BTN_SPLIT_DEFAULT_FOLDERS')]]
    frm = sg.Frame('Split CSVs', layout)
    return frm

def mergeFrame():
    layout = [[sg.Text('Experiment Folder'), sg.InputText(default_input_path,key='merge_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path)],
              [sg.Button('Merge Folder', key = 'BTN_MERGE_DEFAULT_FOLDERS')]]
    frm = sg.Frame('Merge CSVs', layout)
    return frm

def summarizeFrame():
    layout = [[sg.Text('Experiment Folder'), sg.InputText(default_input_path,key='summarize_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path)],
              [sg.Check('Best archs', key = 'CHK_BEST_ARCHS'), sg.Check('GAs', key = 'CHK_GAS')],
              [sg.Button('Summarize', key = 'BTN_SUMMARIZE')]]
    frm = sg.Frame('Summarize', layout)
    return frm

def plotsFrame():
    layout = [[sg.Text('Experiment Folder'), sg.InputText(default_input_path,key='plots_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path)],
              [sg.Check('Summarized measures', key = 'CHK_SUMMARIES_PLOTS'), sg.Check('Best archs', key = 'CHK_BESTARCHS_SUMMARIZE_PLOTS'), sg.Check('GAs', key = 'CHK_GAS_SUMMARIZE_PLOTS')], 
              [sg.Check('Convergence single archs', key = 'CHK_CONV_SINGLEARCHS_PLOT')], 
              [sg.Check('Convergence per Execution', key = 'CHK_CONVERGENCE_EXEC_PLOT')],
              [sg.Check('Medians', key = 'CHK_MEDIANS_PLOT')],
              [sg.Button('Plot', key = 'BTN_PLOT')]]
    frm = sg.Frame('Plots', layout)
    return frm

def allFrame():
    layout = [[sg.Text('Experiment Folder'), sg.InputText(default_input_path,key='all_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path)],
              [sg.Button('Whole process!', key = 'BTN_ALL')]]
    frm = sg.Frame('All', layout)
    return frm


def main():
    os.system("cls")
    layout = [[splitFrame(), mergeFrame(), summarizeFrame(), plotsFrame()], [allFrame()], [sg.Button('EXIT', key = 'BTN_EXIT')]]
    window = sg.Window('TECNAS', layout)
    while True:
        event, values = window.read()
        if event == 'BTN_SPLIT_DEFAULT_FOLDERS':
            split(values['split_input_folder'])
            sg.popup('CSV files split successfully!')

        elif event == 'BTN_MERGE_DEFAULT_FOLDERS':
            merge(values['merge_input_folder'])
            sg.popup('CSV files merged successfully!')

        elif event == 'BTN_SUMMARIZE':
             if values['CHK_GAS']:
                summarize_GA_folder(values['summarize_input_folder'])
                sg.popup('GA summaries created successfully!')
             if values['CHK_BEST_ARCHS']:
                summarize_bestarchs_folder(values['summarize_input_folder'])
                sg.popup('Best architectures summaries created successfully!')
        elif event == 'BTN_PLOT':
            plotter = Plotter(values['plots_input_folder'])
            if values['CHK_SUMMARIES_PLOTS']:
                if values['CHK_BESTARCHS_SUMMARIZE_PLOTS']:
                    columns = plotter.columns_arch
                    plotter.plot_measures_from_folder(columns)
                if values['CHK_GAS_SUMMARIZE_PLOTS']:
                    columns = plotter.columns_GA
                    plotter.plot_measures_from_folder(columns)
                sg.popup('Summarized measures plots created successfully!')
            if values['CHK_CONV_SINGLEARCHS_PLOT']:
                plotter.plot_acc_loss_arch()
                sg.popup('Convergence single architectures plots created successfully!')
            if values['CHK_CONVERGENCE_EXEC_PLOT']:
                plotter.plot_convergence_exec()
                sg.popup('Convergence per execution plots created successfully!')
            if values['CHK_MEDIANS_PLOT']:
                plotter.get_all_medians_folder()
                sg.popup('Medians plots created successfully!')

        elif event == 'BTN_ALL':
            print(f' ====== Whole process begins! ======')
            print(f'====== Splitting CSV ======' )
            split(values['all_input_folder'])
            print('====== Splitting completed!======\n')
            print('======Sumnarizing GAs======')
            summarize_GA_folder(values['all_input_folder'])
            print('======Summarizing GAs completed!======\n')
            print('======Summarizing best architectures======')
            summarize_bestarchs_folder(values['all_input_folder'])
            print('======Summarizing best architectures completed!======\n')
            plotter = Plotter(values['all_input_folder'])
            columns = plotter.columns_arch
            print('======\nPlotting architectures measures======')
            plotter.plot_measures_from_folder(columns)
            print('======Plotting architectures measures completed!======\n')
            columns = plotter.columns_GA
            print('======Plotting GAs measures======')
            plotter.plot_measures_from_folder(columns)
            print('======Plotting GAs measures completed!======\n')
            #plotter.plot_acc_loss_arch() Predicted archs dont have accuracy or loss history
            print('======Plotting convergence======')
            plotter.plot_convergence_exec()
            print('======Plotting convergence completed!======\n')
            plotter.get_all_medians_folder()
            print('Medians plots created successfully!')
            sg.popup('All processes completed successfully!')

        elif event in (sg.WIN_CLOSED, 'BTN_EXIT'):
            break

    window.close()


default_experiment_path = r'C:\Users\xaero\OneDrive\ITESM DCC\Layer_article\results\experiment2'
#default_experiment_path = os.getcwd()
#default_input_path = ''
default_input_path = default_experiment_path
#os.system("cls")
#print(os.getcwd())
#split()
main()
