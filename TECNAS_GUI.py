
import FreeSimpleGUI as sg

from file_handler import split_all_csvs_in_folder, merge_csv_files, ensure_folder_exists, split, merge, organize_csv
from ReportENAS import ReportENAS
from PlotterENAS import Plotter
from globalsENAS import *
from Surrogate_ENAS import Surrogate_ENAS
from Stats_ENAS import *


def plot_medians(experiment_folder):
    #Plot median measures
    input_folder = experiment_folder
    output_folder = os.path.join(experiment_folder, 'plots')
    ensure_folder_exists(input_folder)
    ensure_folder_exists(output_folder)
    plotter = Plotter(input_folder)
    plotter.plot_medians_from_folder(output_folder)

def summarize_indicators_folder(experiment_folder = '', archs_info = True):
    splitted_files_path = os.path.join(experiment_folder, 'splitted')
    summaries_path = os.path.join(experiment_folder, 'bestarchs_summaries' if archs_info == True else 'GAs_summaries')
    reporter = ReportENAS()
    ensure_folder_exists(splitted_files_path)
    ensure_folder_exists(summaries_path)
    reporter.summarize_indicators(input_folder = splitted_files_path, output_folder = summaries_path, archs_info = archs_info)

        
def train_surrogates_folder(individual_archs_CSV_folder, model_choice):
    
    merge(individual_archs_CSV_folder)
    regressor_folder = os.path.join(individual_archs_CSV_folder, 'regressors')
    merged_path = os.path.join(individual_archs_CSV_folder, 'merged.csv')
    surrogate = Surrogate_ENAS(model_choice, regressor_folder = regressor_folder, archs_CSV = merged_path, SAVE_MODELS = True, TRAINING_NEW_MODELS = True)
    surrogate.train_all_surrogates()

def rank1_statistics_folder(experiment_folder):
    ensure_folder_exists(experiment_folder)
    rank1_stats(rank1_folder = experiment_folder)

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

def organizerFrame():
    layout = [[sg.Text('Experiment Folder'), sg.InputText(default_input_path,key='organizer_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path)],
              [sg.Check('CROSS', key = 'CHK_CROSS'), sg.Check('MUT', key = 'CHK_MUT'), sg.Check('CROSS_MUT', key = 'CHK_CROSS_MUT'), sg.Check('Only window', key = 'CHK_ONLY_WINDOW')],
              [sg.Button('Organize', key = 'BTN_ORGANIZE')]]
    frm = sg.Frame('Organize CSVs', layout)
    return frm

def summarizeFrame():
    layout = [[sg.Text('Experiment Folder'), sg.InputText(default_input_path,key='summarize_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path)],
              [sg.Check('Best archs', key = 'CHK_BEST_ARCHS'), sg.Check('GAs', key = 'CHK_GAS')],
              [sg.Button('Summarize', key = 'BTN_SUMMARIZE')]]
    frm = sg.Frame('Summarize', layout)
    return frm
def csvToolsFrame():
    layout = [ [splitFrame()], [mergeFrame()], [organizerFrame()], [summarizeFrame()]]
    return sg.Frame('CSV Tools', layout)

def plotsFrame():
    generation_status_frame = sg.Frame('Generation Status', [ [
        sg.Check('Generation status', key='CHK_GENERATION_STATUS_PLOT')],
        [
        sg.Check('Individual gstatus plots', key='CHK_GSTATUS_INDIVIDUAL'),
        sg.Check('Top_K gstatus plots', key='CHK_GSTATUS_TOP_K')],
        [sg.Check('Gen Status all subfolders', key='CHK_GENERATION_STATUS_SUBFOLDERS'),sg.Check('Grouped gstatus plot (no window)', key='CHK_GSTATUS_GROUPED'), sg.Check('Win Method (special markers)', key = 'CHK_WIN_METHOD'),
         sg.Check('Slice generations', key = 'CHK_SLICE_GENERATIONS')]
    ])
    layout = [
        [sg.Text('Experiment Folder'), sg.InputText(default_input_path, key='plots_input_folder', size=(None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path)],
        [sg.Check('Summarized measures', key='CHK_SUMMARIES_PLOTS'), sg.Check('Best archs', key='CHK_BESTARCHS_SUMMARIZE_PLOTS'), sg.Check('GAs', key='CHK_GAS_SUMMARIZE_PLOTS')],
        [sg.Check('Convergence single archs', key='CHK_CONV_SINGLEARCHS_PLOT')],
        [sg.Check('Convergence per Execution', key='CHK_CONVERGENCE_EXEC_PLOT'), sg.Check('Convergence grouped', key='CHK_CONVERGENCE_GROUPED')],
        [sg.Check('Medians', key='CHK_MEDIANS_PLOT')],
        [sg.Check('Boxplots', key='CHK_BOXPLOTS')],
        [generation_status_frame],
        [sg.Button('Plot', key='BTN_PLOT')]
    ]
    return sg.Frame('Plots', layout)

def allFrame():
    txt1 = sg.Text('Split - summarize - plots - Wilcoxon - Correlation')
    layout = [[txt1], [sg.Text('Experiment Folder'), sg.InputText(default_input_path,key='all_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path)],
              [sg.Button('Whole process!', key = 'BTN_ALL')]]
    frm = sg.Frame('All', layout)
    return frm

def train_surrogate_frame():
    layout = [[sg.Text('CSVs Folder'), sg.InputText(default_input_path,key='surrogate_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path)],
              [sg.Text('Regressor model choice'), sg.Slider(range=(0, 10), orientation='h', size=(20, 15), default_value=10, key = 'SLD_MODEL_CHOICE')],
              [sg.Button('Train and create regressors (config_tecnas)', key = 'BTN_TRAIN_SURROGATES')]]
    frm = sg.Frame('Train Surrogates', layout)
    return frm

def wilcoxonFrame():
    layout = [[sg.Text('Experiment Folder'), sg.InputText(default_input_path,key='wilcoxon_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path)],
              [sg.Button('Wilcoxon', key = 'BTN_WILCOXON')]]
    frm = sg.Frame('Wilcoxon', layout)
    return frm

def correlationFrame():
    layout = [[sg.Text('Experiment Folder'), sg.InputText(default_input_path,key='correlation_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path), sg.Checkbox('Per execution', key = 'CHK_CORR_PER_EXECUTION')],
              [sg.Button('Correlation', key = 'BTN_CORRELATION')]]
    frm = sg.Frame('Correlation', layout)
    return frm

def flops_params_frame():
    layout = [[sg.Text('Experiment Folder'), sg.InputText(default_input_path,key='flops_params_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path), sg.Check('Real metrics', key = 'CHK_REAL_METRICS')],
              [sg.Button('Flops and params', key = 'BTN_FLOPS_PARAMS')]]
    frm = sg.Frame('Flops and params', layout)
    return frm

def filter_csv_frame():
    chk_gen = sg.Combo(['ALL', 'LAST'], default_value='ALL', key = 'CMB_GENERATION')
    cmb_status = sg.Combo(['ALL', 'BEST', 'MEDIAN', 'WORST', 'BLANK'], default_value='BEST', key = 'CMB_ARCH_STATUS')
    cmb_rank = sg.Combo(['ALL', 'HIGHEST', 'MIDDLE'], default_value='ALL', key = 'CMB_RANK')
    sld_nparts = sg.Slider(range=(1, 10), orientation='h', size=(20, 15), default_value=1, key = 'SLD_NPARTS')
    layout = [[sg.Text('Experiment Folder'), sg.InputText(default_input_path,key='filter_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path)],
              [sg.Text('Generation'), chk_gen, sg.Text('Status'), cmb_status, sg.Text('Rank'), cmb_rank, sg.Text('N parts'), sld_nparts],
              [sg.Button('Filter', key = 'BTN_FILTER')]]
    frm = sg.Frame('Filter', layout)
    return frm

def rank1_stats_frame():
    layout = [[sg.Text('Experiment Folder'), sg.InputText(default_input_path,key='rank1_stats_input_folder', size = (None, 10)), sg.FolderBrowse(initial_folder=default_experiment_path)],
              [sg.Button('Rank 1 Stats', key = 'BTN_RANK1_STATS')]]
    frm = sg.Frame('Rank 1 Stats', layout)
    return frm

def main():
    os.system("cls")
    layout = [[csvToolsFrame(), plotsFrame()], [allFrame(), train_surrogate_frame(), wilcoxonFrame(), correlationFrame()], [flops_params_frame(),filter_csv_frame(),rank1_stats_frame()],
              [sg.Button('EXIT', key = 'BTN_EXIT')]]
    window = sg.Window('TECNAS', layout)
    while True:
        start_time = time.time()

        event, values = window.read()
        if event == 'BTN_SPLIT_DEFAULT_FOLDERS':
            split(values['split_input_folder'])
            sg.popup('CSV files split successfully!')

        elif event == 'BTN_MERGE_DEFAULT_FOLDERS':
            merge(values['merge_input_folder'])
            sg.popup('CSV files merged successfully!')

        elif event == 'BTN_SUMMARIZE':
            if values['CHK_BEST_ARCHS']:
                summarize_indicators_folder(experiment_folder = values['summarize_input_folder'], archs_info = True)
                sg.popup('Best architectures summaries created successfully!')
            if values['CHK_GAS']:
                summarize_indicators_folder(experiment_folder = values['summarize_input_folder'], archs_info = False)
                sg.popup('GA summaries created successfully!')
        elif event == 'BTN_ORGANIZE':
            if values['CHK_CROSS']:
                organize_csv(folder_path = values['organizer_input_folder'], cross_family=True, mut_family=False, only_window=False)
                sg.popup('CSVs organized by crossover successfully!')
            if values['CHK_MUT']:
                organize_csv(folder_path = values['organizer_input_folder'], cross_family=False, mut_family=True, only_window=False)
                sg.popup('CSVs organized by mutation successfully!')
            if values['CHK_CROSS_MUT']:
                organize_csv(folder_path = values['organizer_input_folder'], cross_family=True, mut_family=True, only_window=False)
                sg.popup('CSVs organized by crossover and mutation successfully!')
            if values['CHK_ONLY_WINDOW']:
                organize_csv(folder_path = values['organizer_input_folder'], cross_family=False, mut_family=False, only_window=True)
                sg.popup('CSVs organized by window successfully!')

        elif event == 'BTN_PLOT':
            plotter = Plotter(values['plots_input_folder'])
            if values['CHK_SUMMARIES_PLOTS']:
                if values['CHK_BESTARCHS_SUMMARIZE_PLOTS']:
                    plotter.plot_measures_from_folder(plot_bestarchs = True)
                if values['CHK_GAS_SUMMARIZE_PLOTS']:
                    plotter.plot_measures_from_folder(plot_bestarchs = False)
                sg.popup('Summarized measures plots created successfully!')
            if values['CHK_CONV_SINGLEARCHS_PLOT']:
                plotter.plot_acc_loss_arch()
                sg.popup('Convergence single architectures plots created successfully!')
            if values['CHK_CONVERGENCE_EXEC_PLOT']:
                plotter.plot_convergence_exec()
                sg.popup('Convergence per execution plots created successfully!')
            if values['CHK_CONVERGENCE_GROUPED']:
                plotter.plot_convergence_exec_grouped()
                sg.popup('Convergence grouped plots created successfully!')
            if values['CHK_MEDIANS_PLOT']:
                plotter.get_all_medians_folder()
                sg.popup('Medians plots created successfully!')
            if values['CHK_BOXPLOTS']:
                plotter.boxplot_from_folder()
                sg.popup('Boxplots created successfully!')
            if values['CHK_GENERATION_STATUS_PLOT']: 
                if values['CHK_GSTATUS_INDIVIDUAL']:
                    print('Individual convergence plots for generation status begin')
                    plotter.plot_generation_status(individual_plots = True)
                    print('Individual convergence plots for generation status done\n')
                if values['CHK_GSTATUS_TOP_K']:
                    print('Top K convergence plots for generation status begin')
                    plotter.plot_generation_status(topK_plots = True, nsga2_window = values['CHK_WIN_METHOD'])
                    print('Top K convergence plots for generation status done\n')
                if values['CHK_GSTATUS_GROUPED']:
                    print(f'Convergence grouped plots for generation status begins')
                    plotter.plot_grouped_generation_status_all_subfolders(root_folder = values['plots_input_folder'], nsga2_window = values['CHK_WIN_METHOD'], slice_generations = values['CHK_SLICE_GENERATIONS']) if values['CHK_GENERATION_STATUS_SUBFOLDERS'] else plotter.plot_grouped_generation_status(nsga2_window = values['CHK_WIN_METHOD'], slice_generations = values['CHK_SLICE_GENERATIONS'])
                    print(f'Convergence grouped plots for generation status done\n')
                sg.popup('Generation status plots created successfully!')

        elif event == 'BTN_TRAIN_SURROGATES':
            train_surrogates_folder(values['surrogate_input_folder'], int(values['SLD_MODEL_CHOICE']))
            sg.popup('Surrogates trained successfully!')

        elif event == 'BTN_WILCOXON':
            Wilcoxon_ENAS(values['wilcoxon_input_folder'])
            sg.popup('Wilcoxon test completed successfully!')

        elif event == 'BTN_CORRELATION':
            correlation_matrix_folder(experiment_folder = values['correlation_input_folder']) if values['CHK_CORR_PER_EXECUTION'] == False else  correlation_matrix_execution(experiment_folder = values['correlation_input_folder'])
            sg.popup('Correlation matrix plots created successfully!')

        elif event == 'BTN_FLOPS_PARAMS':
            from TECNAS import TECNAS
            flopsfolder = values['flops_params_input_folder']
            tecNAS = TECNAS(experiment_folder = flopsfolder)
            tecNAS.flops_params_folder(folder_path = flopsfolder, realMetrics = values['CHK_REAL_METRICS'])
            sg.popup('Flops and params computed successfully!')

        elif event == 'BTN_ALL':
            plotter = Plotter(values['all_input_folder'])
            print(f' ====== Whole process begins! ======')
            print(f'====== Splitting CSV ======' )
            split(values['all_input_folder'])
            print('====== Splitting completed!======\n')
            print(f'====== Correlation matrix ======' )
            #correlation_matrix_folder(values['all_input_folder'])
            print('====== Correlation matrix completed!======\n')
            print(f'====== Correlation matrix per execution ======' )
            #correlation_matrix_execution(values['all_input_folder'])
            print('====== Correlation matrix per execution completed!======\n')
            print('======Plotting convergence======')
            plotter.plot_convergence_exec()
            print('======Plotting convergence completed!======\n')
            print('====== Plotting medians ')
            plotter.get_all_medians_folder()
            print('Medians plots created successfully!')
            print('======== Boxplots ========')
            plotter.boxplot_from_folder()
            print('Boxplots created successfully!')
            print('======Summarizing best architectures======')
            summarize_indicators_folder(experiment_folder = values['all_input_folder'], archs_info = True)
            print('======Summarizing best architectures completed!======\n')
            print('======Sumnarizing GAs======')
            summarize_indicators_folder(experiment_folder = values['all_input_folder'], archs_info = False)
            print('======Summarizing GAs completed!======\n')
            print('======\nPlotting architectures measures======')
            plotter.plot_measures_from_folder(plot_bestarchs = True)
            print('======Plotting architectures measures completed!======\n')
            print('======Plotting GAs measures======')
            plotter.plot_measures_from_folder(plot_bestarchs = False)
            print('======Plotting GAs measures completed!======\n')
            print(f'====== Wilcoxon test ======' )
            #Wilcoxon_ENAS(values['all_input_folder'])
            print('====== Wilcoxon test completed!======\n')
            sg.popup('All processes completed successfully!')

        elif event == 'BTN_FILTER':
            df_folder_path = os.path.join(values['filter_input_folder'])
            generation = values['CMB_GENERATION']
            arch_status = values['CMB_ARCH_STATUS']
            rank = values['CMB_RANK']
            nparts = int(values['SLD_NPARTS'])
            filter_csv(df_folder_path = df_folder_path, df = None, generation = generation, arch_status = arch_status, rank = rank, nparts = nparts) 
            sg.popup('CSV file filtered successfully!')

        elif event == 'BTN_RANK1_STATS':
            df_folder_path = os.path.join(values['rank1_stats_input_folder'])
            rank1_statistics_folder(experiment_folder = df_folder_path)
            sg.popup('Rank 1 statistics computed successfully!')
        elif event in (sg.WIN_CLOSED, 'BTN_EXIT'):
            break
        else:
            print('Unknown event: ', event)
        end_time = time.time()

    elapsed_seconds = end_time - start_time
    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_seconds / 3600
    print()
    print(Fore.LIGHTBLUE_EX + f"PROCESS DONE!!!!!!!!!!!!!!")
    print(f"{elapsed_seconds:.1f} sec.")
    print(f"{elapsed_minutes:.1f}min.")
    print(f"{elapsed_hours:.1f} hrs.")

    window.close()


default_experiment_path = r'C:\Users\xaero\OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey\ITESM DCC\Layer_article'
#default_experiment_path = os.getcwd()
#default_input_path = ''
default_input_path = default_experiment_path

main()

#df_full_path = r'C:\Users\xaero\OneDrive\ITESM DCC\Layer_article\results\test_experiment\merged.csv'


