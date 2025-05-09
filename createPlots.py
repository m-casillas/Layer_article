from PlotterENAS import *
from TECNAS import *

tecNAS = TECNAS()
if RUN_ENAS == False:
    plotter = PlotterENAS(tecNAS)
    bar_columns_all = ['Acc_mean', "Loss_mean", "CPU_hrs_mean", "Num_params_mean", "FLOPs_mean"]
    box_columns_all = ["Best_accuracy", "Loss", "CPU_hrs", "Num_params", "FLOPs"]

    bar_columns = ["FLOPs_mean"]
    box_columns = ["FLOPs"]
    plotter.create_grouped_plots(tecNAS.general_report_filenames, bar_columns, box_columns)
    #tecNAS.plot_accuracy_loss_fromFiles(tecNAS.medians_report_filenames, 'A')
    #tecNAS.plot_accuracy_loss_fromFiles(tecNAS.medians_report_filenames, 'L')