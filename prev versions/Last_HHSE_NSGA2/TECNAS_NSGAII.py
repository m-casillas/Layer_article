from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
import config_tecnas

class GeneticOperator:
    def __init__(self, name = ''):
        self.name = name
        self.usage_count = 0
        self.pareto_appearance_count = 0
        self.best_children_metrics = {} #{'Accuracy': best_acc, 'FLOPs': best_flops}
    
    def __str__(self):
        return f"{self.name}, {self.best_children_metrics}"
        
        
    def evaluate_best_children(self, tecnas = None, children = None, objective_names = None, objective_maxmin_names = None, flops_params_gen_list_str_func = None):
        self.children = children
        self.objective_names = objective_names
        self.objective_maxmin_names = objective_maxmin_names
        self.children = sorted(self.children, key=lambda obj: obj.acc, reverse = True) #Best archs first
        #If FLOPs and NumParams are not part of the objectives, do not calculate them at all.
        self.getflops = True if 'FLOPs' in self.objective_maxmin_names else False
        self.get_numparams = True if 'Num_params' in self.objective_names else False
        self.children[0].flops, self.children[0].num_params, self.children[0].sizeMB = flops_params_gen_list_str_func(self.getflops, self.get_numparams, str(self.children[0].genotype.gen_list))
        self.children[0].succ_cross_ratio = tecnas.succ_cross/tecnas.total_cross if tecnas.total_cross > 0 else 0
        self.children[0].succ_mut_ratio = tecnas.succ_mut/tecnas.total_mut if tecnas.total_mut > 0 else 0
        best_acc = self.children[0].acc
        best_flops = self.children[0].flops
        best_num_params = self.children[0].num_params
        best_hp1 = self.children[0].dP1
        best_hp2 = self.children[0].dP2
        best_hbm = self.children[0].dBM
        best_succCross = self.children[0].succ_cross_ratio #0 if tecnas.total_cross == 0 else tecnas.succ_cross/tecnas.total_cross
        best_succMut = self.children[0].succ_mut_ratio #0 if tecnas.total_mut == 0 else  tecnas.succ_mut/tecnas.total_mut
        mean_acc = np.mean([child.acc for child in self.children])
        mean_flops = np.mean([child.flops for child in self.children])
        mean_num_params = np.mean([child.num_params for child in self.children])
        mean_dbm = np.mean([child.dBM for child in self.children])
        mean_dp1 = np.mean([child.dP1 for child in self.children])
        mean_dp2 = np.mean([child.dP2 for child in self.children])

        all_metric_dict = {'Accuracy': best_acc, 'FLOPs': best_flops, 'Num_params': best_num_params, 'DP1': best_hp1, 'DP2': best_hp2, 'DBM': best_hbm, 'Succ_Cross': best_succCross, 'Succ_Mut': best_succMut,
                            'mean_Accuracy': mean_acc, 'mean_FLOPs': mean_flops, 'mean_Num_params': mean_num_params, 'mean_DBM': mean_dbm, 'mean_DP1': mean_dp1, 'mean_DP2': mean_dp2}
        self.best_children_metrics = {metric_name:all_metric_dict[metric_name] for metric_name in all_metric_dict}#self.objective_names if metric_name in all_metric_dict}

       

class TECNAS_NSGAII:
    def __init__(self, tecnas_obj=None, objective_maxmin_names  = None, all_operators_names = []):
        self.tecnas = tecnas_obj
        self.operator_usage = {} #Keeps record of how many times each operator was selected from the Pareto front in one execution. Keys: operator names, Values: usage count obtained from GeneticOperator objects
        self.pareto_operator_appearance = {} #Keeps record of how many times each operator appeared in the Pareto front
        self.objective_maxmin_names = objective_maxmin_names #{'Accuracy': 'MAX', 'FLOPs': 'MIN', 'Num_params': 'MIN'}
        self.metric_names = list(self.objective_maxmin_names.keys()) #['Accuracy', 'FLOPs', 'Num_params']
        self.operator_names = []  #['SPC_MPAR', 'SPC_MSWAP', ...]
        self.genetic_operators_obj_dict = {op_name:GeneticOperator(op_name) for op_name in all_operators_names} #Dict of GeneticOperator objects
        self.NSGA_II_SELECTION = config_tecnas.NSGA_II_SELECTION 

    def sort_operators_by(self, dict_operators, metric_name, descending = True):
        sorted_ops_list = sorted(dict_operators.values(), key=lambda go: go.best_children_metrics[metric_name], reverse = descending)
        names = [op.name for op in sorted_ops_list]
        return names


    def save_aggregate_operator_stats_csv(self, root_folder):
        all_dfs = []
        for sub in os.listdir(root_folder):
            sub_path = os.path.join(root_folder, sub)
            if os.path.isdir(sub_path):
                csv_path = os.path.join(sub_path, "operator_stats.csv")
                df = pd.read_csv(csv_path)
                all_dfs.append(df)
        df_total = pd.concat(all_dfs, ignore_index=True)
        df_summary = df_total.groupby("Genetic_operators", as_index=False)[["Times_Used", "Pareto_Appearances"]].sum()
        output_csv = os.path.join(root_folder, "operator_stats_totals.csv")
        df_summary.to_csv(output_csv, index=False)
        print(f"CSV operator stats total :\n{output_csv}")

        plt.figure(figsize=(10,4))
        plt.bar(df_summary["Genetic_operators"], df_summary["Times_Used"])
        plt.xticks(rotation=45, ha='right')
        plt.title("Times Used per Operator")
        plt.tight_layout()
        plt.savefig(os.path.join(root_folder, "times_used.png"))
        plt.close()

        plt.figure(figsize=(10,4))
        plt.bar(df_summary["Genetic_operators"], df_summary["Pareto_Appearances"])
        plt.xticks(rotation=45, ha='right')
        plt.title("Pareto Appearances per Operator")
        plt.tight_layout()
        plt.savefig(os.path.join(root_folder, "pareto_appearances.png"))
        plt.close()

        print("Plots saved as:")
        print(os.path.join(root_folder, "times_used.png"))
        print(os.path.join(root_folder, "pareto_appearances.png"))

        return df_summary

    def save_operator_stats_csv(self):
        #Generate a CSV with the usage count and pareto appearance count of each genetic operator, as well as bar plots for each metric.
        exec_id = self.tecnas.exec if self.tecnas is not None else 0
        output_dir = os.path.join("Pareto_Fronts", f"Exec_{exec_id}")
        os.makedirs(output_dir, exist_ok=True)
        operator_usage = {op_name: self.genetic_operators_obj_dict[op_name].usage_count for op_name in self.genetic_operators_obj_dict.keys()}
        pareto_appearance = {op_name: self.genetic_operators_obj_dict[op_name].pareto_appearance_count for op_name in self.genetic_operators_obj_dict.keys()}
        df = pd.DataFrame({"Genetic_operators": list(operator_usage.keys()), "Times_Used": list(operator_usage.values()), "Pareto_Appearances": [pareto_appearance[op] for op in operator_usage.keys()]})
        filename = os.path.join(output_dir, "operator_stats.csv")
        df.to_csv(filename, index=False)

        plt.figure(figsize=(10, 5))
        plt.bar(operator_usage.keys(), operator_usage.values())
        plt.ylabel("Times Used")
        plt.xlabel("Genetic Operator")
        plt.title(f"Genetic Operator Usage - Exec {exec_id}")
        plt.xticks(rotation=45)
        barfile1 = os.path.join(output_dir, "operator_usage_barplot.png")
        plt.tight_layout()
        plt.savefig(barfile1, dpi=300)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(pareto_appearance.keys(), pareto_appearance.values(), color="orange")
        plt.ylabel("Pareto Appearances")
        plt.xlabel("Genetic Operator")
        plt.title(f"Genetic Operator Pareto Appearances - Exec {exec_id}")
        plt.xticks(rotation=45)

        barfile2 = os.path.join(output_dir, "pareto_operator_appearance_barplot.png")
        plt.tight_layout()
        plt.savefig(barfile2, dpi=300)
        plt.close()

        print(f"[NSGA-II] Combined CSV saved in: {filename}")
        print(f"[NSGA-II] Usage plot saved in: {barfile1}")
        print(f"[NSGA-II] Pareto plot saved in: {barfile2}")


    def plot_pareto_front_2D(self, F, pareto_ids, operator_names=None):
        print("[NSGA-II] Plotting Pareto Front...")
        F = np.array(F)
        plt.figure(figsize=(10, 7))
        plt.scatter(F[:,0], F[:,1], color="gray", alpha=0.5, label="All combinations")
        pf = F[pareto_ids]
        plt.scatter(pf[:,0], pf[:,1], color="red", s=80, label="Pareto Front")
        legend_entries = []
        for i, idx in enumerate(pareto_ids):
            x, y = F[idx]
            plt.text(x, y, str(i+1), fontsize=10, color="black")

            if operator_names is not None:
                legend_entries.append(f"{i+1} → {operator_names[idx]}")

        if operator_names is not None:
            legend_text = "\n".join(legend_entries)
            plt.figtext(0.75, 0.25, legend_text, fontsize=8, ha="left", va="bottom")

        labelsXY = [f"{optimization}_{metric}" for metric, optimization in self.objective_maxmin_names.items()]
        plt.xlabel(labelsXY[0])
        plt.ylabel(labelsXY[1])
        plt.title(f"Pareto front (2D) Exec {self.tecnas.exec}, Gen {self.tecnas.generation}")
        plt.legend()
        exec_id = self.tecnas.exec if self.tecnas is not None else 0
        output_dir = os.path.join("Pareto_Fronts", f"Exec_{exec_id}")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"pareto_front_2D_gen_{self.tecnas.generation}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[NSGA-II] 2D Plot saved in: {filename}")


    def plot_pareto_front_3D(self, F, pareto_ids, operator_names=None):
        #Plot the Pareto Front
        fig = plt.figure(figsize=(13, 7))
        ax = fig.add_subplot(111, projection="3d")
        F = np.array(F)
        #All points
        ax.scatter(F[:,0], F[:,1], F[:,2], c='gray', alpha=0.5, label="All combinations")
        #Pareto front points
        pf = F[pareto_ids]
        ax.scatter(pf[:,0], pf[:,1], pf[:,2], c='red', s=80, label="Pareto Front")

        #Labels (floating text)
        '''
        if operator_names is not None:
            for idx in pareto_ids:
                name = operator_names[idx]
                x, y, z = F[idx]
                ax.text(x, y, z, name, fontsize=8)
        
        '''
        legend_entries = []
        for i, idx in enumerate(pareto_ids):
            x, y, z = F[idx]
            ax.text(x, y, z, str(i+1), fontsize=10, color='black')
            if operator_names is not None:
                legend_entries.append(f"{i+1} → {operator_names[idx]}")
        if operator_names is not None:
            legend_text = "\n".join(legend_entries)
            plt.figtext(0.72, 0.25, legend_text, fontsize=8, ha="left", va="bottom")
            labelsXYZ = [f'{optimization}_{metric}' for metric, optimization in self.objective_maxmin_names.items()]
            self.objective_maxmin_names
            ax.set_xlabel(labelsXYZ[0])
            ax.set_ylabel(labelsXYZ[1])
            ax.set_zlabel(labelsXYZ[2])
            ax.set_title(f"Pareto front Exec {self.tecnas.exec}, Gen {self.tecnas.generation}")
            plt.legend()
        #Save in different folders
        exec_id = self.tecnas.exec if self.tecnas is not None else 0
        output_dir = os.path.join("Pareto_Fronts", f"Exec_{exec_id}")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"pareto_front_gen_{self.tecnas.generation}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[NSGA-II] Plot saved in: {filename}")

         
    def _create_problem(self):
        #Objective vectors for pymoo
        objs = []
        objs_per_operator = []
        #Create a matrix. Rows: genetic operators, Columns: metrics. Multiply by -1 if the objective is MAX
        for name in self.operator_names: #['SPC_MPAR', 'SPC_MSWAP', ...]
            objs_per_operator = []
            for metric in self.metric_names: #['Accuracy', 'FLOPs', 'Num_params']
                metric_value = self.genetic_operators_obj_dict[name].best_children_metrics[metric]
                metric_value = -metric_value if self.objective_maxmin_names[metric] == "MAX" else metric_value #-1 to convert MAX to MIN
                objs_per_operator.append(metric_value)
            objs.append(objs_per_operator)

        F = np.array(objs)
        n_objectives = len(self.metric_names)
        operator_names = self.operator_names
        class OperatorSelectionProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=1, n_obj=n_objectives, n_constr=0, xl=0, xu=len(operator_names) - 1, type_var=int)
            def _evaluate(self, x, out, *args, **kwargs):
                idx = int(x[0])
                out["F"] = F[idx]
        return OperatorSelectionProblem(), F

    def run_nsga2(self, n_select=1):
        #Returns the name of the selected genetic operator from the Pareto front
        self.operator_names = list(self.genetic_operators_obj_dict.keys())
        problem, F = self._create_problem()
        #Initial population: all combinations
        pop_size = len(self.operator_names)
        init_pop = np.array([[i] for i in range(pop_size)])
        algorithm = NSGA2(pop_size=pop_size, sampling=init_pop, eliminate_duplicates=True)
        termination = get_termination("n_gen", 1)
        result = minimize(problem, algorithm, termination, verbose=False)
        #Indexes of Pareto invdividuals
        pareto_ids = result.opt.get("X")
        pareto_ids = pareto_ids.flatten().tolist()
        #Calculate genetic operator Pareto Front appearance and fill a dictionary with their objects
        pareto_objects_dict = {}
        for idx in pareto_ids:
            op_name = self.operator_names[idx]
            self.genetic_operators_obj_dict[op_name].pareto_appearance_count += 1
            pareto_objects_dict[op_name] = self.genetic_operators_obj_dict[op_name]

        #Selection
        if self.NSGA_II_SELECTION[0] == 'CROWD':
            selected = self.operator_names[pareto_ids[0]] #Always take the first one
        elif self.NSGA_II_SELECTION[0] == 'RAND':
            selected = self.operator_names[random.sample(pareto_ids, 1)[0]]
        else:
            sorted_opNames_list = self.sort_operators_by(pareto_objects_dict, self.NSGA_II_SELECTION[0], descending = self.NSGA_II_SELECTION[1])
            selected = sorted_opNames_list[0] 
      
        print("[NSGA-II PYMOO] Pareto:", [self.operator_names[i] for i in pareto_ids])
        print("[NSGA-II PYMOO] Selected:", selected)
        self.genetic_operators_obj_dict[selected].usage_count += 1

        if self.tecnas.generation == 1 or self.tecnas.generation == config_tecnas.GENERATIONS or  self.tecnas.generation % 5  == 0:  
            if len(self.metric_names) == 3:
                self.plot_pareto_front_3D(F, pareto_ids, self.operator_names)
            elif len(self.metric_names) == 2:
                self.plot_pareto_front_2D(F, pareto_ids, self.operator_names)
            else:
                print("[NSGA-II PYMOO] Cannot plot Pareto front: only 2D and 3D supported.")

        self.tecnas.nsga2_logger.log_generation_to_csv(exec_id=self.tecnas.exec,generation=self.tecnas.generation,operator_names=self.operator_names,F=F,pareto_ids=pareto_ids,selected_ops=selected)
        return selected
        

