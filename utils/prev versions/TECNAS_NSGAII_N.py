from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


class TECNAS_NSGAII:
    def __init__(self, tecnas_obj=None, verbose=True, objective_list=None):
        self.tecnas = tecnas_obj
        self.verbose = verbose
        self.operator_usage = {}
        self.pareto_operator_appearance = {}
        self.objective_list = objective_list

    def _extract_accuracy(self, m):
        for k in ['best_acc', 'Acc_mean', 'acc_mean', 'accuracy']:
            if k in m:
                acc = m[k]
                if acc > 1: acc /= 100
                return float(np.clip(acc, 0.0, 1.0))
        return 0.0

    def _extract_flops(self, m):
        for k in ['best_flops', 'FLOPs_mean', 'flops']:
            if k in m and not np.isnan(m[k]):
                return float(m[k])
        return 1e20

    def _extract_params(self, m):
        for k in ['best_num_params', 'Num_Params', 'num_params']:
            if k in m and not np.isnan(m[k]):
                return float(m[k])
        return 1e20

    def _create_problem(self, metrics_dict, names):
        F = []
        for name in names:
            metric_dict = metrics_dict[name]
            values = [obj(metric_dict) for obj in self.objective_list]
            F.append(values)
        F = np.array(F)

        n_obj = len(self.objective_list)
        class OperatorSelectionProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=1, n_obj=n_obj, n_constr=0,
                                 xl=0, xu=len(names) - 1, type_var=int)

            def _evaluate(self, x, out, *args, **kwargs):
                idx = int(x[0])
                out["F"] = F[idx]

        return OperatorSelectionProblem(), F

    def plot_pareto_front(self, F, pareto_ids, operator_names):
        n_obj = F.shape[1]

        if n_obj == 2:
            self._plot_2d(F, pareto_ids, operator_names)
        elif n_obj == 3:
            self._plot_3d(F, pareto_ids, operator_names)
        else:
            print(f"[NOTE] Non-available plot with {n_obj} objectives.")
            return


    def _plot_2d(self, F, pareto_ids, operator_names):
        plt.figure(figsize=(8,6))
        plt.scatter(F[:,0], F[:,1], c="gray", alpha=0.5)
        pf = F[pareto_ids]
        plt.scatter(pf[:,0], pf[:,1], c="red")
        for i, idx in enumerate(pareto_ids):
            x, y = F[idx]
            plt.text(x, y, str(i+1))
        plt.title("Pareto Front (2D)")
        plt.xlabel("Obj 1")
        plt.ylabel("Obj 2")
        plt.tight_layout()
        plt.show()


    def _plot_3d(self, F, pareto_ids, operator_names):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(F[:,0], F[:,1], F[:,2], c="gray", alpha=0.5)
        pf = F[pareto_ids]
        ax.scatter(pf[:,0], pf[:,1], pf[:,2], c='red', s=60)
        for i, idx in enumerate(pareto_ids):
            x,y,z = F[idx]
            ax.text(x, y, z, str(i+1))

        ax.set_title("Pareto Front (3D)")
        ax.set_xlabel("Obj 1")
        ax.set_ylabel("Obj 2")
        ax.set_zlabel("Obj 3")
        plt.show()


    def run_nsga2(self, all_children_metrics=None, n_select=1):

        all_children_metrics = self.tecnas.all_children_metrics
        operator_names = list(all_children_metrics.keys())

        problem, F = self._create_problem(all_children_metrics, operator_names)

        pop_size = len(operator_names)
        init_pop = np.array([[i] for i in range(pop_size)])

        algorithm = NSGA2(pop_size=pop_size, sampling=init_pop,
                          eliminate_duplicates=True)
        termination = get_termination("n_gen", 1)

        result = minimize(problem, algorithm, termination, verbose=False)

        pareto_ids = result.opt.get("X").flatten().tolist()

        for idx in pareto_ids:
            name = operator_names[idx]
            self.pareto_operator_appearance[name] = self.pareto_operator_appearance.get(name, 0) + 1
        
        for idx in pareto_ids:
            if operator_names[idx] == 'UC_MSWAP': #This has the best succesful crossover rate.
                selected  = operator_names[idx]
                break
            else:
                selected = operator_names[pareto_ids[0]]

        selected = operator_names[pareto_ids[0]]

        print("Pareto:", [operator_names[i] for i in pareto_ids])
        print("Selected:", selected)

        self.plot_pareto_front(F, pareto_ids, operator_names)

        return selected
