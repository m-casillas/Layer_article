import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # backend sin GUI
import numpy as np
import random
import os
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.survival.rank_and_crowding.metrics import calc_crowding_distance
from pymoo.indicators.hv import HV
import time
import config_tecnas


class TECNAS_NSGAII:
    def __init__(self, pop = None, pop_size = None, tecnasObj = None):
        self.pop = pop
        self.pop_size = pop_size
        self.tecnas = tecnasObj
        self.evaluate(self.pop)
        
    def evaluate(self, archs):
        print('\nNSGA-II: Evaluating FLOPs and Num_params for architectures in the population...')
        for i,arch in enumerate(archs):
            if arch.flops > 0:
                #print(f'{arch.idx} already has flops and num_params calculated')
                ...
            else:
                arch.flops, arch.num_params, arch.sizeMB = self.tecnas.flops_params_gen_list_str(get_realMetrics = False, gen_list_str = str(arch.genotype.gen_list))
            print(f'FLOPs calculation progress {i+1}/{len(archs)}') if (i+1) % 100 == 0 or i == len(archs)-1 else None
        print('NSGA-II: Evaluating FLOPs and Num_params complete\n')

    def _build_objective_matrix(self, archs):
        return np.column_stack([
            -np.array([a.acc for a in archs]),
             np.array([a.flops for a in archs]),
             np.array([a.num_params for a in archs])
        ])

    def normalize_objectives(self, raw_objectives):
        mins = np.min(raw_objectives, axis=0)
        maxs = np.max(raw_objectives, axis=0)
        # Avoids division by zero if all values are the same
        ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
        normalized = (raw_objectives - mins) / ranges
        return normalized#, mins, maxs, ranges

    def hypervolume(self, F, fronts, ref_point=None):
        if len(fronts[0]) == 0:
            return 0.0
        
        F_normalized = self.normalize_objectives(F)
        pareto_front = F_normalized[fronts[0]]
        
        # Reference point. Slighty worse than the worst
        if ref_point is None:
            ref_point = np.array([1.01, 1.01, 1.01])
      
        ind = HV(ref_point=ref_point)
        hv = ind.do(pareto_front)
        print(f"[NSGA-II] Hypervolume (normalized objectives): {hv:.4f}")
        return hv

    def assign_rank_and_crowding(self):
        F = self._build_objective_matrix(self.pop)
        fronts = NonDominatedSorting().do(F)

        for i, front in enumerate(fronts):
            if len(front) == 0:
                continue
            cd = calc_crowding_distance(F[front])
            for j, idx in enumerate(front):
                ind = self.pop[idx]
                ind.rank = i
                ind.crowding_distance = cd[j]


    def tournament_selection(self):
        a, b = np.random.choice(self.pop, size=2, replace=False)
        if a.rank < b.rank:
            return a
        if b.rank < a.rank:
            return b
        if a.crowding_distance > b.crowding_distance:
            return a
        return b

    def select_parent_pairs(self):
        print('NSGA-II: Selecting parent pairs using tournament selection based on rank and crowding distance...')
        self.assign_rank_and_crowding()
        pairs = []
        for i in range(len(self.pop) // 2):
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            while p1 is p2:
                p2 = self.tournament_selection()
            pairs.append((p1, p2))
            print(f'Parents {i+1}/{len(self.pop)//2} selected') if (i+1) % 100 == 0 or i == (len(self.pop)//2)-1 else None
        print('NSGA-II: Parent pair selection complete.\n')
        return pairs
   

    def select_and_HV(self, archs):
        """
        Select pop_size architectures using Pareto fronts + crowding distance.
        """
        F = self._build_objective_matrix(archs)

        print('Non dominated sorting')
        nds = NonDominatedSorting()
        fronts = nds.do(F)
        print('Non dominated sorting complete')

        print('Calculating HyperVolume')
        HV = self.hypervolume(F, fronts, ref_point=None)
        print('Calculating HyperVolume')

        selected_indices = []

        for front in fronts:
            if len(selected_indices) + len(front) <= self.pop_size:
                selected_indices.extend(front.tolist())
            else:
                remaining = self.pop_size - len(selected_indices)
                crowding = calc_crowding_distance(F[front])
                order = np.argsort(-crowding)
                selected_indices.extend(front[order[:remaining]].tolist())
                break

        
        if config_tecnas.PLOT_PARETO and (self.tecnas.generation == 1 or self.tecnas.generation == config_tecnas.GENERATIONS or  self.tecnas.generation % 5  == 0):  
            self.plot_pareto_front_3D(F, fronts)

        return [archs[i] for i in selected_indices], HV

    def plot_pareto_front_3D2(self, F, fronts):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        F = np.array(F)
        # Make accuracy positive (assuming F[:, 0] = -Accuracy)
        F_plot = F.copy()
        F_plot[:, 0] = -F_plot[:, 0]

        total_points = len(F_plot)
        pareto_points = len(fronts[0])

        #All points (background)
        ax.scatter(F_plot[:, 0], F_plot[:, 1],  F_plot[:, 2],  c='lightgray',  alpha=0.5, label=f"All architectures ({total_points})")

        #First Pareto front
        pf0_idx = fronts[0]
        pf0 = F_plot[pf0_idx]

        ax.scatter(pf0[:, 0], pf0[:, 1], pf0[:, 2], c='red', s = 70, alpha=0.9, label=f"Pareto Front ({pareto_points})")
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('FLOPs')
        ax.set_zlabel('NParams')
        ax.set_title(f"Pareto Front — Exec {self.tecnas.exec}, Gen {self.tecnas.generation}")
        ax.legend()
        #Rotate view
        ax.view_init(elev=20, azim=60)

        exec_id = self.tecnas.exec if self.tecnas is not None else 0
        output_dir = os.path.join("Pareto_Fronts", f"Exec_{exec_id}")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"pareto_front_gen_{self.tecnas.generation}.png")
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"[NSGA-II] Plot saved in: {filename}")

    def plot_pareto_front_3D(self, F, fronts):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        F = np.array(F)

        # Make accuracy positive (assuming F[:, 0] = -Accuracy)
        F_plot = F.copy()
        F_plot[:, 0] = -F_plot[:, 0]

        # Pareto front 0 indices
        pf0_idx = np.array(fronts[0])

        # Mask to exclude Pareto front points from background
        mask = np.ones(len(F_plot), dtype=bool)
        mask[pf0_idx] = False

        total_points = len(F_plot)
        pareto_points = len(pf0_idx)

        # Dominated points (background) — plotted once
        ax.scatter(
            F_plot[mask, 0],
            F_plot[mask, 1],
            F_plot[mask, 2],
            c='lightgray',
            alpha=0.5,
            label=f"Dominated ({mask.sum()})"
        )

        # Pareto front (front 0) — plotted once
        ax.scatter(
            F_plot[pf0_idx, 0],
            F_plot[pf0_idx, 1],
            F_plot[pf0_idx, 2],
            c='red',
            s=70,
            alpha=0.9,
            label=f"Pareto Front ({pareto_points})"
        )

        # Labels
        ax.set_xlabel('Accuracy (maximized)')
        ax.set_ylabel('FLOPs (minimized)')
        ax.set_zlabel('NParams (minimized)', labelpad=15)

        ax.set_title(f"Pareto Front — Exec {self.tecnas.exec},Gen {self.tecnas.generation}\n{self.tecnas.crossover_type}_{self.tecnas.mutation_type} HV: {self.tecnas.HV:.4f}")

        ax.legend()

        # Rotate view
        ax.view_init(elev=25, azim=-45)

        # Output
        exec_id = self.tecnas.exec if self.tecnas is not None else 0
        op_type = f'{self.tecnas.crossover_type}_{self.tecnas.mutation_type}'
        op_type += '_NSGAII' if self.tecnas.NSGA2 else ''
        op_type = '_HHSE' if self.tecnas.HHSE else op_type

        output_dir = os.path.join("Pareto_Fronts", op_type, f"Exec_{exec_id}")
        os.makedirs(output_dir, exist_ok=True)

        filename = os.path.join(output_dir, f"pareto_front_gen_{self.tecnas.generation}.png" )
        plt.savefig(filename, dpi=100, pad_inches=0.2)
        plt.close()

        print(f"[NSGA-II] Plot saved in: {filename}")


