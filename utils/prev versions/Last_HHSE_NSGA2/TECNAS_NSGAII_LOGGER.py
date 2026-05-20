import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines
import glob


class TECNAS_NSGAII_LOGGER:

    def __init__(self, base_dir="Pareto_Fronts"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def log_generation_to_csv(self, exec_id, generation, operator_names, F, pareto_ids, selected_ops):
        """
        Logs any number of objectives automatically: f0, f1, f2, ..., fk
        """
        output_dir = os.path.join(self.base_dir, f"Exec_{exec_id}")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "full_log.csv")
        n_obj = F.shape[1]
        data = []
        for i, op in enumerate(operator_names):
            row = {"exec": exec_id, "generation": generation,"operator": op,"is_pareto": 1 if i in pareto_ids else 0,"is_selected": 1 if op in selected_ops else 0 }
            # dynamically add f0, f1, ... fN
            for j in range(n_obj):
                row[f"f{j}"] = F[i][j]
            data.append(row)
        df = pd.DataFrame(data)
        # Append or write new
        if not os.path.exists(filename):
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, index=False, mode="a", header=False)

    def load_csv_log(self, exec_id):
        filename = os.path.join(self.base_dir, f"Exec_{exec_id}", "full_log.csv")
        return pd.read_csv(filename)

    def _plot_pareto_2D(self, F, pareto_ids, operator_names, title, output_path):
        plt.figure(figsize=(8, 6))
        plt.scatter(F[:, 0], F[:, 1], color="gray", alpha=0.5, label="All combinations")
        plt.scatter(F[pareto_ids, 0], F[pareto_ids, 1], color="red", s=60, label="Pareto Front")

        for i, idx in enumerate(pareto_ids):
            plt.text(F[idx, 0], F[idx, 1], str(i + 1), fontsize=9)
        
        plt.xlabel("f0")
        plt.ylabel("f1")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def _plot_pareto_3D(self, F, pareto_ids, operator_names, title, output_path):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(F[:, 0], F[:, 1], F[:, 2], c="gray", alpha=0.4, label="All combinations")
        ax.scatter(F[pareto_ids, 0], F[pareto_ids, 1], F[pareto_ids, 2],
                   c="red", s=60, label="Pareto Front")

        for i, idx in enumerate(pareto_ids):
            ax.text(F[idx, 0], F[idx, 1], F[idx, 2], str(i + 1), fontsize=9)

        ax.set_xlabel("f0")
        ax.set_ylabel("f1")
        ax.set_zlabel("f2")
        ax.set_title(title)

        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

  

    def plot_pareto_front_rebuilt(self, exec_id, generation, F, pareto_ids, operator_names, output_path):

        n_obj = F.shape[1]
        title = f"Pareto front Exec {exec_id}, Gen {generation}"

        if n_obj == 2:
            self._plot_pareto_2D(F, pareto_ids, operator_names, title, output_path)
        elif n_obj == 3:
            self._plot_pareto_3D(F, pareto_ids, operator_names, title, output_path)
        else:
            # 4 or more objectives
            self._plot_parallel_coordinates(F, pareto_ids, operator_names, title, output_path)

  
    def replot_pareto_from_csv(self, csv_fullpath):
        df = pd.read_csv(csv_fullpath)

        exec_id = int(df["exec"].unique()[0])
        generations = sorted(df["generation"].unique())

        out_dir = os.path.dirname(csv_fullpath)

        # detect objective columns f0, f1, f2, ...
        obj_cols = sorted([col for col in df.columns if col.startswith("f")],
                          key=lambda x: int(x[1:]))

        for gen in generations:
            dfg = df[df["generation"] == gen].reset_index(drop=True)

            F = dfg[obj_cols].values
            pareto_ids = dfg.index[dfg["is_pareto"] == 1].tolist()
            operator_names = dfg["operator"].tolist()

            output_path = os.path.join(out_dir, f"replot_gen_{gen}.png")

            self.plot_pareto_front_rebuilt(
                exec_id=exec_id,
                generation=gen,
                F=F,
                pareto_ids=pareto_ids,
                operator_names=operator_names,
                output_path=output_path
            )

            print(f"[LOGGER] Rebuilt plot saved: {output_path}")
