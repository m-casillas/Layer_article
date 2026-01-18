import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import glob


class TECNAS_NSGAII_LOGGER:

    def __init__(self, base_dir="Pareto_Fronts"):
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir

    def log_generation_to_csv(self, exec_id, generation, operator_names, F, pareto_ids, selected_ops):
        output_dir = os.path.join(self.base_dir, f"Exec_{exec_id}")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "full_log.csv")

        n_obj = F.shape[1]
        data = []

        for i, op in enumerate(operator_names):
            row = {
                "exec": exec_id,
                "generation": generation,
                "operator": op,
                "is_pareto": 1 if i in pareto_ids else 0,
                "is_selected": 1 if op in selected_ops else 0
            }
            for j in range(n_obj):
                row[f"f{j+1}"] = F[i][j]
            data.append(row)

        df = pd.DataFrame(data)

        if not os.path.exists(filename):
            df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, index=False, mode="a", header=False)

    def load_csv_log(self, exec_id):
        filename = os.path.join(self.base_dir, f"Exec_{exec_id}", "full_log.csv")
        return pd.read_csv(filename)

    def plot_pareto_front_rebuilt(self, exec_id, generation, F, pareto_ids, operator_names, output_path):
        n_obj = F.shape[1]

        if n_obj == 2:
            plt.figure(figsize=(8,6))
            plt.scatter(F[:,0], F[:,1], c="gray", alpha=0.5)
            pf = F[pareto_ids]
            plt.scatter(pf[:,0], pf[:,1], c="red")
            for i, idx in enumerate(pareto_ids):
                x, y = F[idx]
                plt.text(x, y, str(i+1))
            plt.title(f"Pareto Front Exec {exec_id} Gen {generation}")
            plt.xlabel("f1")
            plt.ylabel("f2")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            return

        if n_obj == 3:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(F[:,0], F[:,1], F[:,2], c='gray', alpha=0.5)
            pf = F[pareto_ids]
            ax.scatter(pf[:,0], pf[:,1], pf[:,2], c='red', s=80)
            legend_entries = []
            for i, idx in enumerate(pareto_ids):
                x, y, z = F[idx]
                ax.text(x, y, z, str(i+1), fontsize=10, color='black')
                legend_entries.append(f"{i+1} → {operator_names[idx]}")
            legend_text = "\n".join(legend_entries)
            plt.figtext(0.02, 0.02, legend_text, fontsize=10,
                        ha="left", va="bottom")
            ax.set_xlabel("f1")
            ax.set_ylabel("f2")
            ax.set_zlabel("f3")
            ax.set_title(f"Pareto front Exec {exec_id}, Gen {generation}")
            plt.legend()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            return

        print(f"No se puede graficar un Pareto de {n_obj} objetivos.")

    def replot_pareto_from_csv(self, csv_fullpath):
        df = pd.read_csv(csv_fullpath)
        exec_id = int(df["exec"].unique()[0])
        generations = sorted(df["generation"].unique())
        out_dir = os.path.dirname(csv_fullpath)

        for generation in generations:
            dfg = df[df["generation"] == generation].reset_index(drop=True)
            obj_cols = [c for c in dfg.columns if c.startswith("f")]
            F = dfg[obj_cols].values
            pareto_ids = dfg.index[dfg["is_pareto"] == 1].tolist()
            operator_names = dfg["operator"].tolist()
            output = os.path.join(out_dir, f"replot_gen_{generation}.png")

            self.plot_pareto_front_rebuilt(
                exec_id=exec_id,
                generation=generation,
                F=F,
                pareto_ids=pareto_ids,
                operator_names=operator_names,
                output_path=output
            )

            print(f"Rebuilt plot saved: {output}")
