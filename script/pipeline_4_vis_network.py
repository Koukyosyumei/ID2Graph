import numpy as np
import argparse
import os
import networkx as nx
from matplotlib import pyplot as plt
import glob


def add_args(parser):
    parser.add_argument(
        "-p",
        "--path_to_dir",
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)
    list_adj_files = glob.glob(os.path.join(parsed_args.path_to_dir, "*adj_mat.txt"))

    for path_to_adj_file in list_adj_files:
        G = nx.Graph()
        with open(path_to_adj_file, mode="r") as f:
            lines = f.readlines()
            round_num = int(lines[0])
            node_num = int(lines[1])

            for i in range(round_num):
                for j in range(node_num):
                    temp_row = lines[i * node_num + 2 + j].split(" ")[:-1]
                    temp_adj_num = int(temp_row[0])
                    for k in range(temp_adj_num):
                        G.add_edge(
                            j, int(temp_row[2 * k]), weight=float(temp_row[2 * k + 1])
                        )

            adj_mat = np.zeros((node_num, node_num))

            for i in range(round_num):
                for j in range(node_num):
                    temp_row = lines[i * node_num + 2 + j].split(" ")[1:-1]
                    for k in temp_row:
                        adj_mat[j][int(k)] += 1
                        adj_mat[int(k)][j] += 1

        plt.style.use("ggplot")
        pos = nx.spring_layout(G)
        # cmap = cm.get_cmap("cool", max(partition.values()) + 1)
        nx.draw_networkx(
            G,
            pos,
            with_labels=False,
            alpha=0.3,
            node_size=60,
            linewidths=0.1,
            width=0.1,
            # cmap=cmap,
            # node_color=list(partition.values()),
        )
        plt.savefig(path_to_adj_file.split(".")[0] + "_plot.png")
