import numpy as np
import argparse
import os
import networkx as nx
from matplotlib import pyplot as plt


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
    path_to_adj_mat_file = os.path.join(parsed_args.path_to_dir, "adj_mat.txt")

    with open(path_to_adj_mat_file, mode="r") as f:
        lines = f.readlines()
        round_num = lines[0]
        line_idx = 1
        list_adj_mat = []
        for i in range(int(round_num)):
            dim = int(lines[line_idx])
            line_idx += 1
            temp_adj_mat = []
            for j in range(dim):
                temp_row = lines[line_idx].split(" ")[:-1]
                temp_adj_mat.append(temp_row)
                line_idx += 1
            list_adj_mat.append(np.array(temp_adj_mat).astype(int))

    G = nx.from_numpy_matrix(sum(list_adj_mat[1:5]))
    nx.draw_networkx(
        G, with_labels=False, alpha=0.3, node_size=60, linewidths=0.1, width=0.1
    )
    plt.savefig(os.path.join(parsed_args.path_to_dir, "adj_mat_plot.png"))
