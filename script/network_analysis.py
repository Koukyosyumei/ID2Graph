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
    list_input_files = glob.glob(os.path.join(parsed_args.path_to_dir, "*.in"))
    print(list_input_files)
    for path_to_input_file in list_input_files:
        round_idx = path_to_input_file.split("_")[-1].split(".")[0]
        with open(path_to_input_file, mode="r") as f:
            lines = f.readlines()
            first_line = lines[0].split(" ")
            num_row, num_col, num_party = (
                int(first_line[0]),
                int(first_line[1]),
                int(first_line[2]),
            )
            y_train = lines[num_col + num_party + 1].split(" ")
            y_train = [int(y) for y in y_train]

        path_to_adj_mat_file = os.path.join(
            parsed_args.path_to_dir, f"{round_idx}_adj_mat.txt"
        )
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

        G = nx.from_numpy_matrix(sum(list_adj_mat[0:2]))
        nx.draw_networkx(
            G,
            with_labels=False,
            alpha=0.3,
            node_size=60,
            linewidths=0.1,
            width=0.1,
            node_color=y_train,
        )
        plt.savefig(path_to_adj_mat_file.split(".")[0] + "_plot.png")
