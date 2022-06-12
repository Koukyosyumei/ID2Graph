import numpy as np
import argparse
import os
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import cm
import glob


def add_args(parser):
    parser.add_argument(
        "-e",
        "--edge_weight",
        type=float,
    )
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
        with open(path_to_adj_file, mode="r") as f:
            lines = f.readlines()
            round_num = int(lines[0])
            node_num = int(lines[1])

            adj_mat = np.zeros((node_num, node_num))

            for i in range(round_num):
                for j in range(node_num):
                    temp_row = lines[i * node_num + 2 + j].split(" ")[:-1]
                    temp_adj_num = int(temp_row[0])
                    for k in range(temp_adj_num):
                        adj_mat[
                            j, int(temp_row[2 * k + 1])
                        ] += parsed_args.edge_weight * float(temp_row[2 * (k + 1)])

        G = nx.from_numpy_matrix(
            adj_mat, create_using=nx.MultiGraph, parallel_edges=False
        )

        path_to_community_file = path_to_adj_file.split("_")[0] + "_communities.out"
        with open(path_to_community_file, mode="r") as f:
            lines = f.readlines()
            comm_num = int(lines[0])
            node_num = int(lines[1])
            node2comm = np.zeros(node_num)
            for i in range(comm_num):
                comm_list = lines[2 + i].split(" ")[:-1]
                for v in comm_list:
                    node2comm[int(v)] = i

        path_to_input_file = path_to_adj_file.split("_")[0] + "_data.in"
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

        edge_in_list = []
        edge_ex_list = []
        edge_color_in_list = []
        edge_color_ex_list = []
        for j in range(node_num):
            for k in range(node_num):
                if adj_mat[j][k] != 0:
                    if node2comm[j] != node2comm[k]:
                        edge_ex_list.append((j, k))
                        edge_color_ex_list.append("k")
                    else:
                        edge_in_list.append((j, k))
                        edge_color_in_list.append(node2comm[j] + 3)

        cmap = cm.get_cmap("plasma", comm_num + 3)

        plt.style.use("ggplot")
        pos = nx.spring_layout(G)
        nx.draw_networkx(
            G,
            pos,
            with_labels=False,
            alpha=0.2,
            node_size=0,
            width=0.2,
            edgelist=edge_ex_list,
            edge_color=edge_color_ex_list,
        )
        nx.draw_networkx(
            G,
            pos,
            with_labels=False,
            alpha=0.2,
            node_size=0,
            edgelist=edge_in_list,
            cmap=cmap,
            edge_color=edge_color_in_list,
        )
        nx.draw_networkx(
            G,
            pos,
            with_labels=False,
            alpha=1.0,
            node_size=10,
            edgelist=[],
            edge_color=[],
            node_color=["g" if y == 0 else "r" for y in y_train],
        )
        plt.savefig(path_to_adj_file.split(".")[0] + "_plot.pdf")
