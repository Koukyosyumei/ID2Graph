import numpy as np
import argparse
import os
import networkx as nx
from matplotlib import pyplot as plt
import glob
from sklearn.cluster import KMeans
from community import community_louvain
from sklearn import preprocessing
import matplotlib.cm as cm
from sklearn import metrics


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

            start_line_num_of_active_party = 3 + int(lines[1][:-1])
            X_train = np.array(
                [
                    lines[col_idx][:-1].split(" ")
                    for col_idx in range(
                        start_line_num_of_active_party,
                        start_line_num_of_active_party
                        + int(lines[start_line_num_of_active_party - 1][:-1]),
                    )
                ]
            )
            min_max_scaler = preprocessing.MinMaxScaler()
            X_train_minmax = min_max_scaler.fit_transform(X_train.T)

            y_train = lines[num_col + num_party + 1].split(" ")
            y_train = [int(y) for y in y_train]

        path_to_adj_mat_file = os.path.join(
            parsed_args.path_to_dir, f"{round_idx}_adj_mat.txt"
        )
        with open(path_to_adj_mat_file, mode="r") as f:
            lines = f.readlines()

            round_num = int(lines[0])
            node_num = int(lines[1])
            adj_mat = np.zeros((node_num, node_num))

            for i in range(round_num):
                for j in range(node_num):
                    temp_row = lines[i * node_num + 2 + j].split(" ")[1:-1]
                    for k in temp_row:
                        adj_mat[j][int(k)] += 1
                        adj_mat[int(k)][j] += 1

        kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train_minmax)
        baseline_roc_auc_score = metrics.roc_auc_score(y_train, kmeans.labels_)
        baseline_roc_auc_score = max(1 - baseline_roc_auc_score, baseline_roc_auc_score)
        print("baseline: ", baseline_roc_auc_score)

        print("creating a graph ...")
        G = nx.from_numpy_matrix(
            adj_mat, create_using=nx.MultiGraph, parallel_edges=False
        )
        print("detecting communities ...")
        partition = community_louvain.best_partition(G)
        com_labels = list(partition.values())
        com_num = len(list(set(com_labels)))
        X_com = np.zeros((X_train.shape[1], com_num))
        for i, j in enumerate(com_labels):
            X_com[i, j] = 1

        kmeans_only_com = KMeans(n_clusters=2, random_state=0).fit(X_com)
        onlycom_roc_auc_score = metrics.roc_auc_score(y_train, kmeans_only_com.labels_)
        onlycom_roc_auc_score = max(1 - onlycom_roc_auc_score, onlycom_roc_auc_score)
        print("only community: ", onlycom_roc_auc_score)

        kmeans_with_com = KMeans(n_clusters=2, random_state=0).fit(
            np.hstack([X_train_minmax, X_com])
        )
        withcom_roc_auc_score = metrics.roc_auc_score(y_train, kmeans_with_com.labels_)
        withcom_roc_auc_score = max(1 - withcom_roc_auc_score, withcom_roc_auc_score)
        print("with community: ", withcom_roc_auc_score)

        print("saving a graph ...")
        plt.style.use("ggplot")
        pos = nx.spring_layout(G)
        cmap = cm.get_cmap("cool", max(partition.values()) + 1)
        nx.draw_networkx(
            G,
            pos,
            with_labels=False,
            alpha=0.3,
            node_size=60,
            linewidths=0.1,
            width=0.1,
            cmap=cmap,
            node_color=list(partition.values()),
        )
        plt.savefig(path_to_adj_mat_file.split(".")[0] + "_plot.png")
