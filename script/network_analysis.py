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

        kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train_minmax)
        baseline_roc_auc_score = metrics.roc_auc_score(y_train, kmeans.labels_)
        baseline_roc_auc_score = max(1 - baseline_roc_auc_score, baseline_roc_auc_score)
        print("baseline: ", baseline_roc_auc_score)

        print("creating a graph ...")
        G = nx.from_numpy_matrix(sum(list_adj_mat[0:2]))
        partition = community_louvain.best_partition(G)
        com_labels = list(partition.values())
        com_num = len(list(set(com_labels)))
        X_com = np.zeros((X_train.shape[1], com_num))
        for i, j in enumerate(com_labels):
            X_com[i, j] = 1
        kmeans_with_com = KMeans(n_clusters=2, random_state=0).fit(
            np.hstack([X_train_minmax, X_com])
        )
        withcom_roc_auc_score = metrics.roc_auc_score(y_train, kmeans_with_com.labels_)
        withcom_roc_auc_score = max(1 - withcom_roc_auc_score, withcom_roc_auc_score)
        print("with community: ", withcom_roc_auc_score)
        """
        cmap = cm.get_cmap("viridis", max(partition.values()) + 1)
        print("saving a graph ...")
        nx.draw_networkx(
            G,
            with_labels=False,
            alpha=0.3,
            node_size=60,
            linewidths=0.1,
            width=0.1,
            cmap=cmap,
            node_color=list(partition.values()),
        )
        plt.savefig(path_to_adj_mat_file.split(".")[0] + "_plot.png")
        """
