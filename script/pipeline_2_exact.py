import argparse
import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pyclustering.cluster.xmeans import xmeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from scipy import sparse

from llatvfl.clustering import get_f_p_r, SSEMeans

N_INIT = 10


def add_args(parser):
    parser.add_argument(
        "-p",
        "--path_to_input_file",
        type=str,
    )
    parser.add_argument(
        "-q",
        "--path_to_com_file",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
    )
    parser.add_argument(
        "-v",
        "--clustering_type",
        type=str,
    )
    parser.add_argument(
        "-k",
        "--weight_for_community_variables",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    clustering_cls = KMeans

    print(
        "baseline_c,baseline_h,baseline_v,baseline_p,baseline_ip,baseline_f,our_c,our_h,our_v,our_p,our_ip,our_f"
    )

    with open(parsed_args.path_to_input_file, mode="r") as f:
        lines = f.readlines()
        first_line = lines[0].split(" ")
        num_classes, num_row, num_col, num_party = (
            int(first_line[0]),
            int(first_line[1]),
            int(first_line[2]),
            int(first_line[3]),
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
        y_train = np.array([int(y) for y in y_train])
        unique_labels = np.unique(y_train)

    xm = xmeans(data=X_train_minmax, tolerance=0.0001)
    xm.process()
    baseline_labels = xm.predict(X_train_minmax)

    with open(parsed_args.path_to_com_file, mode="r") as f:
        lines = f.readlines()
        comm_num = int(lines[0])
        node_num = int(lines[1])
        X_com = [0 for _ in range(num_row)]

        for i in range(comm_num):
            temp_nodes_in_comm = lines[i + 2].split(" ")[:-1]
            for k in temp_nodes_in_comm:
                X_com[int(k)] = i

    with_com_labels = X_com

    random.seed(parsed_args.seed)
    num_train = X_train_minmax.shape[0]
    num_train_aux = int(num_train * 0.05)
    public_idxs = random.sample(list(range(num_train)), num_train_aux)
    private_idxs = list(set(list(range(num_train))) - set(public_idxs))

    clf_baseline = RandomForestClassifier(random_state=parsed_args.seed)
    clf_baseline.fit(X_train_minmax[public_idxs], y_train[public_idxs])
    y_pred_baseline = clf_baseline.predict(X_train_minmax[private_idxs])
    f1_baseline = f1_score(y_pred_baseline, y_train[private_idxs], average="micro")

    clf_cluster = RandomForestClassifier(random_state=parsed_args.seed)
    clf_cluster.fit(
        np.hstack([X_train_minmax, np.array(baseline_labels).reshape(-1, 1)])[public_idxs],
        y_train[public_idxs],
    )
    y_pred_cluster = clf_cluster.predict(
        np.hstack([X_train_minmax, np.array(baseline_labels).reshape(-1, 1)])[private_idxs]
    )
    f1_cluster = f1_score(y_pred_cluster, y_train[private_idxs], average="micro")

    clf_graph = RandomForestClassifier(random_state=parsed_args.seed)
    clf_graph.fit(
        np.hstack([X_train_minmax, np.array(with_com_labels).reshape(-1, 1)])[public_idxs],
        y_train[public_idxs],
    )
    y_pred_graph = clf_graph.predict(
        np.hstack([X_train_minmax, np.array(with_com_labels).reshape(-1, 1)])[private_idxs]
    )
    f1_graph = f1_score(y_pred_graph, y_train[private_idxs], average="micro")

    print(f"{f1_baseline},{f1_cluster},{f1_graph},0,0,0,0,0,0,0,0,0")
