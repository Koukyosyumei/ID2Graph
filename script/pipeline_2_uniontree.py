import argparse

import numpy as np
import sklearn
from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from pyclustering.cluster.xmeans import xmeans

from llatvfl.clustering import get_f_p_r, SSEMeans


# from matplotlib import pyplot as plt
N_INIT = 10
label2maker = {0: "o", 1: "x"}


def add_args(parser):
    parser.add_argument(
        "-p",
        "--path_to_input_file",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
    )
    parser.add_argument(
        "-q",
        "--path_to_union_file",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

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

    path_to_adj_file = parsed_args.path_to_input_file[:-8] + "_adj_mat.txt"
    with open(path_to_adj_file, mode="r") as f:
        lines_adjmat = f.readlines()
        node_num = int(lines_adjmat[0])

        adj_mat = sparse.lil_matrix((num_row, num_row))
        for j in range(node_num):
            temp_row = lines_adjmat[1 + j].split(" ")[:-1]
            temp_adj_num = int(temp_row[0])
            for k in range(temp_adj_num):
                adj_mat[j, int(temp_row[2 * k + 1])] += float(temp_row[2 * (k + 1)])
                adj_mat[int(temp_row[2 * k + 1]), j] = adj_mat[
                    j, int(temp_row[2 * k + 1])
                ]

    with open(parsed_args.path_to_union_file, mode="r") as f:
        lines = f.readlines()
        union_clusters = lines[0].split(" ")[:-1]

    union_clusters = LabelEncoder().fit_transform(union_clusters)

    c_score_baseline = metrics.completeness_score(y_train, union_clusters)
    h_score_baseline = metrics.homogeneity_score(y_train, union_clusters)
    v_score_baseline = metrics.v_measure_score(y_train, union_clusters)

    f_score_baseline, p_score_baseline, ip_score_baseline = get_f_p_r(
        y_train, union_clusters
    )

    """
    num_union = len(np.unique(union_clusters))
    X_com = np.zeros((num_row, num_union))
    for i in range(num_row):
        X_com[i, union_clusters[i]] = 1

    kmeans_with_com = clustering_cls(
        n_clusters=num_classes, n_init=N_INIT, random_state=parsed_args.seed
    ).fit(np.hstack([X_train_minmax, X_com]))
    """

    if num_row > 10000:
        clf = sklearn.decomposition.TruncatedSVD(50)
        adj_mat = clf.fit_transform(adj_mat)
    else:
        adj_mat = adj_mat.toarray()

    xm = xmeans(data=adj_mat, tolerance=0.0001)
    xm.process()
    baseline_labels_ = xm.predict(adj_mat)

    c_score_with_com = metrics.completeness_score(y_train, baseline_labels_)
    h_score_with_com = metrics.homogeneity_score(y_train, baseline_labels_)
    v_score_with_com = metrics.v_measure_score(y_train, baseline_labels_)

    f_score_with_com, p_score_with_com, ip_score_with_com = get_f_p_r(
        y_train, baseline_labels_
    )

    print(
        f"{c_score_baseline},{h_score_baseline},{v_score_baseline},{p_score_baseline},{ip_score_baseline},{f_score_baseline},{c_score_with_com},{h_score_with_com},{v_score_with_com},{p_score_with_com},{ip_score_with_com},{f_score_with_com}"
    )
