import numpy as np
import argparse
import os
import glob
from sklearn.cluster import KMeans
from sklearn import preprocessing
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
    print(
        "baseline_c,baseline_h,baseline_v,baseline_p,baseline_ip,baseline_f,our_c,our_h,our_v,our_p,our_ip,our_f"
    )
    for path_to_input_file in list_input_files:
        round_idx = path_to_input_file.split("/")[-1].split("_")[0]
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

        kmeans = KMeans(n_clusters=2, random_state=int(round_idx)).fit(X_train_minmax)
        c_score_baseline = metrics.completeness_score(y_train, kmeans.labels_)
        h_score_baseline = metrics.homogeneity_score(y_train, kmeans.labels_)
        v_score_baseline = metrics.v_measure_score(y_train, kmeans.labels_)

        cm_matrix = metrics.cluster.contingency_matrix(y_train, kmeans.labels_)
        p_score_baseline = cm_matrix.max(axis=0).sum() / num_row
        ip_score_baseline = cm_matrix.max(axis=1).sum() / num_row
        f_score_baseline = (
            2
            * p_score_baseline
            * ip_score_baseline
            / (p_score_baseline + ip_score_baseline)
        )

        path_to_adj_mat_file = os.path.join(
            parsed_args.path_to_dir, f"{round_idx}_communities.out"
        )
        with open(path_to_adj_mat_file, mode="r") as f:
            lines = f.readlines()
            comm_num = int(lines[0])
            node_num = int(lines[1])
            X_com = np.zeros((node_num, comm_num))

            for i in range(comm_num):
                temp_nodes_in_comm = lines[i + 2].split(" ")[:-1]
                for k in temp_nodes_in_comm:
                    X_com[int(k), i] += 1

        kmeans_with_com = KMeans(n_clusters=2, random_state=int(round_idx)).fit(
            np.hstack([X_train_minmax, X_com])
        )
        c_score_with_com = metrics.completeness_score(y_train, kmeans_with_com.labels_)
        h_score_with_com = metrics.homogeneity_score(y_train, kmeans_with_com.labels_)
        v_score_with_com = metrics.v_measure_score(y_train, kmeans_with_com.labels_)

        cm_matrix = metrics.cluster.contingency_matrix(y_train, kmeans_with_com.labels_)
        p_score_with_com = cm_matrix.max(axis=0).sum() / num_row
        ip_score_with_com = cm_matrix.max(axis=1).sum() / num_row
        f_score_with_com = (
            2
            * p_score_with_com
            * ip_score_with_com
            / (p_score_with_com + ip_score_with_com)
        )

        print(
            f"{c_score_baseline},{h_score_baseline},{v_score_baseline},{p_score_baseline},{ip_score_baseline},{f_score_baseline},{c_score_with_com},{h_score_with_com},{v_score_with_com},{p_score_with_com},{ip_score_with_com},{f_score_with_com}"
        )
