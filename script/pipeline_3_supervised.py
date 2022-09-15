import argparse
import random

import numpy as np
from llatvfl.clustering import ReducedKMeans
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# from matplotlib import pyplot as plt

label2maker = {0: "o", 1: "x"}
clustering_type2cls = {"vanila": KMeans, "reduced": ReducedKMeans}


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
        "-k",
        "--clustering_type",
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    clustering_cls = clustering_type2cls[parsed_args.clustering_type]

    print("baseline_a,our_a")

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

        y_train = lines[num_col + num_party + 1].split(" ")
        y_train = np.array([int(y) for y in y_train])
        unique_labels = np.unique(y_train)

    with open(parsed_args.path_to_com_file, mode="r") as f:
        lines = f.readlines()
        comm_num = int(lines[0])
        node_num = int(lines[1])
        X_com = np.zeros((num_row, comm_num))

        for i in range(comm_num):
            temp_nodes_in_comm = lines[i + 2].split(" ")[:-1]
            for k in temp_nodes_in_comm:
                X_com[int(k), i] += 1

    X_train_with_com = np.hstack([X_train.T, X_com])
    num_train = X_train_with_com.shape[0]
    num_train_aux = int(num_train * 0.1)
    public_idxs = random.sample(list(range(num_train)), num_train_aux)
    private_idxs = list(set(list(range(num_train))) - set(public_idxs))

    clf_baseline = RandomForestClassifier()
    clf_baseline.fit(X_train.T[public_idxs], y_train[public_idxs])

    clf_our = RandomForestClassifier()
    clf_our.fit(X_train_with_com[public_idxs], y_train[public_idxs])

    accuracy_baseline = clf_baseline.score(
        X_train.T[private_idxs], y_train[private_idxs]
    )
    accuracy_with_com = clf_our.score(
        X_train_with_com[private_idxs], y_train[private_idxs]
    )

    print(f"{accuracy_baseline},{accuracy_with_com}")
