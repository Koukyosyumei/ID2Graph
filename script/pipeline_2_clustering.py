import argparse

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pyclustering.cluster.xmeans import xmeans

from llatvfl.clustering import get_f_p_r

N_INIT = 10
label2maker = {0: "o", 1: "x"}
plt.style.use("ggplot")


marker_shapes = ["o", "o", "*", "*", "^", "^", "d", "d"]
marker_colors = ["k", "white", "k", "white", "k", "white", "k", "white"]


def visualize_clusters(X, y_true, num_classes, title, saved_path, h=0.02, eps=0.02):
    reduced_data = PCA(n_components=2).fit_transform(X)
    kmeans = clustering_cls(
        n_clusters=num_classes, n_init=N_INIT, random_state=parsed_args.seed
    ).fit(reduced_data)
    x_min, x_max = reduced_data[:, 0].min() - eps, reduced_data[:, 0].max() + eps
    y_min, y_max = reduced_data[:, 1].min() - eps, reduced_data[:, 1].max() + eps
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        # cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
        alpha=0.7,
    )

    for i in range(num_classes):
        idx = np.where(y_true == i)[0]
        plt.scatter(
            reduced_data[idx, 0],
            reduced_data[idx, 1],
            c=marker_colors[i],
            marker=marker_shapes[i],
            s=15,
            alpha=0.9,
            edgecolors="black",
        )
    # Plot the centroids as a white X
    # centroids = kmeans.cluster_centers_
    # plt.scatter(
    #    centroids[:, 0],
    #    centroids[:, 1],
    #    marker="x",
    #    s=169,
    #    linewidths=3,
    #    color="w",
    #    zorder=10,
    # )
    # plt.title("Breastcance " + title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(saved_path, bbox_inches="tight", dpi=300)


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
    parser.add_argument("-g", "--graph_plot", action="store_true")

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

    if parsed_args.clustering_type == "kmeans":
        kmeans = KMeans(
            n_clusters=num_classes, n_init=N_INIT, random_state=parsed_args.seed
        ).fit(X_train_minmax)
        baseline_labels = kmeans.labels_
    elif parsed_args.clustering_type == "xmeans":
        xm = xmeans(data=X_train_minmax, tolerance=0.0001)
        xm.process()
        baseline_labels = xm.predict(X_train_minmax)

    c_score_baseline = metrics.completeness_score(y_train, baseline_labels)
    h_score_baseline = metrics.homogeneity_score(y_train, baseline_labels)
    v_score_baseline = metrics.v_measure_score(y_train, baseline_labels)

    _, p_score_baseline, ip_score_baseline = get_f_p_r(y_train, baseline_labels)
    f_score_baseline = metrics.fowlkes_mallows_score(y_train, baseline_labels)
    cm_matrix = metrics.cluster.contingency_matrix(y_train, baseline_labels)

    if parsed_args.graph_plot:
        visualize_clusters(
            X_train_minmax,
            y_train,
            num_classes,
            "CL",
            f"{parsed_args.path_to_input_file.split('.')[0]}_CL.png",
        )

    with open(parsed_args.path_to_com_file, mode="r") as f:
        lines = f.readlines()
        comm_num = int(lines[0])
        node_num = int(lines[1])
        X_com = np.zeros((num_row, comm_num))

        for i in range(comm_num):
            temp_nodes_in_comm = lines[i + 2].split(" ")[:-1]
            for k in temp_nodes_in_comm:
                X_com[int(k), i] += parsed_args.weight_for_community_variables

    if parsed_args.clustering_type == "kmeans":
        kmeans_with_com = KMeans(
            n_clusters=num_classes, n_init=N_INIT, random_state=parsed_args.seed
        ).fit(np.hstack([X_train_minmax, X_com]))
        with_com_labels = kmeans_with_com.labels_
    elif parsed_args.clustering_type == "xmeans":
        xm_with_com = xmeans(data=np.hstack([X_train_minmax, X_com]), tolerance=0.0001)
        xm_with_com.process()
        clusters = xm_with_com.get_clusters()
        cluster_size = len(clusters)
        kmeans_with_com = KMeans(
            n_clusters=cluster_size, n_init=N_INIT, random_state=parsed_args.seed
        ).fit(np.hstack([X_train_minmax, X_com]))
        with_com_labels = kmeans_with_com.labels_
        # with_com_labels = xm_with_com.predict(
        #    np.hstack([X_train_minmax, X_com]))

    c_score_with_com = metrics.completeness_score(y_train, with_com_labels)
    h_score_with_com = metrics.homogeneity_score(y_train, with_com_labels)
    v_score_with_com = metrics.v_measure_score(y_train, with_com_labels)

    _, p_score_with_com, ip_score_with_com = get_f_p_r(y_train, with_com_labels)
    f_score_with_com = metrics.fowlkes_mallows_score(y_train, with_com_labels)

    if parsed_args.graph_plot:
        visualize_clusters(
            np.hstack([X_train_minmax, X_com]),
            y_train,
            num_classes,
            "ID2Graph",
            f"{parsed_args.path_to_input_file.split('.')[0]}_ID2Graph.png",
        )

    print(
        f"{c_score_baseline},{h_score_baseline},{v_score_baseline},{p_score_baseline},{ip_score_baseline},{f_score_baseline},{c_score_with_com},{h_score_with_com},{v_score_with_com},{p_score_with_com},{ip_score_with_com},{f_score_with_com}"
    )
