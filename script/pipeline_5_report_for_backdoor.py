import argparse
import glob
import os

from llatvfl.clustering import get_f_p_r
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

    paths = glob.glob(
        os.path.join(parsed_args.path_to_dir, "*_clusters_and_labels.out")
    )

    for p in paths:
        with open(p, mode="r") as f:
            lines = f.readlines()

        clusters_ids = [int(a) for a in lines[0].split(" ")[:-1]]
        true_labels = [int(a) for a in lines[1].split(" ")[:-1]]

        f_score, _, _ = get_f_p_r(true_labels, clusters_ids)
        v_score = metrics.v_measure_score(true_labels, clusters_ids)
        print(v_score, f_score)
