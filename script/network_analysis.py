import numpy as np
import argparse
import os


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
    path_to_adj_mat_file = os.path.join(parsed_args.path_to_dir, "adj_mat.txt")

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
