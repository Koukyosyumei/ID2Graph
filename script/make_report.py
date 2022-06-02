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

    with open(
        os.path.join(parsed_args.path_to_dir, "temp_lp_tree_1.out"), mode="r"
    ) as f:
        leaf_purity_of_first_tree = [float(s.strip()) for s in f.readlines()]
    with open(
        os.path.join(parsed_args.path_to_dir, "temp_lp_tree_2.out"), mode="r"
    ) as f:
        leaf_purity_of_second_tree = [float(s.strip()) for s in f.readlines()]

    print(
        f"LP (1st tree): {np.round(np.mean(leaf_purity_of_first_tree), decimals=4)}±{np.round(np.std(leaf_purity_of_first_tree), decimals=4)}"
    )
    print(
        f"LP (2nd tree): {np.round(np.mean(leaf_purity_of_second_tree), decimals=4)}±{np.round(np.std(leaf_purity_of_second_tree), decimals=4)}"
    )

    with open(
        os.path.join(parsed_args.path_to_dir, "temp_train_auc.out"), mode="r"
    ) as f:
        train_auc = [float(s.strip()) for s in f.readlines()]
    with open(os.path.join(parsed_args.path_to_dir, "temp_val_auc.out"), mode="r") as f:
        val_auc = [float(s.strip()) for s in f.readlines()]

    print(
        f"AUC (train): {np.round(np.mean(train_auc), decimals=4)}±{np.round(np.std(train_auc), decimals=4)}"
    )
    print(
        f"AUC (validation): {np.round(np.mean(val_auc), decimals=4)}±{np.round(np.std(val_auc), decimals=4)}"
    )
