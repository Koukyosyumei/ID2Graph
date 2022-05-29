import argparse
import pandas as pd
import numpy as np

def convert_df_to_input(X, Y, output_path, col_alloc=None, parties_num=2):
    row_num, col_num = X.shape
    if col_alloc is not None:
        col_alloc = np.array_split(list(range(col_num)), parties_num)

    with open(output_path, mode="w") as f:
        f.write(f"{row_num} {col_num} {parties_num}\n")
        for ca in col_alloc:
            f.write(f"{len(ca)}\n")
            for i in ca:
                f.write(" ".join([str(x) for x in X[:, i]])+"\n")
        f.write(" ".join([str(y) for y in Y]))

def add_args(parser):
    parser.add_argument(
        "-t",
        "--fedkd_type",
        type=str,
        default="fedgems",
        help="type of FedKD; FedMD, FedGEMS, or FedGEMS",
    )

def main():
    pass