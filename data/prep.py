import argparse
import pandas as pd
import numpy as np

def convert_df_to_input(X_train, y_train, X_val, y_val, output_path, col_alloc=None, parties_num=2):
    row_num_train, col_num = X_train.shape
    row_num_val = X_val.shape[0]
    
    if col_alloc is None:
        col_alloc = np.array_split(list(range(col_num)), parties_num)
        
    with open(output_path, mode="w") as f:
        f.write(f"{row_num_train} {col_num} {parties_num}\n")
        for ca in col_alloc:
            f.write(f"{len(ca)}\n")
            for i in ca:
                f.write(" ".join([str(x) for x in X_train[:, i]])+"\n")
        f.write(" ".join([str(y) for y in y_train]))
        f.write(f"{row_num_val}\n")
        for i in range(col_num):
            f.write(" ".join([str(x) for x in X_val[:, i]])+"\n")
        f.write(" ".join([str(y) for y in y_val]))