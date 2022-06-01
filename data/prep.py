import argparse
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split


def add_args(parser):
    parser.add_argument(
        "-d",
        "--dataset_type",
        type=str,
        default="givemesomecredit",
    )

    parser.add_argument(
        "-p",
        "--path_to_dir",
        type=str,
        default="./data/givemesomecredit/",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    return args


def convert_df_to_input(
    X_train, y_train, X_val, y_val, output_path, col_alloc=None, parties_num=2
):
    row_num_train, col_num = X_train.shape
    row_num_val = X_val.shape[0]

    if col_alloc is None:
        col_alloc = np.array_split(list(range(col_num)), parties_num)

    with open(output_path, mode="w") as f:
        f.write(f"{row_num_train} {col_num} {parties_num}\n")
        for ca in col_alloc:
            f.write(f"{len(ca)}\n")
            for i in ca:
                f.write(
                    " ".join(
                        [str(x) if not np.isnan(x) else "-1" for x in X_train[:, i]]
                    )
                    + "\n"
                )
        f.write(" ".join([str(y) for y in y_train]) + "\n")
        f.write(f"{row_num_val}\n")
        for i in range(col_num):
            f.write(
                " ".join([str(x) if not np.isnan(x) else "-1" for x in X_val[:, i]])
                + "\n"
            )
        f.write(" ".join([str(y) for y in y_val]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    if parsed_args.dataset_type == "givemesomecredit":
        df = pd.read_csv(os.path.join(parsed_args.path_to_dir, "cs-training.csv"))
        X = df[
            [
                "RevolvingUtilizationOfUnsecuredLines",
                "age",
                "NumberOfTime30-59DaysPastDueNotWorse",
                "DebtRatio",
                "MonthlyIncome",
                "NumberOfOpenCreditLinesAndLoans",
                "NumberOfTimes90DaysLate",
                "NumberRealEstateLoansOrLines",
                "NumberOfTime60-89DaysPastDueNotWorse",
                "NumberOfDependents",
            ]
        ]
        y = df["SeriousDlqin2yrs"]
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.33,
            random_state=parsed_args.seed,
            stratify=y,
        )

        convert_df_to_input(
            X_train.values,
            y_train.values,
            X_val.values,
            y_val.values,
            os.path.join(parsed_args.path_to_dir, "givemesomecredit.in"),
            col_alloc=None,
            parties_num=2,
        )
