import argparse
import pandas as pd
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


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
        "-n",
        "--num_samples",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "-i",
        "--imbalance",
        type=float,
        default=1.0,
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
    X_train,
    y_train,
    X_val,
    y_val,
    output_path,
    col_alloc=None,
    parties_num=2,
    replace_nan="-1",
):
    row_num_train, col_num = X_train.shape
    row_num_val = X_val.shape[0]

    if col_alloc is None:
        col_alloc = np.array_split(
            random.sample(list(range(col_num)), col_num), parties_num
        )

    with open(output_path, mode="w") as f:
        f.write(f"{row_num_train} {col_num} {parties_num}\n")
        for ca in col_alloc:
            f.write(f"{len(ca)}\n")
            for i in ca:
                f.write(
                    " ".join(
                        [
                            str(x) if not np.isnan(x) else replace_nan
                            for x in X_train[:, i]
                        ]
                    )
                    + "\n"
                )
        f.write(" ".join([str(y) for y in y_train]) + "\n")
        f.write(f"{row_num_val}\n")
        for ca in col_alloc:
            for i in ca:
                f.write(
                    " ".join(
                        [
                            str(x) if not np.isnan(x) else replace_nan
                            for x in X_val[:, i]
                        ]
                    )
                    + "\n"
                )
        f.write(" ".join([str(y) for y in y_val]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    random.seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)

    if parsed_args.dataset_type == "givemesomecredit":
        df = pd.read_csv(os.path.join(parsed_args.path_to_dir, "cs-training.csv"))
        pos_df = df[df["SeriousDlqin2yrs"] == 1]
        neg_df = df[df["SeriousDlqin2yrs"] == 0]

        pos_num = int(parsed_args.num_samples / (1 + parsed_args.imbalance))
        neg_num = parsed_args.num_samples - pos_num
        pos_df = pos_df.sample(pos_num)
        neg_df = neg_df.sample(neg_num)

        df = pd.concat([pos_df, neg_df])
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
        ].values
        y = df["SeriousDlqin2yrs"].values

    elif parsed_args.dataset_type == "ucicreditcard":
        df = pd.read_csv(os.path.join(parsed_args.path_to_dir, "UCI_Credit_Card.csv"))
        pos_df = df[df["default.payment.next.month"] == 1]
        neg_df = df[df["default.payment.next.month"] == 0]

        pos_num = int(parsed_args.num_samples / (1 + parsed_args.imbalance))
        neg_num = parsed_args.num_samples - pos_num
        pos_df = pos_df.sample(pos_num)
        neg_df = neg_df.sample(neg_num)

        df = pd.concat([pos_df, neg_df])
        X = df[
            [
                "LIMIT_BAL",
                "SEX",
                "EDUCATION",
                "MARRIAGE",
                "AGE",
                "PAY_0",
                "PAY_2",
                "PAY_3",
                "PAY_4",
                "PAY_5",
                "PAY_6",
                "BILL_AMT1",
                "BILL_AMT2",
                "BILL_AMT3",
                "BILL_AMT4",
                "BILL_AMT5",
                "BILL_AMT6",
                "PAY_AMT1",
                "PAY_AMT2",
                "PAY_AMT3",
                "PAY_AMT4",
                "PAY_AMT5",
                "PAY_AMT6",
            ]
        ].values
        y = df["default.payment.next.month"].values

    elif parsed_args.dataset_type == "breastcancer":
        data = load_breast_cancer()
        X = data["data"]
        y = data["target"]

    else:
        raise ValueError(f"{parsed_args.dataset_type} is not supported.")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=1 / 5,
        random_state=parsed_args.seed,
        stratify=y,
    )

    convert_df_to_input(
        X_train,
        y_train,
        X_val,
        y_val,
        os.path.join(parsed_args.path_to_dir, f"{parsed_args.dataset_type}.in"),
        col_alloc=None,
        parties_num=2,
    )
