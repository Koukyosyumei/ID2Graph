import argparse
import os
import random

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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
        "-f",
        "--feature_num_ratio_of_active_party",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "-v",
        "--feature_num_ratio_of_passive_party",
        type=float,
        default=-1,
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


def sampling_col_alloc(
    col_num, feature_num_ratio_of_active_party, feature_num_ratio_of_passive_party
):
    shufled_col_indicies = random.sample(list(range(col_num)), col_num)
    col_num_of_active_party = max(1, int(feature_num_ratio_of_active_party * col_num))
    if feature_num_ratio_of_passive_party < 0:
        col_alloc = [
            shufled_col_indicies[:col_num_of_active_party],
            shufled_col_indicies[col_num_of_active_party:],
        ]
    else:
        col_num_of_passive_party = max(
            1, int(feature_num_ratio_of_passive_party * col_num)
        )
        col_alloc = [
            shufled_col_indicies[:col_num_of_active_party],
            shufled_col_indicies[
                col_num_of_active_party : (
                    min(
                        col_num_of_active_party + col_num_of_passive_party,
                        col_num,
                    )
                )
            ],
        ]

    return col_alloc


def convert_df_to_input(
    X_train,
    y_train,
    X_val,
    y_val,
    output_path,
    col_alloc=None,
    parties_num=2,
    feature_num_ratio_of_active_party=0.5,
    feature_num_ratio_of_passive_party=-1,
    replace_nan="-1",
):
    row_num_train, col_num = X_train.shape
    row_num_val = X_val.shape[0]

    if col_alloc is None:
        col_alloc = sampling_col_alloc(
            col_num,
            feature_num_ratio_of_active_party,
            feature_num_ratio_of_passive_party,
        )

    with open(output_path, mode="w") as f:
        f.write(
            f"{len(list(set(y_train)))} {row_num_train} {len(sum(col_alloc, []))} {parties_num}\n"
        )
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


def sampling(df, yname, parsed_args):
    if parsed_args.num_samples == -1:
        return df
    else:
        pos_df = df[df[yname] == 1]
        neg_df = df[df[yname] == 0]
        pos_num = int(parsed_args.num_samples / (1 + parsed_args.imbalance))
        neg_num = parsed_args.num_samples - pos_num
        pos_df = pos_df.sample(pos_num)
        neg_df = neg_df.sample(neg_num)
        df_ = pd.concat([pos_df, neg_df])
        return df_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    random.seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)

    col_alloc = None

    if parsed_args.dataset_type == "givemesomecredit":
        df = pd.read_csv(os.path.join(parsed_args.path_to_dir, "cs-training.csv"))

        df_pos = df[df["SeriousDlqin2yrs"] == 1]
        df_neg = df[df["SeriousDlqin2yrs"] == 0]

        n_neg = parsed_args.num_samples - df_pos.shape[0]
        df = pd.concat([df_pos, df_neg.sample(n_neg)])

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

    elif parsed_args.dataset_type == "ionosphere":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "ionosphere.data"), header=None
        )
        df[34] = df[34].apply(lambda x: 0 if x == "g" else 1)
        df = sampling(df, 34, parsed_args)

        X = df[list(range(34))].values
        y = df[34].values

    elif parsed_args.dataset_type == "heartdisease":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "processed.cleveland.data"),
            header=None,
        )
        df = sampling(df, 13, parsed_args)

        X = df[list(range(13))].values
        y = df[13].values

    elif parsed_args.dataset_type == "avila":
        df_tr = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "avila-tr.txt"),
            header=None,
        )
        df_ts = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "avila-ts.txt"),
            header=None,
        )
        df = pd.concat([df_tr, df_ts], axis=0)
        string2int = {
            s: i
            for i, s in enumerate(
                ["A", "B", "C", "D", "E", "F", "G", "H", "I", "W", "X", "Y"]
            )
        }
        df[10] = df[10].apply(lambda x: string2int[x])
        df = sampling(df, 10, parsed_args)

        X = df[list(range(10))].values
        y = df[10].values

    elif parsed_args.dataset_type == "glass":
        y_dict = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5}
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "glass.data"), header=None
        )
        df = sampling(df, 10, parsed_args)
        df[10] = df[10].apply(lambda x: y_dict[x])

        X = df[list(range(1, 10))].values
        y = df[10].values

    elif parsed_args.dataset_type == "spambase":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "spambase.data"), header=None
        )
        df = sampling(df, 57, parsed_args)

        X = df[list(range(57))].values
        y = df[57].values

    elif parsed_args.dataset_type in ["dummy0", "dummy1"]:
        df = pd.read_csv(os.path.join(parsed_args.path_to_dir, "dummy.csv"))
        df = sampling(df, "y", parsed_args)

        X = df[[f"x{i}" for i in range(1, 21)]].values
        y = df["y"].values

    elif parsed_args.dataset_type == "parkinson":
        df = pd.read_csv(os.path.join(parsed_args.path_to_dir, "parkinsons.data"))
        df = sampling(df, "status", parsed_args)

        X = df[
            [
                "MDVP:Fo(Hz)",
                "MDVP:Fhi(Hz)",
                "MDVP:Flo(Hz)",
                "MDVP:Jitter(%)",
                "MDVP:Jitter(Abs)",
                "MDVP:RAP",
                "MDVP:PPQ",
                "Jitter:DDP",
                "MDVP:Shimmer",
                "MDVP:Shimmer(dB)",
                "Shimmer:APQ3",
                "Shimmer:APQ5",
                "MDVP:APQ",
                "Shimmer:DDA",
                "NHR",
                "HNR",
                "RPDE",
                "DFA",
                "spread1",
                "spread2",
                "D2",
                "PPE",
            ]
        ].values
        y = df["status"].values

    elif parsed_args.dataset_type == "phishing":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "phishing.data"), header=None
        )
        df[30] = df[30].apply(lambda x: 0 if x == -1 else 1)
        df = sampling(df, 30, parsed_args)

        X = df[list(range(30))].values
        y = df[30].values

    elif parsed_args.dataset_type == "bank":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "bank-full.csv"), sep=";"
        )
        df["education"] = df["education"].replace(
            {"primary": 1, "secondary": 2, "tertiary": 3, "unknown": 0}
        )
        df["month"] = df["month"].replace(
            {
                "jan": 1,
                "feb": 2,
                "mar": 3,
                "apr": 4,
                "may": 5,
                "jun": 6,
                "jul": 7,
                "aug": 8,
                "sep": 9,
                "oct": 10,
                "nov": 11,
                "dec": 12,
            }
        )
        df["marital"] = df["marital"].replace(
            {"single": 1, "married": 2, "divorced": 3}
        )
        df["y"] = df["y"].apply(lambda y: 1 if y == "yes" else 0)

        col_alloc_origin = sampling_col_alloc(
            col_num=df.shape[1] - 1,
            feature_num_ratio_of_active_party=parsed_args.feature_num_ratio_of_active_party,
            feature_num_ratio_of_passive_party=parsed_args.feature_num_ratio_of_passive_party,
        )
        X_d = df.drop("y", axis=1)
        X_a = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[0]]], drop_first=True)
        X_p = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[1]]], drop_first=True)
        col_alloc = [
            list(range(X_a.shape[1])),
            list(range(X_a.shape[1], X_a.shape[1] + X_p.shape[1])),
        ]
        X = pd.concat([X_a, X_p], axis=1).values
        y = df["y"].values

    elif parsed_args.dataset_type == "bankruptcy":
        df = pd.read_csv(os.path.join(parsed_args.path_to_dir, "data.csv"))
        df = sampling(df, "Bankrupt?", parsed_args)

        X = df.drop("Bankrupt?", axis=1).values
        y = df["Bankrupt?"].values

    elif parsed_args.dataset_type == "cnae9":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "CNAE-9.data"), header=None
        )
        df = sampling(df, 0, parsed_args)
        df[0] = df[0] - 1

        X = df[list(range(1, 857))].values
        y = df[0].values

    elif parsed_args.dataset_type == "drive":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "Sensorless_drive_diagnosis.txt"),
            sep=" ",
            header=None,
        )

        X = df[list(range(48))].values
        y = df[48].values - 1

    elif parsed_args.dataset_type == "adult":
        df = pd.concat(
            [
                pd.read_csv(
                    os.path.join(parsed_args.path_to_dir, "adult.data"), header=None
                ),
                pd.read_csv(
                    os.path.join(parsed_args.path_to_dir, "adult.test"), header=None
                ),
            ]
        )
        df[3] = df[3].replace(
            {
                " Preschool": 0,
                " 1st-4th": 1,
                " 5th-6th": 2,
                " 7th-8th": 3,
                " 9th": 4,
                " 10th": 5,
                " 11th": 6,
                " 12th": 7,
                " HS-grad": 8,
                " Prof-school": 9,
                " Assoc-acdm": 10,
                " Assoc-voc": 11,
                " Some-college": 12,
                " Bachelors": 13,
                " Masters": 14,
                " Doctorate": 15,
            }
        )
        df[14] = df[14].apply(lambda y: 0 if y == " <=50K" else 1)
        df = sampling(df, 14, parsed_args)

        col_alloc_origin = sampling_col_alloc(
            col_num=df.shape[1] - 1,
            feature_num_ratio_of_active_party=parsed_args.feature_num_ratio_of_active_party,
            feature_num_ratio_of_passive_party=parsed_args.feature_num_ratio_of_passive_party,
        )
        X_d = df.drop(14, axis=1)
        X_a = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[0]]], drop_first=True)
        X_p = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[1]]], drop_first=True)
        col_alloc = [
            list(range(X_a.shape[1])),
            list(range(X_a.shape[1], X_a.shape[1] + X_p.shape[1])),
        ]
        X = pd.concat([X_a, X_p], axis=1).values
        y = df[14].values

    elif parsed_args.dataset_type == "nursery":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "nursery.data"), header=None
        )
        df[8] = LabelEncoder().fit_transform(df[8].values)

        col_alloc_origin = sampling_col_alloc(
            col_num=df.shape[1] - 1,
            feature_num_ratio_of_active_party=parsed_args.feature_num_ratio_of_active_party,
            feature_num_ratio_of_passive_party=parsed_args.feature_num_ratio_of_passive_party,
        )
        X_d = df.drop(8, axis=1)
        X_a = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[0]]], drop_first=True)
        X_p = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[1]]], drop_first=True)
        col_alloc = [
            list(range(X_a.shape[1])),
            list(range(X_a.shape[1], X_a.shape[1] + X_p.shape[1])),
        ]
        X = pd.concat([X_a, X_p], axis=1).values
        y = df[8].values

    elif parsed_args.dataset_type == "packdd":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "PAKDD2010_Modeling_Data.csv"),
            header=None,
            sep="\t",
        )
        df = sampling(df, 53, parsed_args)

        col_alloc_origin = sampling_col_alloc(
            col_num=df.shape[1] - 1,
            feature_num_ratio_of_active_party=parsed_args.feature_num_ratio_of_active_party,
            feature_num_ratio_of_passive_party=parsed_args.feature_num_ratio_of_passive_party,
        )
        X_d = df.drop(53, axis=1)
        X_a = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[0]]], drop_first=True)
        X_p = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[1]]], drop_first=True)
        col_alloc = [
            list(range(X_a.shape[1])),
            list(range(X_a.shape[1], X_a.shape[1] + X_p.shape[1])),
        ]
        X = pd.concat([X_a, X_p], axis=1).values
        y = df[53].values

    elif parsed_args.dataset_type == "credit":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "credit_risk_dataset.csv")
        )
        df["cb_person_default_on_file"] = df["cb_person_default_on_file"].replace(
            {"Y": 1, "N": 0}
        )
        df["loan_grade"] = df["loan_grade"].replace(
            {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
        )
        df = sampling(df, "loan_status", parsed_args)

        col_alloc_origin = sampling_col_alloc(
            col_num=df.shape[1] - 1,
            feature_num_ratio_of_active_party=parsed_args.feature_num_ratio_of_active_party,
            feature_num_ratio_of_passive_party=parsed_args.feature_num_ratio_of_passive_party,
        )
        X_d = df.drop("loan_status", axis=1)
        X_a = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[0]]], drop_first=True)
        X_p = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[1]]], drop_first=True)
        col_alloc = [
            list(range(X_a.shape[1])),
            list(range(X_a.shape[1], X_a.shape[1] + X_p.shape[1])),
        ]
        X = pd.concat([X_a, X_p], axis=1).values
        y = df["loan_status"].values

    elif parsed_args.dataset_type == "coupon":
        df = pd.read_csv(
            os.path.join(
                parsed_args.path_to_dir, "in-vehicle-coupon-recommendation.csv"
            )
        )
        df = sampling(df, "Y", parsed_args)

        col_alloc_origin = sampling_col_alloc(
            col_num=df.shape[1] - 1,
            feature_num_ratio_of_active_party=parsed_args.feature_num_ratio_of_active_party,
            feature_num_ratio_of_passive_party=parsed_args.feature_num_ratio_of_passive_party,
        )
        X_d = df.drop("Y", axis=1)
        X_a = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[0]]], drop_first=True)
        X_p = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[1]]], drop_first=True)
        col_alloc = [
            list(range(X_a.shape[1])),
            list(range(X_a.shape[1], X_a.shape[1] + X_p.shape[1])),
        ]
        X = pd.concat([X_a, X_p], axis=1).values
        y = df["Y"].values

    elif parsed_args.dataset_type == "dummy":
        n = 30000
        m = 10

        y = np.random.binomial(1, 0.5, n)
        X = np.stack([y + np.random.normal(size=n) * (i + 1) for i in range(m)]).T

        active_col = [
            i for i in range(10 * parsed_args.feature_num_ratio_of_active_party)
        ]
        passive_col = list(set(range(m)) - set(active_col))
        col_alloc = [active_col, passive_col]

    elif parsed_args.dataset_type == "hcv":
        cols = [
            "Age",
            "Sex",
            "ALB",
            "ALP",
            "ALT",
            "AST",
            "BIL",
            "CHE",
            "CHOL",
            "CREA",
            "GGT",
            "PROT",
        ]
        label_dict = {
            "0=Blood Donor": 0,
            "0s=suspect Blood Donor": 1,
            "1=Hepatitis": 2,
            "2=Fibrosis": 3,
            "3=Cirrhosis": 4,
        }
        df = pd.read_csv(os.path.join(parsed_args.path_to_dir, "hcvdat0.csv"))
        df["Category"] = df["Category"].apply(lambda x: label_dict[x])
        df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "m" else 0)
        df = sampling(df, "Category", parsed_args)

        X = df[cols].values
        y = df["Category"].values

    elif parsed_args.dataset_type == "sonar":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "sonar.all-data"), header=None
        )
        df[60] = df[60].apply(lambda x: 1 if x == "R" else 0)
        df = sampling(df, 60, parsed_args)

        X = df[list(range(60))].values
        y = df[60].values

    elif parsed_args.dataset_type == "polish":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "3year.csv"), header=None
        )
        df = df.replace("?", -99)
        df = sampling(df, 64, parsed_args)
        for i in range(65):
            df[i] = df[i].astype(float)

        X = df[list(range(64))].values
        y = df[64].values

    elif parsed_args.dataset_type == "diabetic":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "diabetic_data_diag_encoded.csv")
        )

        df = df.drop(
            ["encounter_id", "patient_nbr", "diag_1", "diag_2", "diag_3"], axis=1
        )
        df = df.replace({"No": 0, "Down": -1, "Steady": 1, "Up": 2})
        df["gender"] = df["gender"].replace(
            {"Female": 0, "Unknown/Invalid": 1, "Male": 2}
        )
        df["age"] = df["age"].apply(lambda x: int(x[1]))
        df["weight"] = df["weight"].replace(
            {
                "?": 0,
                "[0-25)": 1,
                "[25-50)": 2,
                "[50-75)": 3,
                "[75-100)": 4,
                "[100-125)": 5,
                "[125-150)": 6,
                "[150-175)": 7,
                "[175-200)": 8,
                ">200": 9,
            }
        )
        df["max_glu_serum"] = df["max_glu_serum"].replace(
            {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
        )
        df["A1Cresult"] = df["A1Cresult"].replace(
            {"None": 0, "Norm": 1, ">7": 2, ">8": 2}
        )
        df["readmitted"] = LabelEncoder().fit_transform(df["readmitted"].values)

        col_alloc_origin = sampling_col_alloc(
            col_num=df.shape[1] - 1,
            feature_num_ratio_of_active_party=parsed_args.feature_num_ratio_of_active_party,
            feature_num_ratio_of_passive_party=parsed_args.feature_num_ratio_of_passive_party,
        )
        X_d = df.drop("readmitted", axis=1)
        X_a = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[0]]], drop_first=True)
        X_p = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[1]]], drop_first=True)
        col_alloc = [
            list(range(X_a.shape[1])),
            list(range(X_a.shape[1], X_a.shape[1] + X_p.shape[1])),
        ]
        X = pd.concat([X_a, X_p], axis=1).values
        y = df["readmitted"].values

    elif parsed_args.dataset_type == "ucicreditcard":
        df = pd.read_csv(os.path.join(parsed_args.path_to_dir, "UCI_Credit_Card.csv"))
        df = sampling(df, "default.payment.next.month", parsed_args)

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

    elif parsed_args.dataset_type == "waveform":
        df = pd.read_csv(os.path.join(parsed_args.path_to_dir, "waveform-5000.csv"))
        if parsed_args.num_samples != -1:
            df = df.sample(parsed_args.num_samples)
        X = df[[f"x{i}" for i in range(1, 41)]].values
        y = df["class"].values

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

    if parsed_args.dataset_type == "dummy0":
        col_alloc = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ]
    elif parsed_args.dataset_type == "dummy1":
        col_alloc = [
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ]

    convert_df_to_input(
        X_train,
        y_train,
        X_val,
        y_val,
        os.path.join(
            parsed_args.path_to_dir, f"{parsed_args.dataset_type}_{parsed_args.seed}.in"
        ),
        col_alloc=col_alloc,
        feature_num_ratio_of_active_party=parsed_args.feature_num_ratio_of_active_party,
        feature_num_ratio_of_passive_party=parsed_args.feature_num_ratio_of_passive_party,
        parties_num=2,
    )
