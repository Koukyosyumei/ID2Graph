import argparse
import os
import random

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def add_args(parser):
    parser.add_argument(
        "-d",
        "--dataset_type",
        type=str,
    )

    parser.add_argument(
        "-p",
        "--path_to_dir",
        type=str,
    )

    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=-1,
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
        "-s",
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "-i",
        "--feature_importance",
        type=int,
        default=-1,
    )

    args = parser.parse_args()
    return args


def sampling_col_alloc(
    col_num, feature_num_ratio_of_active_party, feature_num_ratio_of_passive_party
):
    shufled_col_indicies = random.sample(list(range(col_num)), col_num)
    col_num_of_active_party = max(
        1, int(feature_num_ratio_of_active_party * col_num))
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
                col_num_of_active_party: (
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
        pos_num = int(parsed_args.num_samples / 2)
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

    if parsed_args.dataset_type == "avila":
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

    elif parsed_args.dataset_type == "phishing":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "phishing.data"), header=None
        )
        df[30] = df[30].apply(lambda x: 0 if x == -1 else 1)
        df = sampling(df, 30, parsed_args)

        X = df[list(range(30))].values
        y = df[30].values

    elif parsed_args.dataset_type == "drive":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir,
                         "Sensorless_drive_diagnosis.txt"),
            sep=" ",
            header=None,
        )

        X = df[list(range(48))].values
        y = df[48].values - 1

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
        X_a = pd.get_dummies(
            X_d[X_d.columns[col_alloc_origin[0]]], drop_first=True)
        X_p = pd.get_dummies(
            X_d[X_d.columns[col_alloc_origin[1]]], drop_first=True)
        col_alloc = [
            list(range(X_a.shape[1])),
            list(range(X_a.shape[1], X_a.shape[1] + X_p.shape[1])),
        ]
        X = pd.concat([X_a, X_p], axis=1).values
        y = df[8].values

    elif parsed_args.dataset_type == "fraud":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir,
                         "fraud_detection_bank_dataset.csv")
        )

        X = df[[f"col_{i}" for i in range(112)]].values
        y = df["targets"].values

    elif parsed_args.dataset_type == "ucicreditcard":
        df = pd.read_csv(os.path.join(
            parsed_args.path_to_dir, "UCI_Credit_Card.csv"))
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

    elif parsed_args.dataset_type == "breastcancer":
        data = load_breast_cancer()
        X = data["data"]
        y = data["target"]
    elif parsed_args.dataset_type == "givemesomecredit":
        df = pd.read_csv(os.path.join(
            parsed_args.path_to_dir, "cs-training.csv"))
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
    elif parsed_args.dataset_type == "bank":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "bank-full.csv"), sep=";"
        )
        df["y"] = df["y"].apply(lambda x: 1 if x == "yes" else 0)
        df = pd.get_dummies(df)
        X = df.drop("y", axis=1).values
        y = df["y"].values
    elif parsed_args.dataset_type == "dota2":
        df1 = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "dota2Train.csv"), header=None
        )
        df2 = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "dota2Train.csv"), header=None
        )
        df = pd.concat([df1, df2])
        X = df.drop(0, axis=1).values
        y = (df[0].values + 1) / 2
    elif parsed_args.dataset_type == "sepsis":
        df = pd.read_csv(
            os.path.join(
                parsed_args.path_to_dir,
                "s41598-020-73558-3_sepsis_survival_primary_cohort.csv",
            )
        )
        X = df.drop("hospital_outcome_1alive_0dead", axis=1).values
        y = df["hospital_outcome_1alive_0dead"].values
    elif parsed_args.dataset_type == "miniboone":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "MiniBooNE_PID.txt"),
            skiprows=1,
            header=None,
            delim_whitespace=True,
        )
        X = df.values
        y = np.array([0] * 36499 + [1] * 93565)
    elif parsed_args.dataset_type == "drugs":
        df = pd.read_csv(os.path.join(parsed_args.path_to_dir, "drugs.csv"))
        X = df[
            [
                "condition",
                "usefulCount",
                "sentiment",
                "day",
                "month",
                "Year",
                "sentiment_clean_ss",
                "count_word",
                "count_unique_word",
                "count_letters",
                "count_punctuations",
                "count_words_upper",
                "count_words_title",
                "count_stopwords",
                "mean_word_len",
            ]
        ].values
        y = df["ratings"].values

    elif parsed_args.dataset_type == "obesity":
        df = pd.read_csv(
            os.path.join(
                parsed_args.path_to_dir, "ObesityDataSet_raw_and_data_sinthetic.csv"
            )
        )
        df["NObeyesdad"] = LabelEncoder().fit_transform(df["NObeyesdad"].values)

        col_alloc_origin = sampling_col_alloc(
            col_num=df.shape[1] - 1,
            feature_num_ratio_of_active_party=parsed_args.feature_num_ratio_of_active_party,
            feature_num_ratio_of_passive_party=parsed_args.feature_num_ratio_of_passive_party,
        )
        X_d = df.drop("NObeyesdad", axis=1)
        X_a = pd.get_dummies(
            X_d[X_d.columns[col_alloc_origin[0]]], drop_first=True)
        X_p = pd.get_dummies(
            X_d[X_d.columns[col_alloc_origin[1]]], drop_first=True)
        col_alloc = [
            list(range(X_a.shape[1])),
            list(range(X_a.shape[1], X_a.shape[1] + X_p.shape[1])),
        ]
        X = pd.concat([X_a, X_p], axis=1).values
        y = df["NObeyesdad"]

    elif parsed_args.dataset_type == "pucrio":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "pucrio.csv"),
            sep=";",
            low_memory=False,
        )
        df = df.drop_duplicates()
        df["gender"] = df["gender"].apply(lambda x: 1 if x == "Woman" else 0)
        df["how_tall_in_meters"] = df["how_tall_in_meters"].apply(
            lambda x: float(x.replace(",", "."))
        )
        df["body_mass_index"] = df["body_mass_index"].apply(
            lambda x: float(x.replace(",", "."))
        )
        df["z4"] = df["z4"].apply(lambda x: int(str(x)[:4]))
        df["class"] = LabelEncoder().fit_transform(df["class"].values)
        X = df.drop(["user", "class"], axis=1).values
        y = df["class"].values

    elif parsed_args.dataset_type == "indoor":
        df_timestamp1 = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "measure1_timestamp_id.csv"),
            header=None,
        )
        df_phonesens1 = pd.read_csv(
            os.path.join(parsed_args.path_to_dir,
                         "measure1_smartphone_sens.csv")
        )
        df_phonewifi1 = pd.read_csv(
            os.path.join(parsed_args.path_to_dir,
                         "measure1_smartphone_wifi.csv"),
            header=None,
        )
        df_watchsens1 = pd.read_csv(
            os.path.join(parsed_args.path_to_dir,
                         "measure1_smartwatch_sens.csv")
        )
        df_phonesens2 = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "measure2_phone_sens.csv")
        )
        df_phonewifi2 = pd.read_csv(
            os.path.join(parsed_args.path_to_dir,
                         "measure2_smartphone_wifi.csv"),
            header=None,
        )
        df_timestamp2 = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "measure2_timestamp_id.csv"),
            header=None,
        )
        df_watchsens2 = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "measure2_watch_sens.csv")
        )

        df_phonesens1 = df_phonesens1.sort_values("timestamp")
        df_watchsens1 = df_watchsens1.sort_values("timestamp")
        df_phonesens2 = df_phonesens2.sort_values("timestamp")
        df_watchsens2 = df_watchsens2.sort_values("timestamp")

        df1_1 = pd.merge_asof(
            df_watchsens1, df_phonesens1, on="timestamp", direction="backward"
        )
        df1_2 = pd.merge_asof(
            df_phonesens1, df_watchsens1, on="timestamp", direction="backward"
        )
        df2_1 = pd.merge_asof(
            df_watchsens2, df_phonesens2, on="timestamp", direction="backward"
        )
        df2_2 = pd.merge_asof(
            df_phonesens2, df_watchsens2, on="timestamp", direction="backward"
        )

        df1_merged = (
            df1_1.append(df1_2)
            .sort_values("timestamp")
            .drop_duplicates(subset="timestamp")
        )
        df2_merged = (
            df2_1.append(df2_2)
            .sort_values("timestamp")
            .drop_duplicates(subset="timestamp")
        )

        df1_merged = df1_merged.fillna(0)
        df2_merged = df2_merged.fillna(0)

        dep_1 = df_timestamp1[0].values
        arr_1 = df_timestamp1[1].values
        pos_1 = df_timestamp1[2].values

        def f1(x):
            for d, a, p in zip(dep_1, arr_1, pos_1):
                if (d <= x) and (x <= a):
                    return p
            return p

        dep_2 = df_timestamp2[0].values
        arr_2 = df_timestamp2[1].values
        pos_2 = df_timestamp2[2].values

        def f2(x):
            for d, a, p in zip(dep_2, arr_2, pos_2):
                if (d <= x) and (x <= a):
                    return p
            return p

        timestamp1 = df1_merged["timestamp"].values
        posid1 = [f1(t) for t in timestamp1]
        timestamp2 = df2_merged["timestamp"].values
        posid2 = [f2(t) for t in timestamp2]

        df1_merged["PosID"] = posid1
        df2_merged["PosID"] = posid2

        df_merged = pd.concat([df1_merged, df2_merged])

        X = df_merged.drop(["timestamp", "PosID"], axis=1).values
        df_merged["PosID"] = LabelEncoder().fit_transform(
            df_merged["PosID"].values)
        y = df_merged["PosID"].values

    elif parsed_args.dataset_type == "fars":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "fars.arff"), skiprows=37, header=None
        )
        if parsed_args.num_samples != -1:
            df = df.sample(parsed_args.num_samples)
        X = df.drop(29, axis=1).values
        y = LabelEncoder().fit_transform(df[29].values)

    elif parsed_args.dataset_type == "asteroids":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "dataset"), skiprows=40, header=None
        )
        df = df.dropna()
        if parsed_args.num_samples != -1:
            df = df.sample(parsed_args.num_samples)
        df[2] = LabelEncoder().fit_transform(df[2].values)
        df[33] = LabelEncoder().fit_transform(df[33].values)
        df[4] = df[4].apply(lambda x: -1 if "?" == x else float(x))
        df[6] = df[6].apply(lambda x: -1 if "?" == x else float(x))
        X = df.drop([0, 1, 33], axis=1).values
        y = df[33].values

    elif parsed_args.dataset_type == "brich1":
        X = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "birch1.txt"),
            delim_whitespace=True,
            header=None,
        )[[0, 1]].values
        y = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "b1-gt.pa"),
            delim_whitespace=True,
            header=None,
            skiprows=4,
        )[0].values

    elif parsed_args.dataset_type == "diabetes":
        df = pd.read_csv(os.path.join(
            parsed_args.path_to_dir, "diabetic_data.csv"))

        df["readmitted"] = LabelEncoder().fit_transform(df["readmitted"])
        df["age"] = df["age"].apply(lambda x: int(x.split("-")[0][1:]))
        df["weight"] = df["weight"].apply(lambda x: "[250-" if x == "?" else x)
        df["weight"] = df["weight"].apply(
            lambda x: "[200-" if x == ">200" else x)
        df["weight"] = df["weight"].apply(lambda x: int(x.split("-")[0][1:]))

        df = df.drop(
            [
                "citoglipton",
                "examide",
                "payer_code",
                "medical_specialty",
                "encounter_id",
                "patient_nbr",
            ],
            axis=1,
        )

        four_status_cols = [
            "metformin",
            "repaglinide",
            "nateglinide",
            "chlorpropamide",
            "glimepiride",
            "glipizide",
            "glyburide",
            "pioglitazone",
            "rosiglitazone",
            "acarbose",
            "miglitol",
            "insulin",
            "glyburide-metformin",
            "tolazamide",
            "metformin-pioglitazone",
            "metformin-rosiglitazone",
            "glimepiride-pioglitazone",
            "glipizide-metformin",
            "troglitazone",
            "tolbutamide",
            "acetohexamide",
        ]
        four_maps = {"Down": 0, "No": -1, "Steady": 1, "Up": 2}
        for c in four_status_cols:
            df[c] = df[c].apply(lambda x: four_maps[x])

        df["change"] = df["change"].replace("Ch", 1)
        df["change"] = df["change"].replace("No", 0)
        df["gender"] = df["gender"].replace("Male", 1)
        df["gender"] = df["gender"].replace("Female", 0)
        df["gender"] = df["gender"].replace("Unknown/Invalid", -1)
        df["diabetesMed"] = df["diabetesMed"].replace("Yes", 1)
        df["diabetesMed"] = df["diabetesMed"].replace("No", 0)

        df["A1Cresult"] = df["A1Cresult"].replace(">7", 1)
        df["A1Cresult"] = df["A1Cresult"].replace(">8", 2)
        df["A1Cresult"] = df["A1Cresult"].replace("Norm", 0)
        df["A1Cresult"] = df["A1Cresult"].replace("None", -1)
        df["max_glu_serum"] = df["max_glu_serum"].replace(">200", 1)
        df["max_glu_serum"] = df["max_glu_serum"].replace(">300", 2)
        df["max_glu_serum"] = df["max_glu_serum"].replace("Norm", 0)
        df["max_glu_serum"] = df["max_glu_serum"].replace("None", -1)

        df["diag_1"] = df["diag_1"].apply(lambda x: "-1" if "?" == x else x)
        df["diag_2"] = df["diag_2"].apply(lambda x: "-1" if "?" == x else x)
        df["diag_3"] = df["diag_3"].apply(lambda x: "-1" if "?" == x else x)
        df["diag_1"] = df["diag_1"].apply(
            lambda x: 0 if ("V" in x) or ("E" in x) else x
        )
        df["diag_2"] = df["diag_2"].apply(
            lambda x: 0 if ("V" in x) or ("E" in x) else x
        )
        df["diag_3"] = df["diag_3"].apply(
            lambda x: 0 if ("V" in x) or ("E" in x) else x
        )
        df["diag_1"] = df["diag_1"].replace("?", -1).astype(float)
        df["diag_2"] = df["diag_2"].replace("?", -1).astype(float)
        df["diag_3"] = df["diag_3"].replace("?", -1).astype(float)

        col_alloc_origin = sampling_col_alloc(
            col_num=df.shape[1] - 1,
            feature_num_ratio_of_active_party=parsed_args.feature_num_ratio_of_active_party,
            feature_num_ratio_of_passive_party=parsed_args.feature_num_ratio_of_passive_party,
        )
        X_d = df.drop("readmitted", axis=1)
        X_a = pd.get_dummies(
            X_d[X_d.columns[col_alloc_origin[0]]], drop_first=True)
        X_p = pd.get_dummies(
            X_d[X_d.columns[col_alloc_origin[1]]], drop_first=True)
        col_alloc = [
            list(range(X_a.shape[1])),
            list(range(X_a.shape[1], X_a.shape[1] + X_p.shape[1])),
        ]
        X = pd.concat([X_a, X_p], axis=1).values
        y = df["readmitted"].values

    else:
        raise ValueError(f"{parsed_args.dataset_type} is not supported.")

    mm = preprocessing.MinMaxScaler()
    X_minmax = mm.fit_transform(X)
    selector = SelectKBest(mutual_info_classif, k=X.shape[1] * 0.5)
    selector.fit(X_minmax, y)

    np.save(
        os.path.join(
            parsed_args.path_to_dir,
            f"{parsed_args.dataset_type}_fti",
        ),
        selector.scores_,
    )
