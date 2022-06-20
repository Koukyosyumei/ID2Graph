#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <numeric>
#include <string>
#include <cassert>
#include <chrono>
#include <unistd.h>
#include "../src/llatvfl/attack/attack.h"
#include "../src/llatvfl/utils/metric.h"
using namespace std;

const int min_leaf = 1;
const int max_bin = 32;
const double learning_rate = 0.3;
const double lam = 0.0;
const double const_gamma = 0.0;
const double eps = 1.0;
const double min_child_weight = -1 * numeric_limits<double>::infinity();
const double subsample_cols = 0.8;

string folderpath;
string fileprefix;
int boosting_rounds = 20;
int completelly_secure_round = 0;
int depth = 3;
int n_job = 1;
bool use_missing_value = false;
bool is_weighted_graph = false;

void parse_args(int argc, char *argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "f:p:r:c:h:j:mw")) != -1)
    {
        switch (opt)
        {
        case 'f':
            folderpath = string(optarg);
            break;
        case 'p':
            fileprefix = string(optarg);
            break;
        case 'r':
            boosting_rounds = stoi(string(optarg));
            break;
        case 'c':
            completelly_secure_round = stoi(string(optarg));
            break;
        case 'h':
            depth = stoi(string(optarg));
            break;
        case 'j':
            n_job = stoi(string(optarg));
            break;
        case 'm':
            use_missing_value = true;
            break;
        case 'w':
            is_weighted_graph = true;
            break;
        default:
            printf("unknown parameter %s is specified", optarg);
            printf("Usage: %s [-f] [-p] [-r] [-c] [-j] [-m] [-w] ...\n", argv[0]);
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    parse_args(argc, argv);

    // --- Load Data --- //
    int num_row_train, num_row_val, num_col, num_party;
    int num_nan_cell = 0;
    scanf("%d %d %d", &num_row_train, &num_col, &num_party);
    vector<vector<double>> X_train(num_row_train, vector<double>(num_col));
    vector<double> y_train(num_row_train);
    vector<XGBoostParty> parties(num_party);

    int temp_count_feature = 0;
    for (int i = 0; i < num_party; i++)
    {
        int num_col = 0;
        scanf("%d", &num_col);
        vector<int> feature_idxs(num_col);
        vector<vector<double>> x(num_row_train, vector<double>(num_col));
        for (int j = 0; j < num_col; j++)
        {
            feature_idxs[j] = temp_count_feature;
            for (int k = 0; k < num_row_train; k++)
            {
                scanf("%lf", &x[k][j]);
                if (use_missing_value && x[k][j] == -1)
                {
                    x[k][j] = nan("");
                    num_nan_cell += 1;
                }
                X_train[k][temp_count_feature] = x[k][j];
            }
            temp_count_feature += 1;
        }
        XGBoostParty party(x, feature_idxs, i, min_leaf, subsample_cols, max_bin, use_missing_value);
        parties[i] = party;
    }
    for (int j = 0; j < num_row_train; j++)
        scanf("%lf", &y_train[j]);

    scanf("%d", &num_row_val);
    vector<vector<double>> X_val(num_row_val, vector<double>(num_col));
    vector<double> y_val(num_row_val);
    for (int i = 0; i < num_col; i++)
    {
        for (int j = 0; j < num_row_val; j++)
        {
            scanf("%lf", &X_val[j][i]);
            if (use_missing_value && X_val[j][i] == -1)
            {
                X_val[j][i] = nan("");
            }
        }
    }
    for (int j = 0; j < num_row_val; j++)
        scanf("%lf", &y_val[j]);

    std::ofstream result_file;
    string result_filepath = folderpath + "/" + fileprefix + "_result.ans";
    result_file.open(result_filepath, std::ios::out);
    result_file << "train size," << num_row_train << "\n";
    result_file << "val size," << num_row_val << "\n";
    result_file << "column size," << num_col << "\n";
    result_file << "party size," << num_party << "\n";
    result_file << "num of nan," << num_nan_cell << "\n";

    // --- Check Initialization --- //
    XGBoostClassifier clf = XGBoostClassifier(subsample_cols,
                                              min_child_weight,
                                              depth, min_leaf,
                                              learning_rate,
                                              boosting_rounds,
                                              lam, const_gamma, eps,
                                              0, completelly_secure_round,
                                              0.5, n_job, true);

    chrono::system_clock::time_point start, end;
    start = chrono::system_clock::now();
    clf.fit(parties, y_train);
    end = chrono::system_clock::now();
    double elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("%f [ms]\n", elapsed);

    for (int i = 0; i < clf.logging_loss.size(); i++)
    {
        result_file << "round " << i + 1 << ": " << clf.logging_loss[i] << "\n";
    }

    for (int i = 0; i < clf.estimators.size(); i++)
    {
        result_file << "Tree-" << i + 1 << ": " << clf.estimators[i].get_leaf_purity() << "\n";
        result_file << clf.estimators[i].print(true, true).c_str() << "\n";
    }

    vector<double> predict_proba_train = clf.predict_proba(X_train);
    vector<int> y_true_train(y_train.begin(), y_train.end());
    result_file << "Train AUC," << roc_auc_score(predict_proba_train, y_true_train) << "\n";
    vector<double> predict_proba_val = clf.predict_proba(X_val);
    vector<int> y_true_val(y_val.begin(), y_val.end());
    result_file << "Val AUC," << roc_auc_score(predict_proba_val, y_true_val) << "\n";
    result_file.close();

    std::ofstream adj_mat_file;
    string filepath = folderpath + "/" + fileprefix + "_adj_mat.txt";
    adj_mat_file.open(filepath, std::ios::out);
    vector<vector<vector<int>>> vec_adi_mat = extract_adjacency_matrix_from_forest(&clf, 1, is_weighted_graph);
    adj_mat_file << vec_adi_mat.size() << "\n";
    adj_mat_file << vec_adi_mat[0].size() << "\n";
    for (int i = 0; i < vec_adi_mat.size(); i++)
    {
        for (int j = 0; j < vec_adi_mat[i].size(); j++)
        {
            adj_mat_file << count_if(vec_adi_mat[i][j].begin() + j + 1, vec_adi_mat[i][j].end(), [](int x)
                                     { return x != 0; })
                         << " ";
            for (int k = j + 1; k < vec_adi_mat[i].size(); k++)
            {
                if (vec_adi_mat[i][j][k] != 0)
                {
                    adj_mat_file << k << " " << vec_adi_mat[i][j][k] << " ";
                }
            }
            adj_mat_file << "\n";
        }
    }
    adj_mat_file.close();
}
