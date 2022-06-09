#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <numeric>
#include <string>
#include <cassert>
#include <unistd.h>
#include "secureboost/attack.h"
#include "secureboost/metric.h"
using namespace std;

const int min_leaf = 1;
const int depth = 3;
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
bool use_missing_value = false;

void parse_args(int argc, char *argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "f:p:r:c:m")) != -1)
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
            boosting_rounds = int(*optarg);
            break;
        case 'c':
            completelly_secure_round = int(*optarg);
            break;
        case 'm':
            use_missing_value = true;
            break;
        default:
            printf("unknown parameter %s is specified", optarg);
            printf("Usage: %s [-f] [-p] [-r] [-c] [-m] ...\n", argv[0]);
            break;
        }
        cout << 1 << endl;
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
    vector<Party> parties(num_party);

    printf("Loading datasets ...\n");
    printf("train size is %d, column size is %d, party size is %d\n", num_row_train, num_col, num_party);
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
        Party party(x, feature_idxs, i, min_leaf, subsample_cols, max_bin, use_missing_value);
        parties[i] = party;
    }
    for (int j = 0; j < num_row_train; j++)
        scanf("%lf", &y_train[j]);

    scanf("%d", &num_row_val);
    printf("%d\n", num_row_val);
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

    printf("num of nan is %d\n", num_nan_cell);

    // --- Check Initialization --- //
    SecureBoostClassifier clf = SecureBoostClassifier(subsample_cols,
                                                      min_child_weight,
                                                      depth, min_leaf,
                                                      learning_rate,
                                                      boosting_rounds,
                                                      lam, const_gamma, eps,
                                                      0, completelly_secure_round,
                                                      0.5, true);

    printf("Training ...\n");
    clf.fit(parties, y_train);

    for (int i = 0; i < clf.estimators.size(); i++)
    {
        printf("Tree-%d: %lf\n", i + 1, clf.estimators[i].get_leaf_purity());
        printf("%s\n", clf.estimators[i].print(true, true).c_str());
    }

    for (int p = 0; p < num_party; p++)
    {
        printf("lookup talbe of party_id = %d\n", p);
        for (int i = 0; i < parties[p].lookup_table.size(); i++)
            printf("%d: %d, %lf, %d\n", i,
                   get<0>(parties[p].lookup_table.at(i)),
                   get<1>(parties[p].lookup_table.at(i)),
                   get<2>(parties[p].lookup_table.at(i)));
    }

    printf("Evaluating ...\n");
    vector<double> predict_proba_train = clf.predict_proba(X_train);
    vector<int> y_true_train(y_train.begin(), y_train.end());
    printf("Train AUC: %lf\n", roc_auc_score(predict_proba_train, y_true_train));
    vector<double> predict_proba_val = clf.predict_proba(X_val);
    vector<int> y_true_val(y_val.begin(), y_val.end());
    printf("Val AUC: %lf\n", roc_auc_score(predict_proba_val, y_true_val));

    std::ofstream adj_mat_file;
    string filepath = folderpath + "/" + fileprefix + "_adj_mat.txt";
    adj_mat_file.open(filepath, std::ios::out);
    vector<vector<vector<int>>> vec_adi_mat = extract_adjacency_matrix_from_forest(&clf, 1, false);
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
