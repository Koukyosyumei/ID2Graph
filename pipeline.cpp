#include <iostream>
#include <limits>
#include <vector>
#include <cassert>
#include "secureboost/secureboost.h"
#include "secureboost/metric.h"
using namespace std;

const int min_leaf = 1;
const int depth = 3;
const int max_bin = 256;
const double learning_rate = 0.4;
const int boosting_rounds = 2;
const double lam = 1.0;
const double const_gamma = 0.0;
const double eps = 1.0;
const double min_child_weight = -1 * numeric_limits<double>::infinity();
const double subsample_cols = 0.8;
const bool use_missing_value = false;

int main()
{
    // --- Load Data --- //
    int num_row_train, num_row_val, num_col, num_party;
    int num_nan_cell = 0;
    cin >> num_row_train >> num_col >> num_party;
    vector<vector<double>> X_train(num_row_train, vector<double>(num_col));
    vector<double> y_train(num_row_train);
    vector<Party> parties(num_party);

    cout << "Loading datasets ..." << endl;
    cout << num_row_train << " " << num_col << " " << num_party << endl;
    int temp_count_feature = 0;
    for (int i = 0; i < num_party; i++)
    {
        int num_col = 0;
        cin >> num_col;
        vector<int> feature_idxs(num_col);
        vector<vector<double>> x(num_row_train, vector<double>(num_col));
        for (int j = 0; j < num_col; j++)
        {
            feature_idxs[j] = temp_count_feature;
            for (int k = 0; k < num_row_train; k++)
            {
                cin >> x[k][j];
                if (use_missing_value && x[k][j] == -9999)
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
        cin >> y_train[j];

    cin >> num_row_val;
    cout << num_row_val << endl;
    vector<vector<double>> X_val(num_row_val, vector<double>(num_col));
    vector<double> y_val(num_row_val);
    for (int i = 0; i < num_col; i++)
    {
        for (int j = 0; j < num_row_val; j++)
        {
            cin >> X_val[j][i];
            if (use_missing_value && X_val[j][i] == -9999)
            {
                X_val[j][i] = nan("");
            }
        }
    }
    for (int j = 0; j < num_row_val; j++)
        cin >> y_val[j];

    cout << "num of nan is " << num_nan_cell << endl;

    // --- Check Initialization --- //
    SecureBoostClassifier clf = SecureBoostClassifier(subsample_cols,
                                                      min_child_weight,
                                                      depth, min_leaf,
                                                      learning_rate,
                                                      boosting_rounds,
                                                      lam, const_gamma, eps,
                                                      0, true, 0.5);

    cout << "Training ..." << endl;
    clf.fit(parties, y_train);

    for (int i = 0; i < clf.estimators.size(); i++)
    {
        cout << "Tree-" << i + 1 << endl;
        cout << clf.estimators[i].print(true, true) << endl;
    }

    for (int p = 0; p < num_party; p++)
    {
        cout << "lookup talbe of party_id = " << p << endl;
        for (int i = 0; i < parties[p].lookup_table.size(); i++)
            cout << i << ": " << get<0>(parties[p].lookup_table.at(i)) << ", "
                 << get<1>(parties[p].lookup_table.at(i)) << ", "
                 << get<2>(parties[p].lookup_table.at(i)) << endl;
        cout << endl;
    }

    cout << "Evaluating ..." << endl;
    vector<double> predict_proba_train = clf.predict_proba(X_train);
    vector<int> y_true_train(y_train.begin(), y_train.end());
    cout << "Train AUC: " << roc_auc_score(predict_proba_train, y_true_train) << endl;
    vector<double> predict_proba_val = clf.predict_proba(X_val);
    vector<int> y_true_val(y_val.begin(), y_val.end());
    cout << "Val AUC: " << roc_auc_score(predict_proba_val, y_true_val) << endl;
}
