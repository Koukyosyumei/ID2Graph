#include <iostream>
#include <limits>
#include <vector>
#include <cassert>
#include "secureboost/secureboost.h"
#include "secureboost/metric.h"
using namespace std;

const int min_leaf = 1;
const int depth = 3;
const double learning_rate = 0.1;
const int boosting_rounds = 5;
const double lam = 1.0;
const double const_gamma = 0.0;
const double eps = 1.0;
const double min_child_weight = -1 * numeric_limits<double>::infinity();
const double subsample_cols = 1.0;

int main()
{
    // --- Load Data --- //
    int num_row_train, num_row_val, num_col, num_party;
    cin >> num_row_train >> num_col >> num_party;
    vector<vector<double>> X_train(num_row_train, vector<double>(num_col));
    vector<double> y_train(num_row_train);
    vector<double> y_val(num_row_val);
    vector<Party> parties(num_party);

    cout << "Loading datasets..." << endl;
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
                X_train[k][temp_count_feature] = x[k][j];
            }
            temp_count_feature += 1;
        }
        Party party(x, feature_idxs, i, min_leaf, subsample_cols);
        parties[i] = party;
    }

    for (int j = 0; j < num_row_train; j++)
        cin >> y_train[j];

    cin >> num_row_val;
    vector<vector<double>> X_val(num_row_val, vector<double>(num_col));
    for (int i = 0; i < num_col; i++)
    {
        for (int j = 0; j < num_row_val; j++)
        {
            cin >> X_val[j][i];
        }
    }

    for (int j = 0; j < num_row_val; j++)
        cin >> y_val[j];

    // --- Check Initialization --- //
    SecureBoostClassifier clf = SecureBoostClassifier(subsample_cols,
                                                      min_child_weight,
                                                      depth, min_leaf,
                                                      learning_rate,
                                                      boosting_rounds,
                                                      lam, const_gamma, eps,
                                                      0, true);

    cout << "Training..." << endl;
    clf.fit(parties, y_train);

    for (int i = 0; i < clf.estimators.size(); i++)
    {
        cout << "Tree-" << i + 1 << endl;
        cout << clf.estimators[i].get_root_node().print(true, true) << endl;
    }

    vector<double> predict_proba = clf.predict_proba(X_train);
    vector<int> y_true(y_train.begin(), y_train.end());
    cout << roc_auc_score(predict_proba, y_true) << endl;

    // cout << temp_party[0].get_lookup_table().size() << endl;
    // cout << parties[0].get_lookup_table().size() << endl;

    // --- Check Training --- //
    // clf.fit(parties, y);

    // cout << clf.estimators[0].get_root_node().print() << endl;
    /*
    vector<double> predict_proba = clf.predict_proba(X);
    for (int i = 0; i < predict_proba.size(); i++)
    {
        cout << predict_proba[i] << " ";
    }
    */
}
