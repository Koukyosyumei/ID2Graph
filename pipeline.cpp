#include <iostream>
#include <limits>
#include <vector>
#include <cassert>
#include "secureboost.h"
using namespace std;

const int min_leaf = 1;
const int depth = 3;
const double learning_rate = 0.4;
const int boosting_rounds = 5;
const double lam = 1.0;
const double const_gamma = 0.0;
const double eps = 1.0;
const double min_child_weight = -1 * numeric_limits<double>::infinity();
const double subsample_cols = 1.0;

int main()
{
    // --- Load Data --- //
    cout << "Data Loading" << endl;

    int num_row, num_col, num_party;
    cin >> num_row >> num_col >> num_party;
    vector<double> y(num_row);
    vector<vector<double>> X(num_row, vector<double>(num_col));
    vector<Party> parties(num_party);

    int temp_count_feature = 0;
    for (int i = 0; i < num_party; i++)
    {
        int num_col = 0;
        cin >> num_col;
        vector<int> feature_idxs(num_col);
        vector<vector<double>> x(num_row, vector<double>(num_col));
        for (int j = 0; j < num_col; j++)
        {
            feature_idxs[j] = temp_count_feature;
            for (int k = 0; k < num_row; k++)
            {
                cin >> x[k][j];
                X[k][temp_count_feature] = x[k][j];
            }
            temp_count_feature += 1;
        }
        Party party(x, feature_idxs, i, min_leaf, subsample_cols);
        parties[i] = party;
    }

    for (int j = 0; j < num_row; j++)
        cin >> y[j];

    // --- Check Initialization --- //
    SecureBoostClassifier clf = SecureBoostClassifier(subsample_cols,
                                                      min_child_weight,
                                                      depth, min_leaf,
                                                      learning_rate,
                                                      boosting_rounds,
                                                      lam, const_gamma, eps);

    // --- Check Training --- //
    clf.fit(parties, y);

    cout << clf.estimators[0].get_root_node().print() << endl;
    /*
    vector<double> predict_proba = clf.predict_proba(X);
    for (int i = 0; i < predict_proba.size(); i++)
    {
        cout << predict_proba[i] << " ";
    }
    */
}
