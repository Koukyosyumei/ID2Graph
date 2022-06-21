#include <iostream>
#include <algorithm>
#include <limits>
#include <vector>
#include <cassert>
#include "../src/llatvfl/attack/attack.h"
using namespace std;

const int min_leaf = 1;
const int depth = 2;
const int num_trees = 1;
const double subsample_cols = 1.0;
const double max_samples_ratio = 1.0;

int main()
{
    // --- Load Data --- //
    int num_row, num_col, num_party;
    cin >> num_row >> num_col >> num_party;
    vector<double> y(num_row);
    vector<vector<double>> X(num_row, vector<double>(num_col));
    vector<RandomForestParty> parties(num_party);

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
        RandomForestParty party(x, feature_idxs, i, min_leaf, subsample_cols);
        parties[i] = party;
    }

    for (int j = 0; j < num_row; j++)
        cin >> y[j];

    // --- Check Initialization --- //
    RandomForestClassifier clf = RandomForestClassifier(subsample_cols, depth, min_leaf,
                                                        max_samples_ratio, num_trees, -1, 1);

    // --- Check Training --- //
    clf.fit(parties, y);

    assert(clf.estimators[0].dtree.giniimp == 0.46875);
    assert(clf.estimators[0].dtree.score == 0.16875);
    assert(clf.estimators[0].dtree.best_party_id == 0);
    assert(clf.estimators[0].dtree.best_col_id == 0);
    assert(clf.estimators[0].dtree.best_threshold_id == 2);

    assert(clf.estimators[0].dtree.party_id == 0);
    assert(get<0>(clf.estimators[0].dtree.parties->at(
                                                     clf.estimators[0].dtree.party_id)
                      .lookup_table.at(clf.estimators[0].dtree.record_id)) == 0);
    assert(get<1>(clf.estimators[0].dtree.parties->at(
                                                     clf.estimators[0].dtree.party_id)
                      .lookup_table.at(clf.estimators[0].dtree.record_id)) == 16);

    vector<int> test_idxs_left = {0, 2, 7};
    vector<int> test_idxs_right = {1, 3, 4, 5, 6};
    vector<int> idxs_left = clf.estimators[0].dtree.left->idxs;
    sort(idxs_left.begin(), idxs_left.end());
    assert(idxs_left.size() == test_idxs_left.size());
    for (int i = 0; i < idxs_left.size(); i++)
    {
        assert(idxs_left[i] == test_idxs_left[i]);
    }

    vector<int> idxs_right = clf.estimators[0].dtree.right->idxs;
    sort(idxs_right.begin(), idxs_right.end());
    assert(idxs_right.size() == test_idxs_right.size());
    for (int i = 0; i < idxs_right.size(); i++)
    {
        assert(idxs_right[i] == test_idxs_right[i]);
    }

    assert(clf.estimators[0].dtree.right->depth == 1);
    assert(clf.estimators[0].dtree.left->is_leaf() == 1);
    assert(clf.estimators[0].dtree.right->is_leaf() == 0);

    assert(clf.estimators[0].dtree.right->party_id == 1);
    assert(get<1>(clf.estimators[0].dtree.right->parties->at(
                                                            clf.estimators[0].dtree.right->party_id)
                      .lookup_table.at(clf.estimators[0].dtree.right->record_id)) == 0);

    vector<int> test_idxs_right_left = {3, 6};
    vector<int> test_idxs_right_right = {1, 4, 5};
    vector<int> idxs_right_left = clf.estimators[0].dtree.right->left->idxs;
    vector<int> idxs_right_right = clf.estimators[0].dtree.right->right->idxs;
    sort(idxs_right_left.begin(), idxs_right_left.end());
    sort(idxs_right_right.begin(), idxs_right_right.end());
    for (int i = 0; i < test_idxs_right_left.size(); i++)
    {
        assert(test_idxs_right_left[i] == idxs_right_left[i]);
    }
    for (int i = 0; i < test_idxs_right_right.size(); i++)
    {
        assert(test_idxs_right_right[i] == idxs_right_right[i]);
    }

    vector<double> test_predict_raw = {1, 2.0 / 3.0, 1, 0, 2.0 / 3.0, 2.0 / 3.0, 0, 1};
    vector<double> predict_raw = clf.predict_raw(X);
    for (int i = 0; i < predict_raw.size(); i++)
    {
        assert(test_predict_raw[i] == predict_raw[i]);
    }

    vector<double> test_predict_proba = {0.7310585786300049, 0.6607563732194243,
                                         0.7310585786300049, 0.5,
                                         0.6607563732194243, 0.6607563732194243,
                                         0.5, 0.7310585786300049};
    vector<double> predict_proba = clf.predict_proba(X);
    for (int i = 0; i < predict_proba.size(); i++)
    {
        assert(abs(test_predict_proba[i] - predict_proba[i]) < 1e-6);
    }

    vector<vector<float>> test_adj_mat = {{0, 0, 1, 0, 0, 0, 0, 1},
                                          {0, 0, 0, 0, 1, 1, 0, 0},
                                          {1, 0, 0, 0, 0, 0, 0, 1},
                                          {0, 0, 0, 0, 0, 0, 1, 0},
                                          {0, 1, 0, 0, 0, 1, 0, 0},
                                          {0, 1, 0, 0, 1, 0, 0, 0},
                                          {0, 0, 0, 1, 0, 0, 0, 0},
                                          {1, 0, 1, 0, 0, 0, 0, 0}};

    vector<vector<float>> adj_mat = extract_adjacency_matrix_from_forest(&clf, -1, false).to_densematrix();
    for (int j = 0; j < test_adj_mat.size(); j++)
    {
        for (int k = 0; k < test_adj_mat[j].size(); k++)
        {
            assert(adj_mat[j][k] == test_adj_mat[j][k]);
        }
    }

    cout << "test_randomforest: all passed!" << endl;
}