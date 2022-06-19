#include <iostream>
#include <limits>
#include <vector>
#include <cassert>
#include "../src/llatvfl/attack/attack.h"
using namespace std;

const int min_leaf = 1;
const int depth = 3;
const int num_trees = 1;
const double subsample_cols = 1.0;

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
    RandomForestClassifier clf = RandomForestClassifier(subsample_cols, depth, min_leaf, num_trees,
                                                        -1, 1);

    // --- Check Training --- //
    clf.fit(parties, y);

    vector<double> predict_raw = clf.predict_raw(X);
    for (int i = 0; i < predict_raw.size(); i++)
    {
        cout << predict_raw[i] << " ";
    }
    cout << endl;

    assert(clf.estimators[0].dtree.giniimp == 0.46875);
    assert(clf.estimators[0].dtree.score == 0.16875);
    assert(clf.estimators[0].dtree.best_party_id == 0);
    assert(clf.estimators[0].dtree.best_col_id == 0);
    assert(clf.estimators[0].dtree.best_threshold_id == 2);

    assert(clf.estimators[0].dtree.party_id == 0);
    assert(get<0>(clf.estimators[0].dtree.parties->at(clf.estimators[0].dtree.party_id).lookup_table.at(clf.estimators[0].dtree.record_id)) == 0);
    assert(get<1>(clf.estimators[0].dtree.parties->at(clf.estimators[0].dtree.party_id).lookup_table.at(clf.estimators[0].dtree.record_id)) == 16);

    /*
    assert(parties[0].get_lookup_table().size() == 4);
    assert(parties[1].get_lookup_table().size() == 2);

    assert(clf.estimators[0].dtree.get_num_parties() == 2);

    vector<int> test_idxs_root = {0, 1, 2, 3, 4, 5, 6, 7};
    vector<int> idxs_root = clf.estimators[0].dtree.idxs;
    for (int i = 0; i < idxs_root.size(); i++)
        assert(idxs_root[i] == test_idxs_root[i]);
    assert(clf.estimators[0].dtree.depth == 3);
    assert(get<0>(clf.estimators[0].dtree.parties->at(clf.estimators[0].dtree.party_id).lookup_table.at(clf.estimators[0].dtree.record_id)) == 0);
    assert(get<1>(clf.estimators[0].dtree.parties->at(clf.estimators[0].dtree.party_id).lookup_table.at(clf.estimators[0].dtree.record_id)) == 16);
    assert(clf.estimators[0].dtree.is_leaf() == 0);

    vector<int> test_idxs_left = {0, 2, 7};
    vector<int> idxs_left = clf.estimators[0].dtree.left->idxs;
    for (int i = 0; i < idxs_left.size(); i++)
        assert(idxs_left[i] == test_idxs_left[i]);
    assert(clf.estimators[0].dtree.left->is_pure());
    assert(clf.estimators[0].dtree.left->is_leaf());
    assert(clf.estimators[0].dtree.left->val == 0.5074890528001861);

    vector<int> test_idxs_right = {1, 3, 4, 5, 6};
    vector<int> idxs_right = clf.estimators[0].dtree.right->idxs;
    for (int i = 0; i < idxs_right.size(); i++)
        assert(idxs_right[i] == test_idxs_right[i]);
    assert(!clf.estimators[0].dtree.right->is_pure());
    assert(!clf.estimators[0].dtree.right->is_leaf());
    assert(clf.estimators[0].dtree.right->val == -0.8347166357912786);
    XGBoostNode right_node = *clf.estimators[0].dtree.right;
    assert(right_node.party_id == 1);
    assert(get<0>(right_node.parties->at(right_node.party_id)
                      .lookup_table.at(right_node.record_id)) == 0);

    XGBoostNode right_right_node = *right_node.right;
    assert(right_right_node.party_id == 0);
    assert(get<0>(right_right_node.parties->at(right_right_node.party_id)
                      .lookup_table.at(right_right_node.record_id)) == 0);
    assert(get<1>(right_right_node.parties->at(right_right_node.party_id)
                      .lookup_table.at(right_right_node.record_id)) == 25);

    assert(clf.estimators[0].dtree.right->right->left->is_leaf());
    assert(clf.estimators[0].dtree.right->right->right->is_leaf());
    assert(clf.estimators[0].dtree.right->right->left->val == 0.3860706492904221);
    assert(clf.estimators[0].dtree.right->right->right->val == -0.6109404045885225);

    vector<double> predict_raw = clf.predict_raw(X);
    vector<double> test_predcit_raw = {1.38379341, 0.53207456, 1.38379341,
                                       0.22896408, 1.29495549, 1.29495549,
                                       0.22896408, 1.38379341};
    for (int i = 0; i < test_predcit_raw.size(); i++)
        assert((predict_raw[i] - test_predcit_raw[i]) < 1e-6);

    vector<double> predict_proba = clf.predict_proba(X);
    vector<double> test_predcit_proba = {0.79959955, 0.62996684, 0.79959955,
                                         0.55699226, 0.78498478, 0.78498478,
                                         0.55699226, 0.79959955};
    for (int i = 0; i < test_predcit_proba.size(); i++)
        assert(abs(predict_proba[i] - test_predcit_proba[i]) < 1e-6);
    */

    cout << "test_randomforest: all passed!" << endl;
}