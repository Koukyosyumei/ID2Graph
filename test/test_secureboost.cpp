#include <iostream>
#include <limits>
#include <vector>
#include <cassert>
//#include "../secureboost/secureboost.h"
#include "../src/llatvfl/xgboost/attack.h"
using namespace std;

const int min_leaf = 1;
const int depth = 3;
const double learning_rate = 0.4;
const int boosting_rounds = 2;
const double lam = 1.0;
const double const_gamma = 0.0;
const double eps = 1.0;
const double min_child_weight = -1 * numeric_limits<double>::infinity();
const double subsample_cols = 1.0;

int main()
{
    // --- Load Data --- //
    int num_row, num_col, num_party;
    cin >> num_row >> num_col >> num_party;
    vector<double> y(num_row);
    vector<vector<double>> X(num_row, vector<double>(num_col));
    vector<XGBoostParty> parties(num_party);

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
        XGBoostParty party(x, feature_idxs, i, min_leaf, subsample_cols);
        parties[i] = party;
    }

    for (int j = 0; j < num_row; j++)
        cin >> y[j];

    // --- Check Initialization --- //
    XGBoostClassifier clf = XGBoostClassifier(subsample_cols,
                                              min_child_weight,
                                              depth, min_leaf,
                                              learning_rate,
                                              boosting_rounds,
                                              lam, const_gamma, eps,
                                              -1, 0, 1.0, 2);

    vector<double> test_init_pred = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    vector<double> init_pred = clf.get_init_pred(y);
    for (int i = 0; i < init_pred.size(); i++)
        assert(init_pred[i] == test_init_pred[i]);

    vector<double> base_pred;
    copy(init_pred.begin(), init_pred.end(), back_inserter(base_pred));
    vector<double> test_base_grad = {-0.26894, 0.73106, -0.26894, 0.73106,
                                     -0.26894, -0.26894, 0.73106, -0.26894};
    vector<double> grad = clf.get_grad(base_pred, y);
    for (int i = 0; i < grad.size(); i++)
        assert(abs(grad[i] - test_base_grad[i]) <= 1e-5);

    vector<double> hess = clf.get_hess(base_pred, y);
    vector<double> test_hess = {0.19661, 0.19661, 0.19661, 0.19661,
                                0.19661, 0.19661, 0.19661, 0.19661};
    for (int i = 0; i < hess.size(); i++)
        assert(abs(hess[i] - test_hess[i]) <= 1e-5);

    // --- Check Training --- //
    clf.fit(parties, y);

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

    vector<vector<int>> test_adj_mat_weighed = {{0, 0, 1, 0, 0, 0, 0, 1},
                                                {0, 0, 0, 1, 3, 3, 1, 0},
                                                {1, 0, 0, 0, 0, 0, 0, 1},
                                                {0, 1, 0, 0, 1, 1, 3, 0},
                                                {0, 3, 0, 1, 0, 6, 1, 0},
                                                {0, 3, 0, 1, 6, 0, 1, 0},
                                                {0, 1, 0, 3, 1, 1, 0, 0},
                                                {1, 0, 1, 0, 0, 0, 0, 0}};

    vector<vector<vector<int>>> vec_adj_mat_weighted = extract_adjacency_matrix_from_forest(&clf, -1, true);
    for (int i = 0; i < vec_adj_mat_weighted.size(); i++)
    {
        for (int j = 0; j < vec_adj_mat_weighted[i].size(); j++)
        {
            for (int k = 0; k < vec_adj_mat_weighted[i].size(); k++)
            {
                assert(vec_adj_mat_weighted[i][j][k] == test_adj_mat_weighed[j][k]);
            }
        }
    }

    vector<vector<int>> test_adj_mat = {{0, 0, 1, 0, 0, 0, 0, 1},
                                        {0, 0, 0, 0, 0, 0, 0, 0},
                                        {1, 0, 0, 0, 0, 0, 0, 1},
                                        {0, 0, 0, 0, 0, 0, 1, 0},
                                        {0, 0, 0, 0, 0, 1, 0, 0},
                                        {0, 0, 0, 0, 1, 0, 0, 0},
                                        {0, 0, 0, 1, 0, 0, 0, 0},
                                        {1, 0, 1, 0, 0, 0, 0, 0}};

    vector<vector<vector<int>>> vec_adj_mat = extract_adjacency_matrix_from_forest(&clf, -1, false);
    for (int i = 0; i < vec_adj_mat.size(); i++)
    {
        for (int j = 0; j < vec_adj_mat[i].size(); j++)
        {
            for (int k = 0; k < vec_adj_mat[i].size(); k++)
            {
                assert(vec_adj_mat[i][j][k] == test_adj_mat[j][k]);
            }
        }
    }

    vector<vector<int>> test_adj_mat_1 = {{0, 0, 1, 0, 0, 0, 0, 1},
                                          {0, 0, 0, 0, 1, 1, 0, 0},
                                          {1, 0, 0, 0, 0, 0, 0, 1},
                                          {0, 0, 0, 0, 0, 0, 1, 0},
                                          {0, 1, 0, 0, 0, 1, 0, 0},
                                          {0, 1, 0, 0, 1, 0, 0, 0},
                                          {0, 0, 0, 1, 0, 0, 0, 0},
                                          {1, 0, 1, 0, 0, 0, 0, 0}};

    vector<vector<vector<int>>> vec_adj_mat_1 = extract_adjacency_matrix_from_forest(&clf, 1, false);
    for (int i = 0; i < vec_adj_mat_1.size(); i++)
    {
        for (int j = 0; j < vec_adj_mat_1[i].size(); j++)
        {
            for (int k = 0; k < vec_adj_mat_1[i].size(); k++)
            {
                assert(vec_adj_mat_1[i][j][k] == test_adj_mat_1[j][k]);
            }
        }
    }

    cout << "test_xgboost: all passed!" << endl;
}
