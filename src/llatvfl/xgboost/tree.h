#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include "../core/tree.h"
#include "node.h"

struct XGBoostTree : Tree<XGBoostNode>
{
    XGBoostTree() {}
    void fit(vector<XGBoostParty> *parties, vector<float> &y,
             vector<float> &gradient, vector<float> &hessian,
             float min_child_weight, float lam, float gamma, float eps,
             int min_leaf, int depth, float weight_entropy, float max_leaf_purity,
             int active_party_id = -1, bool use_only_active_party = false, int n_job = 1)
    {
        vector<int> idxs(y.size());
        iota(idxs.begin(), idxs.end(), 0);
        for (int i = 0; i < parties->size(); i++)
        {
            parties->at(i).subsample_columns();
        }
        dtree = XGBoostNode(parties, y, gradient, hessian, idxs,
                            min_child_weight, lam, gamma, eps, depth,
                            weight_entropy, max_leaf_purity, active_party_id, use_only_active_party, n_job);
    }
};
