#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include "../core/tree.h"
#include "node.h"
#include "party.h"

struct RandomForestTree : Tree<RandomForestNode>
{
    RandomForestTree() {}
    void fit(vector<RandomForestParty> *parties, vector<double> y,
             int min_leaf, int depth, int active_party_id = -1, int n_job = 1)
    {
        vector<int> idxs(y.size());
        iota(idxs.begin(), idxs.end(), 0);
        for (int i = 0; i < parties->size(); i++)
        {
            parties->at(i).subsample_columns();
        }
        dtree = RandomForestNode(parties, y, idxs, depth, active_party_id, n_job);
    }
};
