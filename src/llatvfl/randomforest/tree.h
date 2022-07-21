#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include <random>
#include "../core/tree.h"
#include "node.h"
#include "party.h"

struct RandomForestTree : Tree<RandomForestNode>
{
    RandomForestTree() {}
    void fit(vector<RandomForestParty> *parties, vector<float> &y,
             int min_leaf, int depth, float max_samples_ratio = 1.0,
             float weight_entropy = 0.0, float max_leaf_purity = 1.0,
             int active_party_id = -1, int n_job = 1, int seed = 0)
    {
        vector<int> idxs(y.size());
        iota(idxs.begin(), idxs.end(), 0);

        if (max_samples_ratio < 1.0)
        {
            mt19937 engine(seed);
            shuffle(idxs.begin(), idxs.end(), engine);
            int temp_subsampled_size = int(max_samples_ratio * float(y.size()));
            idxs.resize(temp_subsampled_size);
        }

        for (int i = 0; i < parties->size(); i++)
        {
            parties->at(i).subsample_columns();
        }
        dtree = RandomForestNode(parties, y, idxs, depth, weight_entropy, max_leaf_purity, active_party_id, n_job);
    }
};
