#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include <random>
#include "../core/tree.h"
#include "node_backdoor.h"
#include "party_backdoor.h"

struct RandomForestBackDoorTree : Tree<RandomForestBackDoorNode>
{
    RandomForestBackDoorTree() {}
    void fit(vector<RandomForestBackDoorParty> *parties, vector<float> &y,
             int num_classes, int min_leaf, int depth, vector<float> &prior,
             float max_samples_ratio = 1.0, float mi_delta = 0,
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
        dtree = RandomForestBackDoorNode(parties, y, num_classes, idxs, depth, prior, mi_delta, active_party_id, n_job, true);
    }
};
