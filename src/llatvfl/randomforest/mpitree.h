#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include "../core/tree.h"
#include "mpinode.h"

struct MPIRandomForestTree : Tree<MPIRandomForestNode>
{
    MPIRandomForestTree() {}
    void fit(MPIRandomForestParty *active_party, int parties_num,
             int min_leaf, int depth, float max_samples_ratio = 1.0,
             int active_party_id = 0, bool use_only_active_party = false, int seed = 0)
    {
        vector<int> idxs(active_party->y.size());
        iota(idxs.begin(), idxs.end(), 0);

        if (max_samples_ratio < 1.0)
        {
            mt19937 engine(seed);
            shuffle(idxs.begin(), idxs.end(), engine);
            int temp_subsampled_size = int(max_samples_ratio * float(active_party->y.size()));
            idxs.resize(temp_subsampled_size);
        }

        active_party->set_instance_space(idxs);
        active_party->subsample_columns();
        dtree = MPIRandomForestNode(active_party, parties_num, idxs,
                                    depth, depth, active_party_id,
                                    use_only_active_party);
    }
};