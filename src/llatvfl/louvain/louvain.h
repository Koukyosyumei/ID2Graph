#pragma once
#include <iostream>
#include <limits>
#include <map>
#include <vector>
#include "community.h"
using namespace std;

struct Louvain
{
    float epsilon;
    int max_itr;
    float precision;
    int ndp;
    int seed;
    int verbose;
    int cnt_trials = 0;
    vector<float> epsilon_random_unfolding_candidates;

    Community community;
    Graph g;

    Louvain(float epsilon_ = 1.0, int max_itr_ = 100, float precision_ = 0.000001,
            int ndp_ = -1, int seed_ = 42, int verbose_ = -1)
    {
        epsilon = epsilon_;
        max_itr = max_itr_;
        precision = precision_;
        ndp = ndp_;
        seed = seed_;
        verbose = verbose_;

        epsilon_random_unfolding_candidates.push_back(epsilon_);
        epsilon_random_unfolding_candidates.push_back(0.5);
        epsilon_random_unfolding_candidates.push_back(0.1);
    }

    void reseed(int seed_)
    {
        seed = seed_;
    }

    void reset_epsilon(float epsilon_)
    {
        epsilon = epsilon_;
    }

    void fit(Graph gc)
    {
        reseed(seed + cnt_trials);
        reset_epsilon(epsilon_random_unfolding_candidates[cnt_trials]);
        cnt_trials++;

        bool improvement = true;

        g = gc;
        community = Community(gc, ndp, precision, seed);
        float mod = community.modularity(), new_mod;

        for (int i = 0; i < max_itr; i++)
        {
            improvement = community.one_level(epsilon);
            new_mod = community.modularity();
            g = community.partition2graph_binary();
            community = Community(g, -1, precision, seed);
            mod = new_mod;

            if (verbose > 0 && i % verbose == 0)
            {
                cout << i << ": " << mod << endl;
            }

            if (!improvement)
            {
                break;
            }

            if (i == max_itr - 1)
            {
                cout << "\033[31moptimization did not converge\033[0m" << endl;
            }
        }
    }
};