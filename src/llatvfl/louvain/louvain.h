#pragma once
#include <iostream>
#include <limits>
#include <map>
#include <vector>
#include "community.h"
using namespace std;

struct Louvain
{
    bool random_unforlding;
    int max_itr;
    float precision;
    int ndp;
    int verbose;

    Community community;
    Graph g;

    Louvain(bool random_unforlding_ = false, int max_itr_ = 30, float precision_ = 0.000001, int ndp_ = -1, int verbose_ = -1)
    {
        random_unforlding = random_unforlding_;
        max_itr = max_itr_;
        precision = precision_;
        ndp = ndp_;
        verbose = verbose_;
    }

    void fit(Graph gc)
    {
        bool improvement = true;

        g = gc;
        community = Community(gc, ndp, precision);
        float mod = community.modularity(), new_mod;

        for (int i = 0; i < max_itr; i++)
        {
            improvement = community.one_level(random_unforlding);
            new_mod = community.modularity();
            g = community.partition2graph_binary();
            community = Community(g, -1, precision);
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