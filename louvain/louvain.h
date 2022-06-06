#include <iostream>
#include <limits>
#include <map>
#include <vector>
#include "community.h"
using namespace std;

struct Louvain
{
    double precision;
    int ndp;
    int max_itr;

    Community c;
    Graph g;

    Louvain(int max_itr_ = 1000, double precision_ = 0.000001, int ndp_ = -1)
    {
        max_itr = max_itr_;
        precision = precision_;
        ndp = ndp_;
    }

    void fit(Graph gc)
    {
        bool improvement = true;

        g = gc;
        c = Community(gc, ndp, precision);
        double mod = c.modularity(), new_mod;

        for (int i = 0; i < max_itr; i++)
        {
            improvement = c.one_level();
            new_mod = c.modularity();
            g = c.partition2graph_binary();
            c = Community(g, -1, precision);
            mod = new_mod;
            if (!improvement)
            {
                break;
            }
        }
    }
};