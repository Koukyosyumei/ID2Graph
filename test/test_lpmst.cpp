#include "../src/llatvfl/lpmst/lpmst.h"
#include <cassert>
#include <iostream>
using namespace std;

int main()
{
    vector<float> prior = {0.1, 0.3, 0.2, 0.4};
    float eps = 1.0;
    RRWithPrior rrp = RRWithPrior(eps, prior);

    assert(rrp.K == 4);
    assert(rrp.k_star == 3);
    assert((rrp.threshold_prob - 0.5761168847658291) < 1e-6);
    assert((rrp.best_w_k - 0.5185051962892462) < 1e-6);
    assert((rrp.rrtop_k(0) < 4));
    assert((rrp.rrtop_k(0) >= 0));

    cout << "test_lpmst: all passed!" << endl;
}