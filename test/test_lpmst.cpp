#include <cassert>
#include <iostream>
#include "llatvfl/lpmst/lpmst.h"
#include "gtest/gtest.h"
using namespace std;

TEST(LPMST, RRPTest)
{
    vector<float> prior = {0.1, 0.3, 0.2, 0.4};
    float eps = 1.0;
    RRWithPrior rrp = RRWithPrior(eps, prior);

    ASSERT_EQ(rrp.K, 4);
    ASSERT_EQ(rrp.k_star, 3);
    ASSERT_NEAR(rrp.threshold_prob, 0.5761168847658291, 1e-6);
    ASSERT_NEAR(rrp.best_w_k, 0.5185051962892462, 1e-6);
    ASSERT_LT(rrp.rrtop_k(0), 4);
    ASSERT_GE(rrp.rrtop_k(0), 0);
}