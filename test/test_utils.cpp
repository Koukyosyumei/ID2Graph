#include <vector>
#include <cassert>
#include <iostream>
#include "llatvfl/utils/utils.h"
#include "gtest/gtest.h"

TEST(Utils, QuantileTest)
{
    vector<float> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    vector<float> quartiles = Quantile<float>(in, {0.25, 0.5, 0.75});
    vector<float> test_quartiles = {3.25, 6, 8.75};
    for (int i = 0; i < quartiles.size(); i++)
    {
        ASSERT_EQ(quartiles[i], test_quartiles[i]);
    }
}

TEST(Utils, NumPartiesTest)
{
    vector<int> num_parties_per_process = get_num_parties_per_process(3, 8);
    vector<int> test_num_parties_per_process = {3, 3, 2};
    for (int i = 0; i < test_num_parties_per_process.size(); i++)
    {
        ASSERT_EQ(num_parties_per_process[i], test_num_parties_per_process[i]);
    }
}