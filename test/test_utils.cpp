#include <vector>
#include <cassert>
#include <iostream>
#include <random>
#include "llatvfl/utils/utils.h"
#include "llatvfl/utils/prime.h"
#include "llatvfl/paillier/paillier.h"
#include "gtest/gtest.h"
using namespace std;

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

TEST(utils, GCDTest)
{
    ASSERT_EQ(gcd(12, 42), 6);
    ASSERT_EQ(gcd(42, 12), 6);
}

TEST(utils, LCMTest)
{
    ASSERT_EQ(lcm(3, 4), 12);
    ASSERT_EQ(lcm(4, 3), 12);
}

TEST(utils, ModPowTest)
{
    ASSERT_EQ(modpow(17, 20, 17345), 13896);
    ASSERT_EQ(modpow(23, 19, 1), 0);
}

TEST(utils, MillerRabinPrimalityTest)
{
    mt19937 mt(42);
    ASSERT_TRUE(miller_rabin_primality_test(2, mt));
    ASSERT_TRUE(miller_rabin_primality_test(3, mt));
    ASSERT_TRUE(miller_rabin_primality_test(5, mt));
    ASSERT_TRUE(miller_rabin_primality_test(1223, mt));
    ASSERT_TRUE(miller_rabin_primality_test(9973, mt));
    ASSERT_TRUE(miller_rabin_primality_test(99991, mt));
    ASSERT_TRUE(miller_rabin_primality_test(524287, mt));
    ASSERT_TRUE(miller_rabin_primality_test(2147483647, mt));
    ASSERT_TRUE(!miller_rabin_primality_test(0, mt));
    ASSERT_TRUE(!miller_rabin_primality_test(1, mt));
    ASSERT_TRUE(!miller_rabin_primality_test(99991 * 9973, mt));
    ASSERT_TRUE(!miller_rabin_primality_test(1234567892, mt));
    ASSERT_TRUE(!miller_rabin_primality_test(12345678900, mt));
    ASSERT_TRUE(!miller_rabin_primality_test(75361, mt));
    ASSERT_TRUE(!miller_rabin_primality_test(512461, mt));
    ASSERT_TRUE(!miller_rabin_primality_test(1565912117761, mt));
}
