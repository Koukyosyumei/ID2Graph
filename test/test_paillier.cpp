#include <vector>
#include <cassert>
#include <iostream>
#include <random>
#include "llatvfl/paillier/paillier.h"
#include "gtest/gtest.h"
using namespace std;

TEST(utils, PaillierTest)
{
    long long p = 3;
    long long q = 5;
    long long n = p * q;
    long long k = 4;
    long long g = (1 + 4 * n) % (n * n);

    ASSERT_EQ(L(g, n), k);

    mt19937 mt(42);
    PaillierPublicKey pk = PaillierPublicKey(n, g, mt);
    PaillierSecretKey sk = PaillierSecretKey(p, q, n, g);
    ASSERT_EQ(sk.lam, 4);
    ASSERT_EQ(sk.l_g2lam_mod_n2, 1);

    PaillierCipherText ct_1 = pk.encrypt(3);
    ASSERT_EQ(sk.decrypt(ct_1), 3);
    PaillierCipherText ct_2 = pk.encrypt(8);
    ASSERT_EQ(sk.decrypt(ct_2), 8);
}