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

    ASSERT_TRUE(ct_1.pk == ct_2.pk);
    ASSERT_TRUE(!(ct_1.pk != ct_2.pk));

    PaillierCipherText ct_3 = ct_1 + ct_2;
    ASSERT_TRUE(ct_1.pk == ct_3.pk);
    ASSERT_EQ(sk.decrypt(ct_3), 11);

    PaillierCipherText ct_4 = ct_1 + 0;
    ASSERT_TRUE(ct_1.pk == ct_4.pk);
    ASSERT_EQ(sk.decrypt(ct_4), 3);

    PaillierCipherText ct_5 = ct_1 + 3;
    ASSERT_TRUE(ct_1.pk == ct_5.pk);
    ASSERT_EQ(sk.decrypt(ct_5), 6);

    PaillierCipherText ct_6 = ct_1 * 0;
    ASSERT_TRUE(ct_1.pk == ct_6.pk);
    ASSERT_EQ(sk.decrypt(ct_6), 0);

    PaillierCipherText ct_7 = ct_1 * 1;
    ASSERT_TRUE(ct_1.pk == ct_7.pk);
    ASSERT_EQ(sk.decrypt(ct_7), 3);

    PaillierCipherText ct_8 = ct_1 * 3;
    ASSERT_TRUE(ct_1.pk == ct_8.pk);
    ASSERT_EQ(sk.decrypt(ct_8), 9);
}