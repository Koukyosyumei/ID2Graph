#include <vector>
#include <cassert>
#include <iostream>
#include "llatvfl/paillier/paillier.h"
#include "gtest/gtest.h"
using namespace std;

TEST(paillier, PaillierBaseTest)
{
    long long p = 3;
    long long q = 5;
    long long n = p * q;
    long long k = 4;
    long long g = (1 + k * n) % (n * n);

    ASSERT_EQ(L(g, n), k);

    boost::random::mt19937 mt(42);
    PaillierPublicKey pk = PaillierPublicKey(n, g, mt);
    PaillierSecretKey sk = PaillierSecretKey(p, q, n, g);
    ASSERT_EQ(sk.lam, 4);
    ASSERT_EQ(sk.mu, 1);

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

TEST(paillier, PaillierKeyGeneratorTest)
{
    PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
    pair<PaillierPublicKey, PaillierSecretKey> keypair = keygenerator.generate_keypair();
    PaillierPublicKey pk = keypair.first;
    PaillierSecretKey sk = keypair.second;

    PaillierCipherText ct_1 = pk.encrypt(1);
    ASSERT_EQ(sk.decrypt(ct_1), 1);
    PaillierCipherText ct_2 = pk.encrypt(2);
    ASSERT_EQ(sk.decrypt(ct_2), 2);
    PaillierCipherText ct_3 = pk.encrypt(123456);
    ASSERT_EQ(sk.decrypt(ct_3), 123456);
    PaillierCipherText ct_4 = ct_3 * 2;
    ASSERT_EQ(sk.decrypt(ct_4), 246912);
    PaillierCipherText ct_5 = ct_4 + ct_2;
    ASSERT_EQ(sk.decrypt(ct_5), 246914);
}

TEST(paillier, PaillierKeyRingTest)
{
    PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
    pair<PaillierPublicKey, PaillierSecretKey> keypair_1 = keygenerator.generate_keypair();
    PaillierPublicKey pk_1 = keypair_1.first;
    PaillierSecretKey sk_1 = keypair_1.second;
    pair<PaillierPublicKey, PaillierSecretKey> keypair_2 = keygenerator.generate_keypair();
    PaillierPublicKey pk_2 = keypair_2.first;
    PaillierSecretKey sk_2 = keypair_2.second;

    ASSERT_TRUE(!(pk_1 == pk_2));

    PaillierKeyRing keyring = PaillierKeyRing();
    keyring.add(sk_1);
    keyring.add(sk_2);

    PaillierCipherText ct_1 = pk_1.encrypt(34567);
    PaillierCipherText ct_2 = pk_2.encrypt(56789);
    ASSERT_EQ(keyring.decrypt(ct_1), 34567);
    ASSERT_EQ(keyring.decrypt(ct_2), 56789);
}
