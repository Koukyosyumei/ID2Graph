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

    PaillierPublicKey pk = PaillierPublicKey(n, g);
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

TEST(paillier, PaillierEncodingTest)
{
    PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
    pair<PaillierPublicKey, PaillierSecretKey> keypair_1 = keygenerator.generate_keypair();
    PaillierPublicKey pk_1 = keypair_1.first;

    EncodedNumber<int> enc_1 = EncodedNumber<int>(pk_1, 15);
    ASSERT_EQ(0, enc_1.exponent);
    ASSERT_EQ(15, enc_1.encoding);

    EncodedNumber<int> enc_2 = EncodedNumber<int>(pk_1, -15);
    ASSERT_EQ(0, enc_2.exponent);
    ASSERT_EQ(-15, enc_2.encoding);

    EncodedNumber<float> enc_3 = EncodedNumber<float>(pk_1, 15.1);
    ASSERT_NEAR(15.1, pow(enc_3.BASE, enc_3.exponent) * float(enc_3.encoding), 1e-6);
}

TEST(paillier, PaillierDecodingTest)
{
    PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
    pair<PaillierPublicKey, PaillierSecretKey> keypair_1 = keygenerator.generate_keypair();
    PaillierPublicKey pk_1 = keypair_1.first;

    EncodedNumber<int> enc_1 = EncodedNumber<int>(pk_1, 15);
    cout << enc_1.encoding << endl;
    ASSERT_EQ(0, enc_1.exponent);
    ASSERT_EQ(15, enc_1.decode());

    EncodedNumber<int> enc_2 = EncodedNumber<int>(pk_1, -15);
    ASSERT_EQ(0, enc_2.exponent);
    ASSERT_EQ(-15, enc_2.decode());

    long large_positive = 9223372036854775807;
    EncodedNumber<long> enc_3 = EncodedNumber<long>(pk_1, large_positive);
    ASSERT_EQ(0, enc_3.exponent);
    ASSERT_EQ(large_positive, enc_3.decode());

    long large_negative = -9223372036854775807;
    EncodedNumber<long> enc_4 = EncodedNumber<long>(pk_1, large_negative);
    ASSERT_EQ(0, enc_4.exponent);
    ASSERT_EQ(large_negative, enc_4.decode());

    EncodedNumber<float> enc_5 = EncodedNumber<float>(pk_1, 15.1);
    ASSERT_NEAR(15.1, enc_5.decode(), 1e-6);

    EncodedNumber<float> enc_6 = EncodedNumber<float>(pk_1, -15.1);
    ASSERT_NEAR(-15.1, enc_6.decode(), 1e-6);

    double large_postive_double = 123456.123456;
    EncodedNumber<double> enc_7 = EncodedNumber<double>(pk_1, large_postive_double, 1e-10);
    ASSERT_NEAR(large_postive_double, enc_7.decode(), 1e-6);
}
