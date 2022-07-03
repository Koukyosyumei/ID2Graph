#pragma once
#include <unordered_map>
#include <random>
#include <cmath>
using namespace std;

struct PublicKey;
struct SecretKey;
struct PaillierCipherText;
struct PaillierKeyRing;
struct PaillierKeyGenerator;

struct PaillierCipherText
{
    PublicKey pk;
    long c;

    PaillierCipherText(PublicKey *pk_, long c_)
    {
        pk = *pk_;
        c = c_;
    };

    PaillierCipherText operator+(PaillierCipherText ct)
    {
    }

    PaillierCipherText operator+(long v)
    {
    }

    PaillierCipherText operator*(long v)
    {
    }
};

struct PublicKey
{
    long n, g;
    uniform_int_distribution<long> distr;
    mt19937 mt;

    PublicKey(){};
    PublicKey(long n_, long g_, mt19937 &mt_)
    {
        n = n_;
        g = g_;
        distr = uniform_int_distribution<long>(0, n * n - 1);
        mt = mt_;
    }

    PaillierCipherText encrypt(long m)
    {
        long r = distr(mt);
        long c = (long(pow(g, m)) * long(pow(r, n))) % (n * n);
        return PaillierCipherText(this, c);
    }
};

struct SecretKey
{
    long p, q;

    SecretKey(){};
    SecretKey(long p_, long q_)
    {
        p = p_;
        q = q_;
    }

    long decrypt(PaillierCipherText pt)
    {
    }
};