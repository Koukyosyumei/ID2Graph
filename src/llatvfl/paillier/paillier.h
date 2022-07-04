#pragma once
#include <unordered_map>
#include <random>
#include <cmath>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <cassert>
#include "../utils/prime.h"
using namespace std;

struct PaillierPublicKey;
struct PaillierSecretKey;
struct PaillierCipherText;
struct PaillierKeyRing;
struct PaillierKeyGenerator;

inline long L(long u, long n)
{
    return (u - 1) / n;
}

struct PaillierPublicKey
{
    long n, n2, g;
    uniform_int_distribution<long> distr;
    mt19937 mt;

    PaillierPublicKey(){};
    PaillierPublicKey(long n_, long g_, mt19937 &mt_)
    {
        n = n_;
        n2 = n * n;
        g = g_;
        distr = uniform_int_distribution<long>(0, n - 1);
        mt = mt_;
    }

    bool operator==(PaillierPublicKey pk2)
    {
        return (n == pk2.n) && (g == pk2.g);
    }

    bool operator!=(PaillierPublicKey pk2)
    {
        return (n != pk2.n) || (g != pk2.g);
    }

    PaillierCipherText encrypt(long m);
};

struct PaillierSecretKey
{
    long p, q, n, n2, g, lam, l_g2lam_mod_n2;

    PaillierSecretKey(){};
    PaillierSecretKey(long p_, long q_, long n_, long g_)
    {
        p = p_;
        q = q_;
        n = n_;
        g = g_;

        n2 = n * n;
        lam = lcm(p - 1, q - 1);
        l_g2lam_mod_n2 = L(modpow(g, lam, n * n), n);
    }

    long decrypt(PaillierCipherText pt);
};

struct PaillierCipherText
{
    PaillierPublicKey pk;
    long c;

    PaillierCipherText(PaillierPublicKey pk_, long c_)
    {
        pk = pk_;
        c = c_;
    };

    PaillierCipherText operator+(PaillierCipherText ct)
    {
        if (ct.pk != pk)
        {
            try
            {
                throw runtime_error("public key does not match");
            }
            catch (runtime_error e)
            {
                cerr << e.what() << endl;
            }
        }

        return PaillierCipherText(pk, c * ct.c);
    }

    PaillierCipherText operator+(long pt)
    {
        return PaillierCipherText(pk, (c * modpow(pk.g, pt, pk.n2) % pk.n2));
    }

    PaillierCipherText operator*(long pt)
    {
        return PaillierCipherText(pk, modpow(c, pt, pk.n2));
    }
};

inline PaillierCipherText PaillierPublicKey::encrypt(long m)
{
    if (m < 0 || m >= n)
    {
        try
        {
            throw range_error("m should be [0, n)");
        }
        catch (range_error e)
        {
            cerr << e.what() << endl;
        }
    }

    long r;
    while (true)
    {
        r = distr(mt);
        if (gcd(r, n) == 1)
        {
            break;
        }
    }
    long c = (modpow(g, m, n * n) * modpow(r, n, n * n)) % (n * n);
    cout << r << " " << c << endl;
    return PaillierCipherText(*this, c);
}

inline long PaillierSecretKey::decrypt(PaillierCipherText pt)
{
    if (pt.c <= 0 || pt.c >= (n2))
    {
        try
        {
            throw range_error("pt.c should be (0, n^2)");
        }
        catch (range_error e)
        {
            cerr << e.what() << endl;
        }
    }
    return L(modpow(pt.c, lam, n2), n) / l_g2lam_mod_n2;
}