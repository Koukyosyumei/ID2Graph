#pragma once
#include <unordered_map>
#include <random>
#include <cmath>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <cassert>
#include <boost/integer/mod_inverse.hpp>
#include <boost/math/special_functions/round.hpp>
#include "../tsl/robin_map.h"
#include "../tsl/robin_set.h"
#include "../utils/prime.h"
using namespace std;

struct PaillierPublicKey;
struct PaillierSecretKey;
struct PaillierCipherText;
struct PaillierKeyGenerator;
struct PaillierKeyRing;

inline Bint L(Bint u, Bint n)
{
    return (u - 1) / n;
}

struct PaillierPublicKey
{
    Bint n, n2, g, max_val;
    boost::random::uniform_int_distribution<Bint> distr;

    PaillierPublicKey(){};
    PaillierPublicKey(Bint n_, Bint g_)
    {
        n = n_;
        n2 = n * n;
        g = g_;
        max_val = (n / Bint(3)) - Bint(1);
        distr = boost::random::uniform_int_distribution<Bint>(0, n - 1);
    }

    PaillierPublicKey(Bint n_, Bint g_, Bint n2_)
    {
        n = n_;
        n2 = n2_;
        g = g_;
        max_val = (n / Bint(3)) - Bint(1);
        distr = boost::random::uniform_int_distribution<Bint>(0, n - 1);
    }

    bool operator==(PaillierPublicKey pk2)
    {
        return (n == pk2.n) && (g == pk2.g);
    }

    bool operator!=(PaillierPublicKey pk2)
    {
        return (n != pk2.n) || (g != pk2.g);
    }

    PaillierCipherText encrypt(Bint m);
};

struct PaillierSecretKey
{
    Bint p, q, n, n2, g, lam, mu;

    PaillierSecretKey(){};
    PaillierSecretKey(Bint p_, Bint q_, Bint n_, Bint g_)
    {
        p = p_;
        q = q_;
        n = n_;
        g = g_;

        n2 = n * n;
        lam = lcm(p - 1, q - 1);
        mu = boost::integer::mod_inverse(L(modpow(g, lam, n * n), n), n);
    }

    PaillierSecretKey(Bint p_, Bint q_, Bint n_, Bint g_,
                      Bint n2_, Bint lam_, Bint mu_)
    {
        p = p_;
        q = q_;
        n = n_;
        g = g_;

        n2 = n2_;
        lam = lam_;
        mu = mu_;
    }

    Bint decrypt(PaillierCipherText pt);
};

template <typename T>
struct EncodedNumber
{
    int BASE = 16;
    float LOG2_BASE = log2(BASE);

    PaillierPublicKey pk;
    Bint encoding;
    int exponent;
    double precision;

    EncodedNumber(PaillierPublicKey pk_, T scalar, double precision_ = 1e-8)
    {
        pk = pk_;
        precision = precision_;
        encode(scalar);
    }

    void encode(T scalar)
    {
        if (floor(scalar) == scalar)
        {
            exponent = 0;
        }
        else
        {
            exponent = int(floor(log(precision) / log(BASE)));
        }
        Bint int_rep = (boost::math::round(Bfloat(scalar) * Bfloat(mp::pow(Bint(BASE), -1 * exponent)))).convert_to<Bint>();
        encoding = int_rep % pk.n;
    }

    T decode()
    {
        Bint mantissa;
        if (encoding <= pk.max_val)
        {
            mantissa = encoding;
        }
        else if (encoding >= (pk.n - pk.max_val))
        {
            mantissa = encoding - pk.n;
        }
        else
        {
            try
            {
                throw overflow_error("overflow detected");
            }
            catch (overflow_error e)
            {
                cerr << e.what() << endl;
            }
        }
        return T(Bfloat(mantissa) * mp::pow(Bfloat(BASE), Bfloat(exponent)));
    }
};

struct PaillierCipherText
{
    PaillierPublicKey pk;
    Bint c;

    PaillierCipherText(PaillierPublicKey pk_, Bint c_)
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

    PaillierCipherText operator+(Bint pt)
    {
        return PaillierCipherText(pk, (c * modpow(pk.g, pt, pk.n2) % pk.n2));
    }

    PaillierCipherText operator*(Bint pt)
    {
        return PaillierCipherText(pk, modpow(c, pt, pk.n2));
    }
};

struct PaillierKeyGenerator
{
    int bit_size;

    PaillierKeyGenerator(int bit_size_ = 512)
    {
        bit_size = bit_size_;
    }

    pair<PaillierPublicKey, PaillierSecretKey> generate_keypair()
    {
        boost::random::random_device rng;

        Bint p = generate_probably_prime(bit_size);
        Bint q = generate_probably_prime(bit_size);

        if (p == q)
        {
            return generate_keypair();
        }

        Bint n = p * q;
        Bint n2 = n * n;
        boost::random::uniform_int_distribution<Bint> distr = boost::random::uniform_int_distribution<Bint>(0, n2 - 1);

        Bint g, lam, l_g2lam_mod_n2, mu;
        do
        {
            g = distr(rng);
            lam = lcm(p - 1, q - 1);
            l_g2lam_mod_n2 = L(modpow(g, lam, n * n), n);

        } while ((gcd(g, n2) != 1) && (gcd(l_g2lam_mod_n2, n) != 1));

        mu = boost::integer::mod_inverse(l_g2lam_mod_n2, n);

        PaillierPublicKey pk = PaillierPublicKey(n, g, n2);
        PaillierSecretKey sk = PaillierSecretKey(p, q, n, g, n2, lam, mu);

        return make_pair(pk, sk);
    }
};

struct HashPairSzudzikBint
{
    // implementation of szudzik paring
    template <class T1, class T2>
    size_t operator()(const pair<T1, T2> &p) const
    {
        size_t seed;
        if (p.first >= p.second)
        {
            seed = std::hash<T1>{}(p.first * p.first + p.first + p.second);
        }
        else
        {
            seed = std::hash<T1>{}(p.second * p.second + p.first);
        }
        return seed;
    }
};

struct PaillierKeyRing
{
    tsl::robin_map<pair<Bint, Bint>, PaillierSecretKey, HashPairSzudzikBint> keyring;

    PaillierKeyRing(){};

    void add(PaillierSecretKey sk)
    {
        keyring.emplace(make_pair(sk.n, sk.g), sk);
    }

    PaillierSecretKey get_sk(PaillierPublicKey pk)
    {
        return keyring[make_pair(pk.n, pk.g)];
    }

    Bint decrypt(PaillierCipherText pt);
};

inline PaillierCipherText
PaillierPublicKey::encrypt(Bint m)
{
    boost::random::random_device rng;

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

    Bint g2m_mod_n2;

    if (n - max_val <= m)
    {
        Bint neg_m = n - m;
        Bint neg_c = modpow(g, neg_m, n2);
        g2m_mod_n2 = boost::integer::mod_inverse(neg_c, n2);
    }
    else
    {
        g2m_mod_n2 = modpow(g, m, n2);
    }

    Bint r;
    while (true)
    {
        r = distr(rng);
        if (gcd(r, n) == 1)
        {
            break;
        }
    }
    Bint c = (g2m_mod_n2 * modpow(r, n, n2)) % n2;
    return PaillierCipherText(*this, c);
}

inline Bint PaillierSecretKey::decrypt(PaillierCipherText pt)
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
    return (L(modpow(pt.c, lam, n2), n) * mu) % n;
}

inline Bint PaillierKeyRing::decrypt(PaillierCipherText pt)
{
    return get_sk(pt.pk).decrypt(pt);
}