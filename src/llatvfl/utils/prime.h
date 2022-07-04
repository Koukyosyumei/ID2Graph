#pragma once
#include <cmath>
#include <random>
using namespace std;

inline long gcd(long a, long b)
{
    if (a % b == 0)
    {
        return b;
    }
    else
    {
        return gcd(b, a % b);
    }
}

inline long lcm(long a, long b)
{
    return abs(a) / gcd(a, b) * abs(b);
}

inline long modpow(long x, long n, long m)
{
    // returns x^n (mod m)
    long ret = 1;
    while (n > 0)
    {
        if (n & 1)
            ret = ret * x % m;
        x = x * x % m;
        n >>= 1;
    }
    return ret;
}

inline bool cond_of_miller_rabin(long d, long a, long n)
{
    long t = d;
    long y = modpow(a, t, n);

    while ((t != n - 1) && (y != 1) && (y != n - 1))
    {
        y = (y * y) % n;
        t <<= 1;
    }

    return (y != n - 1) && (t % 2) == 0;
}

inline bool miller_rabin_primality_test(long n, mt19937 &mt, long k = 40)
{
    if (n <= 0)
    {
        return false;
    }

    if (n == 2)
    {
        return true;
    }

    if (n == 1 || n % 2 == 0)
    {
        return false;
    }

    long d = n - 1;
    long s = 0;
    while ((d % 2 == 0))
    {
        d /= 2;
        s += 1;
    }

    long nm1 = n - 1;
    uniform_int_distribution<long> distr(1, n - 1);

    if (n < 2047)
    {
        for (long a : {2})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < 1373653)
    {
        for (long a : {2, 3})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < 9080191)
    {
        for (long a : {31, 73})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < 25326001)
    {
        for (long a : {2, 3, 5})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < 3215031751)
    {
        for (long a : {2, 3, 5, 7})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < 4759123141)
    {
        for (long a : {2, 7, 61})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < 2152302898747)
    {
        for (long a : {2, 3, 5, 7, 11})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < 3474749660383)
    {
        for (long a : {2, 3, 5, 7, 11, 13})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else if (n < 341550071728321)
    {
        for (long a : {2, 3, 5, 7, 11, 13, 17})
        {
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    else
    {
        long a;
        for (long i = 0; i < k; i++)
        {
            a = distr(mt);
            if (cond_of_miller_rabin(d, a, n))
            {
                return false;
            }
        }
    }
    return true;
}

inline long generate_probably_prime(int bits_size, mt19937 mt)
{
    long min_val = pow(2, bits_size - 1);
    long max_val = min_val * 2 - 1;
    uniform_int_distribution<long> distr(min_val, max_val);
    long p = 0;
    while (!miller_rabin_primality_test(p, mt))
    {
        p = distr(mt);
    }
    return p;
}