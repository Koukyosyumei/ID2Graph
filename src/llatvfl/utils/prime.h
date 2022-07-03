#include <cmath>
#include <random>
using namespace std;

long modpow(long x, long n, long m)
{
    // returns x^n (mod m)
    long long ret = 1;
    while (n > 0)
    {
        if (n & 1)
            ret = ret * x % m;
        x = x * x % m;
        n >>= 1;
    }
    return ret;
}

bool miller_rabin_primality_test(long n, mt19937 &mt, long k = 1024)
{
    if (n <= 0)
    {
        return false;
    }

    if (n == 2)
    {
        return true;
    }

    if (n % 2 == 0)
    {
        return false;
    }

    long d = n - 1;
    long s = 0;
    while ((d % 2 == 0))
    {
        d >>= 2;
        s += 1;
    }

    long nm1 = n - 1;
    uniform_int_distribution<long> distr(1, n - 1);
    long a, t, y;
    for (long i = 0; i < k; i++)
    {
        a = distr(mt);
        t = d;
        y = modpow(a, t, n);

        while ((t != n - 1) && (y != 1) && (y != n - 1))
        {
            y = (y * y) % n;
            t <<= 1;
        }

        if ((y != n - 1) && (t % 2) == 0)
        {
            return false;
        }
    }
    return true;
}