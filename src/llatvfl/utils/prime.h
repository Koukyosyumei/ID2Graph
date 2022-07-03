#include <cmath>
#include <random>
using namespace std;

long long modpow(long long x, long long n, long long m)
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

bool miller_rabin_primality_test(long long n, mt19937 &mt, long long k = 40)
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

    long long d = n - 1;
    long long s = 0;
    while ((d % 2 == 0))
    {
        d /= 2;
        s += 1;
    }

    long long nm1 = n - 1;
    uniform_int_distribution<long long> distr(1, n - 1);
    long long a, t, y;
    for (long long i = 0; i < k; i++)
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