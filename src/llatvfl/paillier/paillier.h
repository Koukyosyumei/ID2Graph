#pragma once
#include <unordered_map>
using namespace std;

struct PublicKey;
struct SecretKey;
struct PaillierCipherText;
struct PaillierKeyRing;
struct PaillierKeyGenerator;

struct PublicKey
{
    PaillierCipherText encrypt(int v)
    {
    }
};

struct SecretKey
{
    int decrypt(PaillierCipherText pt)
    {
    }
};

struct PaillierCipherText
{
    PublicKey pk;
    int val;

    PaillierCipherText(PublicKey pk_, int val_)
    {
        pk = pk_;
        val = val_;
    };

    PaillierCipherText operator+(PaillierCipherText ct)
    {
    }

    PaillierCipherText operator+(int v)
    {
    }

    PaillierCipherText operator*(int v)
    {
    }
};