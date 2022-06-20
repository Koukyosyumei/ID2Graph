#pragma once
#include <random>
#include <unordered_map>
using namespace std;

struct HashPair
{

    static size_t m_hash_pair_random;

    template <class T1, class T2>
    size_t operator()(const pair<T1, T2> &p) const
    {

        auto hash1 = hash<T1>{}(p.first);
        auto hash2 = hash<T2>{}(p.second);

        size_t seed = 0;
        seed ^= hash1 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hash2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= m_hash_pair_random + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

size_t HashPair::m_hash_pair_random = (size_t)random_device()();

template <typename DataType>
struct SparseMatrixDOK
{
    size_t dim_row = 0;
    size_t dim_column = 0;
    DataType zero_val = 0;
    unordered_map<pair<unsigned int, unsigned int>, DataType, HashPair> um_ij2w;

    SparseMatrixDOK(){};
    SparseMatrixDOK(size_t dim_row_, size_t dim_column_, DataType zero_val_ = 0)
    {
        dim_row = dim_row_;
        dim_column = dim_column_;
        zero_val = zero_val_;
    }

    void add(unsigned int i, unsigned int j, DataType w)
    {
        pair<unsigned int, unsigned int> temp_pair = make_pair(i, j);
        if (um_ij2w.find(temp_pair) != um_ij2w.end())
        {
            um_ij2w[temp_pair] = um_ij2w[temp_pair] + w;
            if (um_ij2w[temp_pair] == zero_val)
            {
                um_ij2w.erase(temp_pair);
            }
        }
        else
        {
            um_ij2w.emplace(temp_pair, w);
        }
    }

    vector<vector<DataType>> to_adjacency_matrix(DataType init_val = 0)
    {
        vector<vector<DataType>> adj_mat(dim_row, vector<DataType>(dim_column, init_val));
        auto it = um_ij2w.begin();

        while (it != um_ij2w.end())
        {
            adj_mat[it->first.first][it->first.second] = it->second;
            it++;
        }

        return adj_mat;
    }
};