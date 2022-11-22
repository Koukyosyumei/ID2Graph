#pragma once
#include <algorithm>
#include <random>
#include <utility>
#include <unordered_map>
#include <fstream>
#include "../tsl/robin_map.h"
#include "../tsl/robin_set.h"
using namespace std;

/**
 * @brief Implementation of Szudzik paring.
 *
 */
struct HashPairSzudzik
{
    template <class T1, class T2>
    size_t operator()(const pair<T1, T2> &p) const
    {
        size_t seed;
        if (p.first >= p.second)
        {
            seed = p.first * p.first + p.first + p.second;
        }
        else
        {
            seed = p.second * p.second + p.first;
        }
        return seed;
    }
};

/**
 * @brief Sparse matrix of DOK format.
 *
 * @tparam DataType
 */
template <typename DataType>
struct SparseMatrixDOK
{
    size_t dim_row = 0;
    size_t dim_column = 0;
    DataType zero_val = 0;
    bool is_symmetric = false;
    bool save_row2nonzero_idx = true;

    vector<vector<int>> row2nonzero_idx;

    tsl::robin_map<pair<unsigned int, unsigned int>, DataType, HashPairSzudzik> um_ij2w;

    SparseMatrixDOK(){};
    SparseMatrixDOK(size_t dim_row_, size_t dim_column_, DataType zero_val_ = 0,
                    bool is_symmetric_ = false, bool save_row2nonzero_idx_ = false)
    {
        dim_row = dim_row_;
        dim_column = dim_column_;
        zero_val = zero_val_;
        is_symmetric = is_symmetric_;
        save_row2nonzero_idx = save_row2nonzero_idx_;

        if (save_row2nonzero_idx)
        {
            row2nonzero_idx.resize(dim_row);
        }
    }

    DataType &operator()(unsigned int i, unsigned int j)
    {
        if (is_symmetric && (i < j))
        {
            return um_ij2w[make_pair(j, i)];
        }
        else
        {
            return um_ij2w[make_pair(i, j)];
        }
    }

    /**
     * @brief Adds value at position (i, j).
     *
     * @param i row index
     * @param j column index
     * @param w added value
     */
    void add(unsigned int i, unsigned int j, DataType w)
    {
        if (is_symmetric && (i < j))
        {
            // save only the lower triangle matrix
            swap(i, j);
        }

        pair<unsigned int, unsigned int> temp_pair = make_pair(i, j);
        if (um_ij2w.find(temp_pair) != um_ij2w.end())
        {
            um_ij2w[temp_pair] = um_ij2w[temp_pair] + w;
            if (um_ij2w[temp_pair] == zero_val)
            {
                um_ij2w.erase(temp_pair);

                if (save_row2nonzero_idx)
                {
                    row2nonzero_idx[i].erase(remove_if(row2nonzero_idx[i].begin(),
                                                       row2nonzero_idx[i].end(),
                                                       [j](int v)
                                                       { return v == j; }),
                                             row2nonzero_idx[i].cend());
                }
            }
        }
        else
        {
            um_ij2w.emplace(temp_pair, w);
            if (save_row2nonzero_idx)
            {
                row2nonzero_idx[i].push_back(j);
            }
        }
    }

    /**
     * @brief Converts a dense matrix to a sparse matrix.
     *
     * @param densematrix
     */
    void from_densematrix(vector<vector<DataType>> &densematrix)
    {
        for (int i = 0; i < dim_row; i++)
        {
            int end_pos = dim_column;
            if (is_symmetric)
            {
                end_pos = i;
            }
            for (int j = 0; j < end_pos; j++)
            {
                if (densematrix[i][j] != zero_val)
                {
                    add(i, j, densematrix[i][j]);
                }
            }
        }
    }

    /**
     * @brief Converts this sparse matrix to a dense matrix.
     *
     * @param init_val
     * @return vector<vector<DataType>>
     */
    vector<vector<DataType>> to_densematrix(DataType init_val = 0)
    {
        vector<vector<DataType>> adj_mat(dim_row, vector<DataType>(dim_column, init_val));
        auto it = um_ij2w.begin();

        while (it != um_ij2w.end())
        {
            adj_mat[it->first.first][it->first.second] = it->second;
            if (is_symmetric)
            {
                adj_mat[it->first.second][it->first.first] = it->second;
            }
            it++;
        }

        return adj_mat;
    }

    /**
     * @brief Dumps this sparse matrix to a text file.
     *
     * @param filepath
     */
    void save(string filepath)
    {
        ofstream adj_mat_file;
        adj_mat_file.open(filepath, ios::out);
        adj_mat_file << dim_row << "\n";
        int temp_num_link = 0;
        for (int i = 0; i < dim_row; i++)
        {
            temp_num_link = row2nonzero_idx[i].size();
            adj_mat_file << temp_num_link << " ";
            for (int j = 0; j < temp_num_link; j++)
            {
                adj_mat_file << row2nonzero_idx[i][j] << " " << (*this)(i, row2nonzero_idx[i][j]) << " ";
            }
            adj_mat_file << "\n";
        }
        adj_mat_file.close();
    }
};