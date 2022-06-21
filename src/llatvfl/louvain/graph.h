#pragma once
#include <iostream>
#include <limits>
#include <vector>
#include "../utils/dok.h"
using namespace std;

struct Graph
{
    unsigned long num_nodes = 0;
    unsigned long num_links = 0;
    double total_weight = 0;

    // cumulative degree for each node, deg(0) = degrees[0]
    // deg(k) = degrees[k] - degrees[k-1]
    vector<unsigned long> degrees;
    vector<vector<int>> nodes; // current node idx to original record idxs
    vector<unsigned int> links;
    vector<float> weights; // TODO check if `double` works or not

    Graph(){};
    Graph(vector<vector<int>> &node2original_records)
    {
        num_nodes = 0;
        num_links = 0;
        total_weight = 0;

        nodes.reserve(node2original_records.size());
        for (size_t i = 0; i < node2original_records.size(); i++)
        {
            nodes.push_back(node2original_records[i]);
        }
    }
    Graph(SparseMatrixDOK<float> sm_dok)
    {
        num_nodes = sm_dok.dim_row;
        // initilize first graph without contraction
        for (unsigned int i = 0; i < num_nodes; i++)
        {
            vector<int> n;
            n.push_back(i);
            nodes.push_back(n);
        }

        int temp_num_links = 0;
        int cum_degrees = 0;
        degrees.resize(num_nodes, 0);
        for (int i = 0; i < num_nodes; i++)
        {
            temp_num_links = sm_dok.row2nonzero_idx[i].size();
            cum_degrees += temp_num_links;
            degrees[i] = cum_degrees;

            for (int j = 0; j < temp_num_links; j++)
            {
                links.push_back(sm_dok.row2nonzero_idx[i][j]);
                weights.push_back(sm_dok(i, sm_dok.row2nonzero_idx[i][j]));
            }
        }

        // compute total weight
        for (unsigned int i = 0; i < num_nodes; i++)
        {
            total_weight += (double)get_weighted_degree(i);
        }
    }
    Graph(unsigned long num_nodes_, vector<unsigned long> &degrees_,
          vector<unsigned int> &links_, vector<float> &weights_)
    {
        num_nodes = num_nodes_;
        degrees = degrees_;
        links = links_;
        weights = weights_;

        // initilize first graph without contraction
        for (unsigned int i = 0; i < num_nodes; i++)
        {
            vector<int> n;
            n.push_back(i);
            nodes.push_back(n);
        }

        // compute total weight
        for (unsigned int i = 0; i < num_nodes; i++)
        {
            total_weight += (double)get_weighted_degree(i);
        }
    }

    void add_nose(vector<int> &n)
    {
        nodes.push_back(n);
    }

    // return pointers to the first neighbor and weight of the edge between the node and the neighbor
    pair<vector<unsigned int>::iterator, vector<float>::iterator> get_neighbors(unsigned int node)
    {
        if (node == 0)
        {
            return make_pair(links.begin(), weights.begin());
        }
        else if (weights.size() != 0)
        {
            return make_pair(links.begin() + degrees[node - 1], weights.begin() + degrees[node - 1]);
        }
        else
        {
            return make_pair(links.begin() + degrees[node - 1], weights.begin());
        }
    }

    // return the number of neighbors of the node
    unsigned int get_num_neighbors(unsigned int node)
    {
        if (node == 0)
        {
            return degrees[0];
        }
        else
        {
            return degrees[node] - degrees[node - 1];
        }
    }

    // return the number or the weight of self loops of the node
    double get_num_selfloops(unsigned int node)
    {
        bool is_weights_size_is_not_zero = weights.size() != 0;
        pair<vector<unsigned int>::iterator, vector<float>::iterator> p = get_neighbors(node);
        for (unsigned int i = 0; i < get_num_neighbors(node); i++)
        {
            if (*(p.first + i) == node)
            {
                if (is_weights_size_is_not_zero)
                {
                    return (double)*(p.second + i);
                }
                else
                {
                    return 1;
                }
            }
        }
        return 0.;
    }

    // retrn the weighed degree of the node
    double get_weighted_degree(unsigned int node)
    {
        if (weights.size() == 0)
        {
            return (double)get_num_neighbors(node);
        }
        else
        {
            unsigned int num_neighbors = get_num_neighbors(node);
            pair<vector<unsigned int>::iterator, vector<float>::iterator> p = get_neighbors(node);
            double res = 0;
            for (unsigned int i = 0; i < num_neighbors; i++)
            {
                res += (double)*(p.second + i);
            }
            return res;
        }
    }
};