#include <iostream>
#include <limits>
#include <vector>
#include <cassert>
#include "../src/llatvfl/louvain/louvain.h"
using namespace std;

void test_louvain_from_dense_matrix()
{
    vector<vector<int>> test_adj_mat = {{0, 0, 1, 0, 0, 0, 0, 1},
                                        {0, 0, 0, 0, 1, 1, 0, 0},
                                        {1, 0, 0, 0, 0, 0, 0, 1},
                                        {0, 0, 0, 0, 0, 0, 1, 0},
                                        {0, 1, 0, 0, 0, 1, 0, 0},
                                        {0, 1, 0, 0, 1, 0, 0, 0},
                                        {0, 0, 0, 1, 0, 0, 0, 0},
                                        {1, 0, 1, 0, 0, 0, 0, 0}};

    int num_nodes = test_adj_mat.size();
    vector<unsigned long> degrees;
    vector<unsigned int> links;
    vector<float> weights;

    int cum_degree = 0;
    for (int i = 0; i < num_nodes; i++)
    {
        for (int j = 0; j < num_nodes; j++)
        {
            if (test_adj_mat[i][j] != 0)
            {
                cum_degree += 1;
                links.push_back(j);
                weights.push_back(test_adj_mat[i][j]);
            }
        }
        degrees.push_back(cum_degree);
    }

    Graph g = Graph(num_nodes, degrees, links, weights);

    assert(g.get_num_neighbors(0) == 2);
    assert(g.get_num_neighbors(3) == 1);
    assert(g.get_num_selfloops(0) == 0);
    assert(g.get_weighted_degree(0) == 2);
    assert(g.get_weighted_degree(2) == 2);
    assert(g.total_weight == 14);

    Community c = Community(g, -1, 0.000001);
    assert(abs(c.modularity() - -0.1326530612244898) < 1e-6);
    c.compute_neigh_comms(0);
    assert(c.neigh_last == 3);

    vector<float> test_neigh_weight = {0, -1, 1, -1, -1, -1, -1, 1};
    for (int i = 0; i < c.num_nodes; i++)
    {
        assert(c.neigh_weight[i] == test_neigh_weight[i]);
    }
}

void test_louvain_from_sparse_matrix()
{
    vector<vector<float>> test_adj_mat = {{0, 0, 1, 0, 0, 0, 0, 1},
                                          {0, 0, 0, 0, 1, 1, 0, 0},
                                          {1, 0, 0, 0, 0, 0, 0, 1},
                                          {0, 0, 0, 0, 0, 0, 1, 0},
                                          {0, 1, 0, 0, 0, 1, 0, 0},
                                          {0, 1, 0, 0, 1, 0, 0, 0},
                                          {0, 0, 0, 1, 0, 0, 0, 0},
                                          {1, 0, 1, 0, 0, 0, 0, 0}};

    int num_nodes = test_adj_mat.size();
    SparseMatrixDOK<float> sm = SparseMatrixDOK<float>(num_nodes, num_nodes, 0, false, true);
    sm.from_densematrix(test_adj_mat);

    Graph g = Graph(sm);

    assert(g.get_num_neighbors(0) == 2);
    assert(g.get_num_neighbors(3) == 1);
    assert(g.get_num_selfloops(0) == 0);
    assert(g.get_weighted_degree(0) == 2);
    assert(g.get_weighted_degree(2) == 2);
    assert(g.total_weight == 14);

    Community c = Community(g, -1, 0.000001);
    assert(abs(c.modularity() - -0.1326530612244898) < 1e-6);
    c.compute_neigh_comms(0);
    assert(c.neigh_last == 3);

    vector<float> test_neigh_weight = {0, -1, 1, -1, -1, -1, -1, 1};
    for (int i = 0; i < c.num_nodes; i++)
    {
        assert(c.neigh_weight[i] == test_neigh_weight[i]);
    }
}

int main()
{
    test_louvain_from_dense_matrix();
    test_louvain_from_sparse_matrix();
    cout << "test_louvain: all passed!" << endl;
}