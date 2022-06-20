#include <vector>
#include <iostream>
#include <cassert>
#include "../src/llatvfl/utils/dok.h"

int main()
{
    SparseMatrixDOK<float> sm = SparseMatrixDOK<float>(3, 3, 0);
    sm.add(1, 1, 0.5);
    sm.add(2, 1, 0.1);
    sm.add(1, 1, -0.5);
    sm.add(2, 1, 0.3);
    sm.add(0, 2, 1);

    vector<vector<float>> test_adj_mat = {{0, 0, 1.0},
                                          {0, 0, 0},
                                          {0, 0.4, 0}};
    vector<vector<float>> adj_mat = sm.to_densematrix(0);

    for (int i = 0; i < adj_mat.size(); i++)
    {
        assert(adj_mat.size() == test_adj_mat.size());
        for (int j = 0; j < adj_mat[i].size(); j++)
        {
            assert(adj_mat[i][j] == test_adj_mat[i][j]);
        }
    }

    SparseMatrixDOK<float> sm_symme = SparseMatrixDOK<float>(3, 3, 0, true);
    sm_symme.add(1, 1, 0.5);
    sm_symme.add(2, 1, 0.1);
    sm_symme.add(1, 1, -0.5);
    sm_symme.add(2, 1, 0.3);
    sm_symme.add(0, 2, 1);

    vector<vector<float>> test_adj_mat_symme = {{0, 0, 1.0},
                                                {0, 0, 0.4},
                                                {1.0, 0.4, 0}};
    vector<vector<float>> adj_mat_symme = sm_symme.to_densematrix(0);

    for (int i = 0; i < adj_mat_symme.size(); i++)
    {
        assert(adj_mat_symme.size() == test_adj_mat_symme.size());
        for (int j = 0; j < adj_mat_symme[i].size(); j++)
        {
            assert(adj_mat_symme[i][j] == test_adj_mat_symme[i][j]);
        }
    }

    cout << "test_dok: all passed!" << endl;
}