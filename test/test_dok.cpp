#include <vector>
#include <iostream>
#include <cassert>
#include "../src/llatvfl/utils/dok.h"

int main()
{
    SparseMatrixDOK<float> sm = SparseMatrixDOK<float>(3, 3, 0, false, true);
    sm.add(1, 1, 0.5);
    sm.add(2, 1, 0.1);
    sm.add(1, 1, -0.5);
    sm.add(2, 1, 0.3);
    sm.add(0, 2, 1);

    vector<vector<float>> test_adj_mat = {{0, 0, 1.0},
                                          {0, 0, 0},
                                          {0, 0.4, 0}};
    vector<vector<float>> adj_mat = sm.to_densematrix(0);

    for (int i = 0; i < test_adj_mat.size(); i++)
    {
        assert(adj_mat.size() == test_adj_mat.size());
        for (int j = 0; j < test_adj_mat[i].size(); j++)
        {
            assert(sm(i, j) == test_adj_mat[i][j]);
            assert(adj_mat[i][j] == test_adj_mat[i][j]);
        }
    }

    vector<vector<int>> test_row2nonzero_idx = {{2}, {}, {1}};
    for (int i = 0; i < test_row2nonzero_idx.size(); i++)
    {
        for (int j = 0; j < test_row2nonzero_idx[i].size(); j++)
        {
            assert(test_row2nonzero_idx[i][j] == sm.row2nonzero_idx[i][j]);
        }
    }

    SparseMatrixDOK<float> sm_symme = SparseMatrixDOK<float>(3, 3, 0, true, true);
    sm_symme.add(1, 1, 0.5);
    sm_symme.add(2, 1, 0.1);
    sm_symme.add(1, 1, -0.5);
    sm_symme.add(2, 1, 0.3);
    sm_symme.add(0, 2, 1);

    vector<vector<float>> test_adj_mat_symme = {{0, 0, 1.0},
                                                {0, 0, 0.4},
                                                {1.0, 0.4, 0}};
    vector<vector<float>> adj_mat_symme = sm_symme.to_densematrix(0);

    for (int i = 0; i < test_adj_mat_symme.size(); i++)
    {
        assert(adj_mat_symme.size() == test_adj_mat_symme.size());
        for (int j = 0; j < test_adj_mat_symme[i].size(); j++)
        {
            assert(sm_symme(i, j) == test_adj_mat_symme[i][j]);
            assert(adj_mat_symme[i][j] == test_adj_mat_symme[i][j]);
        }
    }

    vector<vector<int>> test_row2nonzero_idx_symme = {{}, {}, {1, 0}};
    for (int i = 0; i < test_row2nonzero_idx_symme.size(); i++)
    {
        for (int j = 0; j < test_row2nonzero_idx_symme[i].size(); j++)
        {
            assert(test_row2nonzero_idx_symme[i][j] == sm_symme.row2nonzero_idx[i][j]);
        }
    }

    cout << "test_dok: all passed!" << endl;
}