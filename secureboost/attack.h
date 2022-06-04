#include <iostream>
#include <limits>
#include <vector>
#include "secureboost.h"
using namespace std;

void travase_nodes_to_extract_adjacency_matrix(Node *node,
                                               vector<vector<int>> *adj_mat)
{
    if (node->is_leaf())
    {
        for (int i = 0; i < node->idxs.size(); i++)
        {
            for (int j = i + 1; j < node->idxs.size(); j++)
            {
                adj_mat->at(node->idxs[i])[node->idxs[j]] += 1;
                adj_mat->at(node->idxs[j])[node->idxs[i]] += 1;
            }
        }
    }
    else
    {
        travase_nodes_to_extract_adjacency_matrix(node->left, adj_mat);
        travase_nodes_to_extract_adjacency_matrix(node->right, adj_mat);
    }
}

vector<vector<int>> extract_adjacency_matrix_from_tree(XGBoostTree *tree)
{
    int num_row = tree->dtree.idxs.size();
    vector<vector<int>> adj_mat(num_row, vector<int>(num_row, 0));
    travase_nodes_to_extract_adjacency_matrix(&tree->dtree, &adj_mat);
    return adj_mat;
}

vector<vector<vector<int>>> extract_adjacency_matrix_from_forest(SecureBoostBase *model)
{
    vector<vector<vector<int>>> result(model->estimators.size());
    for (int i = 0; i < model->estimators.size(); i++)
    {
        result[i] = extract_adjacency_matrix_from_tree(&model->estimators[i]);
    }
}