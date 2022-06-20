#pragma once
#include <iostream>
#include <limits>
#include <vector>
#include "../utils/dok.h"
#include "../randomforest/randomforest.h"
#include "../xgboost/xgboost.h"
using namespace std;

template <typename NodeType>
bool travase_nodes_to_extract_weighted_adjacency_matrix(NodeType *node,
                                                        int max_depth,
                                                        SparseMatrixDOK<int> &adj_mat,
                                                        int target_party_id = -1)
{
    bool skip_flag = false;
    if (node->is_leaf())
    {
        skip_flag = node->depth <= 0 && target_party_id != -1 && node->party_id != target_party_id;
    }
    else
    {
        travase_nodes_to_extract_weighted_adjacency_matrix(node->left, max_depth, adj_mat, target_party_id);
        travase_nodes_to_extract_weighted_adjacency_matrix(node->right, max_depth, adj_mat, target_party_id);
    }
    if (!skip_flag)
    {
        for (int i = 0; i < node->idxs.size(); i++)
        {
            for (int j = i + 1; j < node->idxs.size(); j++)
            {
                adj_mat.add(node->idxs[i], node->idxs[j], max_depth - node->depth);
                adj_mat.add(node->idxs[j], node->idxs[i], max_depth - node->depth);
            }
        }
    }
    return skip_flag;
}

template <typename NodeType>
bool travase_nodes_to_extract_adjacency_matrix(NodeType *node,
                                               SparseMatrixDOK<int> &adj_mat,
                                               int target_party_id = -1)
{
    bool skip_flag;
    if (node->is_leaf())
    {
        skip_flag = node->depth <= 0 && target_party_id != -1 && node->party_id != target_party_id;
        if (!skip_flag)
        {
            for (int i = 0; i < node->idxs.size(); i++)
            {
                for (int j = i + 1; j < node->idxs.size(); j++)
                {
                    adj_mat.add(node->idxs[i], node->idxs[j], 1);
                    adj_mat.add(node->idxs[j], node->idxs[i], 1);
                }
            }
        }
    }
    else
    {
        skip_flag = false;
        bool left_skip_flag = travase_nodes_to_extract_adjacency_matrix<NodeType>(node->left, adj_mat, target_party_id);
        bool right_skip_flag = travase_nodes_to_extract_adjacency_matrix<NodeType>(node->right, adj_mat, target_party_id);

        if (left_skip_flag && right_skip_flag)
        {
            for (int i = 0; i < node->idxs.size(); i++)
            {
                for (int j = i + 1; j < node->idxs.size(); j++)
                {
                    adj_mat.add(node->idxs[i], node->idxs[j], 1);
                    adj_mat.add(node->idxs[j], node->idxs[i], 1);
                }
            }
        }
    }
    return skip_flag;
}

vector<vector<int>> extract_adjacency_matrix_from_tree(XGBoostTree *tree, int target_party_id = 1,
                                                       bool is_weighted = true)
{
    int num_row = tree->dtree.y.size();
    SparseMatrixDOK<int> adj_mat(num_row, num_row, 0);
    bool skip_flag;
    if (is_weighted)
    {
        skip_flag = travase_nodes_to_extract_weighted_adjacency_matrix<XGBoostNode>(&tree->dtree, tree->dtree.depth, adj_mat, target_party_id);
    }
    else
    {
        skip_flag = travase_nodes_to_extract_adjacency_matrix<XGBoostNode>(&tree->dtree, adj_mat, target_party_id);
    }
    if (skip_flag)
    {
        adj_mat = SparseMatrixDOK<int>(num_row, num_row, 0);
    }
    return adj_mat.to_densematrix();
}

vector<vector<int>> extract_adjacency_matrix_from_tree(RandomForestTree *tree, int target_party_id = 1,
                                                       bool is_weighted = true)
{
    int num_row = tree->dtree.y.size();
    SparseMatrixDOK<int> adj_mat(num_row, num_row, 0);
    bool skip_flag;
    if (is_weighted)
    {
        skip_flag = travase_nodes_to_extract_weighted_adjacency_matrix<RandomForestNode>(&tree->dtree, tree->dtree.depth, adj_mat, target_party_id);
    }
    else
    {
        skip_flag = travase_nodes_to_extract_adjacency_matrix<RandomForestNode>(&tree->dtree, adj_mat, target_party_id);
    }
    if (skip_flag)
    {
        adj_mat = SparseMatrixDOK<int>(num_row, num_row, 0);
    }
    return adj_mat.to_densematrix();
}

vector<vector<vector<int>>> extract_adjacency_matrix_from_forest(XGBoostBase *model,
                                                                 int target_party_id = -1,
                                                                 bool is_weighted = true)
{
    vector<vector<vector<int>>> result(model->estimators.size());
    for (int i = 0; i < model->estimators.size(); i++)
    {
        result[i] = extract_adjacency_matrix_from_tree(&model->estimators[i], target_party_id, is_weighted);
    }
    return result;
}

vector<vector<vector<int>>> extract_adjacency_matrix_from_forest(RandomForestClassifier *model,
                                                                 int target_party_id = -1,
                                                                 bool is_weighted = true)
{
    vector<vector<vector<int>>> result(model->estimators.size());
    for (int i = 0; i < model->estimators.size(); i++)
    {
        result[i] = extract_adjacency_matrix_from_tree(&model->estimators[i], target_party_id, is_weighted);
    }
    return result;
}