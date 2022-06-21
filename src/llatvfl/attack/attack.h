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
                                                        SparseMatrixDOK<float> &adj_mat,
                                                        float weight,
                                                        int target_party_id)
{
    bool skip_flag = false;
    if (node->is_leaf())
    {
        skip_flag = node->depth <= 0 && target_party_id != -1 && node->party_id != target_party_id;
    }
    else
    {
        travase_nodes_to_extract_weighted_adjacency_matrix(node->left, max_depth, adj_mat, weight, target_party_id);
        travase_nodes_to_extract_weighted_adjacency_matrix(node->right, max_depth, adj_mat, weight, target_party_id);
    }
    if (!skip_flag)
    {
        for (int i = 0; i < node->idxs.size(); i++)
        {
            for (int j = i + 1; j < node->idxs.size(); j++)
            {
                adj_mat.add(node->idxs[i], node->idxs[j], weight * float(max_depth - node->depth));
            }
        }
    }
    return skip_flag;
}

template <typename NodeType>
bool travase_nodes_to_extract_adjacency_matrix(NodeType *node,
                                               SparseMatrixDOK<float> &adj_mat,
                                               float weight,
                                               int target_party_id)
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
                    adj_mat.add(node->idxs[i], node->idxs[j], weight);
                }
            }
        }
    }
    else
    {
        skip_flag = false;
        bool left_skip_flag = travase_nodes_to_extract_adjacency_matrix<NodeType>(node->left, adj_mat, weight, target_party_id);
        bool right_skip_flag = travase_nodes_to_extract_adjacency_matrix<NodeType>(node->right, adj_mat, weight, target_party_id);

        if (left_skip_flag && right_skip_flag)
        {
            for (int i = 0; i < node->idxs.size(); i++)
            {
                for (int j = i + 1; j < node->idxs.size(); j++)
                {
                    adj_mat.add(node->idxs[i], node->idxs[j], weight);
                }
            }
        }
    }
    return skip_flag;
}

void extract_adjacency_matrix_from_tree(XGBoostTree *tree,
                                        SparseMatrixDOK<float> &adj_mat,
                                        float weight,
                                        int target_party_id,
                                        bool is_weighted)
{
    int num_row = tree->dtree.y.size();
    if (is_weighted)
    {
        travase_nodes_to_extract_weighted_adjacency_matrix<XGBoostNode>(&tree->dtree, tree->dtree.depth, adj_mat, weight, target_party_id);
    }
    else
    {
        travase_nodes_to_extract_adjacency_matrix<XGBoostNode>(&tree->dtree, adj_mat, weight, target_party_id);
    }
}

void extract_adjacency_matrix_from_tree(RandomForestTree *tree,
                                        SparseMatrixDOK<float> &adj_mat,
                                        float weight,
                                        int target_party_id,
                                        bool is_weighted)
{
    int num_row = tree->dtree.y.size();
    if (is_weighted)
    {
        travase_nodes_to_extract_weighted_adjacency_matrix<RandomForestNode>(&tree->dtree, tree->dtree.depth, adj_mat, weight, target_party_id);
    }
    else
    {
        travase_nodes_to_extract_adjacency_matrix<RandomForestNode>(&tree->dtree, adj_mat, weight, target_party_id);
    }
}

SparseMatrixDOK<float> extract_adjacency_matrix_from_forest(XGBoostBase *model,
                                                            int target_party_id = -1,
                                                            bool is_weighted = true,
                                                            int skip_round = 0,
                                                            float eta = 0.3)
{
    int num_row = model->estimators[0].dtree.y.size();
    SparseMatrixDOK<float> adj_matrix(num_row, num_row, 0.0, true, true);
    for (int i = 0; i < model->estimators.size(); i++)
    {
        if (i >= skip_round)
        {
            extract_adjacency_matrix_from_tree(&model->estimators[i], adj_matrix,
                                               pow(eta, float(i - skip_round)),
                                               target_party_id, is_weighted);
        }
    }
    return adj_matrix;
}

SparseMatrixDOK<float> extract_adjacency_matrix_from_forest(RandomForestClassifier *model,
                                                            int target_party_id = -1,
                                                            bool is_weighted = true,
                                                            int skip_round = 0)
{
    int num_row = model->estimators[0].dtree.y.size();
    SparseMatrixDOK<float> adj_matrix(num_row, num_row, 0.0, true, true);
    for (int i = 0; i < model->estimators.size(); i++)
    {
        if (i >= skip_round)
        {
            extract_adjacency_matrix_from_tree(&model->estimators[i], adj_matrix,
                                               1.0, target_party_id, is_weighted);
        }
    }
    return adj_matrix;
}