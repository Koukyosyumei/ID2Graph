#pragma once
#include <iostream>
#include <limits>
#include <vector>
#include <queue>
#include "../utils/dok.h"
#include "../randomforest/randomforest.h"
#include "../xgboost/xgboost.h"
#include "../secureboost/secureboost.h"
using namespace std;

/**
 * @brief Travase nodes while updating the adjacency matrix with constant weight.
 *
 * @tparam NodeType NodeType The type of Node.
 * @param node The node where you want to start travasing
 * @param adj_mat The space adhacency matrix to be updated
 * @param weight The weight parameter
 * @param target_party_id The target party id
 * @return true
 * @return false
 */
template <typename NodeType>
inline void travase_nodes_to_extract_adjacency_matrix(NodeType *node,
                                                      int max_depth,
                                                      int start_depth,
                                                      SparseMatrixDOK<float> &adj_mat,
                                                      float weight,
                                                      int target_party_id)
{
    queue<NodeType *> que;
    que.push(node);
    NodeType *temp_node;
    int temp_idxs_size;
    while (!que.empty())
    {
        // skip_flag = false;
        temp_node = que.front();
        que.pop();

        if (temp_node->is_leaf_flag)
        {
            // skip_flag = temp_node->depth <= 0 && target_party_id != -1 && temp_node->party_id != target_party_id;
            if (!temp_node->not_splitted_flag || target_party_id == -1)
            {
                temp_idxs_size = temp_node->idxs.size();
                for (int i = 0; i < temp_idxs_size; i++)
                {
                    for (int j = i + 1; j < temp_idxs_size; j++)
                    {
                        adj_mat.add(temp_node->idxs[i], temp_node->idxs[j], weight);
                    }
                }
            }
        }
        else
        {
            // left_skip_flag = temp_node->left->is_leaf() && target_party_id != -1 && temp_node->left->party_id != target_party_id;
            // right_skip_flag = temp_node->right->is_leaf() && target_party_id != -1 && temp_node->right->party_id != target_party_id;

            if ((((temp_node->left->not_splitted_flag &&
                   temp_node->right->not_splitted_flag)) ||
                 (temp_node->left->lmir_flag_exclude_passive_parties &&
                  temp_node->right->lmir_flag_exclude_passive_parties)) &&
                (target_party_id != -1))
            {
                temp_idxs_size = temp_node->idxs.size();
                for (int i = 0; i < temp_idxs_size; i++)
                {
                    for (int j = i + 1; j < temp_idxs_size; j++)
                    {
                        adj_mat.add(temp_node->idxs[i], temp_node->idxs[j], weight);
                    }
                }
            }

            if (!temp_node->left->lmir_flag_exclude_passive_parties ||
                !temp_node->right->lmir_flag_exclude_passive_parties)
            {
                que.push(temp_node->left);
                que.push(temp_node->right);
            }
        }

        // temp_node->idxs.clear();
        // temp_node->idxs.shrink_to_fit();
        temp_node->val.clear();
        temp_node->val.shrink_to_fit();
    }
}

/**
 * @brief Update adjacency matrix with given tree.
 *
 * @param tree The tree to be transformed to a graph.
 * @param adj_mat The adjancecy matrix to be updated.
 * @param weight The weight parameter.
 * @param target_party_id The target party id.
 */
inline void extract_adjacency_matrix_from_tree(XGBoostTree *tree,
                                               int start_depth,
                                               SparseMatrixDOK<float> &adj_mat,
                                               float weight,
                                               int target_party_id)
{
    travase_nodes_to_extract_adjacency_matrix<XGBoostNode>(&tree->dtree, tree->dtree.depth, start_depth, adj_mat, weight, target_party_id);
}

/**
 * @brief Update adjacency matrix with given tree.
 *
 * @param tree The tree to be transformed to a graph.
 * @param adj_mat The adjancecy matrix to be updated.
 * @param weight The weight parameter.
 * @param target_party_id The target party id.
 */
inline void extract_adjacency_matrix_from_tree(SecureBoostTree *tree,
                                               int start_depth,
                                               SparseMatrixDOK<float> &adj_mat,
                                               float weight,
                                               int target_party_id)
{
    travase_nodes_to_extract_adjacency_matrix<SecureBoostNode>(&tree->dtree, tree->dtree.depth, start_depth, adj_mat, weight, target_party_id);
}

/**
 * @brief Update adjacency matrix with given tree.
 *
 * @param tree The tree to be transformed to a graph.
 * @param adj_mat The adjancecy matrix to be updated.
 * @param weight The weight parameter.
 * @param target_party_id The target party id.
 */
inline void extract_adjacency_matrix_from_tree(RandomForestTree *tree,
                                               int start_depth,
                                               SparseMatrixDOK<float> &adj_mat,
                                               float weight,
                                               int target_party_id)
{
    travase_nodes_to_extract_adjacency_matrix<RandomForestNode>(&tree->dtree, tree->dtree.depth, start_depth, adj_mat, weight, target_party_id);
}

/**
 * @brief Extract adjacency matrix from the trained model
 *
 * @param model The target tree-based model
 * @param target_party_id The target party id. He cannot observe the leaf split information.
 * @param skip_round The number of skipped rounds.
 * @param eta The discount factor.
 * @return SparseMatrixDOK<float>
 */
inline SparseMatrixDOK<float> extract_adjacency_matrix_from_forest(XGBoostBase *model,
                                                                   int start_depth,
                                                                   int target_party_id = -1,
                                                                   int skip_round = 0,
                                                                   float eta = 0.3)
{
    int num_row = model->estimators[0].num_row;
    SparseMatrixDOK<float> adj_matrix(num_row, num_row, 0.0, true, true);
    for (int i = 0; i < model->estimators.size(); i++)
    {
        if (i >= skip_round)
        {
            extract_adjacency_matrix_from_tree(&model->estimators[i], start_depth,
                                               adj_matrix,
                                               pow(eta, float(i - skip_round)),
                                               target_party_id);
        }
    }
    return adj_matrix;
}

/**
 * @brief Extract adjacency matrix from the trained model
 *
 * @param model The target tree-based model
 * @param target_party_id The target party id. He cannot observe the leaf split information.
 * @param skip_round The number of skipped rounds.
 * @param eta The discount factor.
 * @return SparseMatrixDOK<float>
 */
inline SparseMatrixDOK<float> extract_adjacency_matrix_from_forest(SecureBoostBase *model,
                                                                   int start_depth,
                                                                   int target_party_id = -1,
                                                                   int skip_round = 0,
                                                                   float eta = 0.3)
{
    int num_row = model->estimators[0].num_row;
    SparseMatrixDOK<float> adj_matrix(num_row, num_row, 0.0, true, true);
    for (int i = 0; i < model->estimators.size(); i++)
    {
        if (i >= skip_round)
        {
            extract_adjacency_matrix_from_tree(&model->estimators[i], start_depth,
                                               adj_matrix,
                                               pow(eta, float(i - skip_round)),
                                               target_party_id);
        }
    }
    return adj_matrix;
}

/**
 * @brief Extract adjacency matrix from the trained model
 *
 * @param model The target tree-based model
 * @param target_party_id The target party id. He cannot observe the leaf split information.
 * @param skip_round The number of skipped rounds.
 * @param eta The discount factor.
 * @return SparseMatrixDOK<float>
 */
inline SparseMatrixDOK<float> extract_adjacency_matrix_from_forest(RandomForestClassifier *model,
                                                                   int start_depth,
                                                                   int target_party_id = -1,
                                                                   int skip_round = 0)
{
    int num_row = model->estimators[0].num_row;
    SparseMatrixDOK<float> adj_matrix(num_row, num_row, 0.0, true, true);
    for (int i = 0; i < model->estimators.size(); i++)
    {
        if (i >= skip_round)
        {
            extract_adjacency_matrix_from_tree(&model->estimators[i], start_depth,
                                               adj_matrix,
                                               1.0, target_party_id);
        }
    }
    return adj_matrix;
}