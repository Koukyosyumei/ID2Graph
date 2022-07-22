#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include "../core/nodeapi.h"
using namespace std;

template <typename NodeType>
struct Tree
{
    NodeType dtree;
    NodeAPI<NodeType> nodeapi;

    Tree() {}

    NodeType &get_root_node()
    {
        return *dtree;
    }

    vector<float> predict(vector<vector<float>> &X)
    {
        return nodeapi.predict(&dtree, X);
    }

    vector<pair<vector<int>, vector<float>>> extract_train_prediction_from_node(NodeType &node)
    {
        if (node.is_leaf())
        {
            vector<pair<vector<int>, vector<float>>> result;
            result.push_back(make_pair(node.idxs,
                                       vector<float>(node.idxs.size(),
                                                     node.val)));
            return result;
        }
        else
        {
            vector<pair<vector<int>, vector<float>>> left_result =
                extract_train_prediction_from_node(*node.left);
            vector<pair<vector<int>, vector<float>>> right_result =
                extract_train_prediction_from_node(*node.right);
            left_result.insert(left_result.end(), right_result.begin(), right_result.end());
            return left_result;
        }
    }

    vector<float> get_train_prediction()
    {
        vector<pair<vector<int>, vector<float>>> result = extract_train_prediction_from_node(dtree);
        vector<float> y_train_pred(dtree.y.size());
        for (int i = 0; i < result.size(); i++)
        {
            for (int j = 0; j < result[i].first.size(); j++)
            {
                y_train_pred[result[i].first[j]] = result[i].second[j];
            }
        }

        return y_train_pred;
    }

    string print(bool show_purity = false, bool binary_color = true, int target_party_id = -1)
    {
        return nodeapi.print(&dtree, show_purity, binary_color, target_party_id);
    }

    float get_leaf_purity()
    {
        return nodeapi.get_leaf_purity(&dtree, dtree.idxs.size());
    }
};
