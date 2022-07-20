#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include <cmath>
#include "../paillier/paillier.h"
#include "../core/model.h"
#include "mpitree.h"
using namespace std;

struct MPIRandomForestClassifier : TreeModelBase<MPIRandomForestParty>
{
    float subsample_cols;
    int depth;
    int min_leaf;
    float max_samples_ratio;
    int num_trees;
    int active_party_id;
    int seed;

    vector<MPIRandomForestTree>
        estimators;

    MPIRandomForestParty *party_for_training;
    int parties_num_for_training;

    MPIRandomForestClassifier(float subsample_cols_ = 0.8, int depth_ = 5, int min_leaf_ = 1,
                              float max_samples_ratio_ = 1.0, int num_trees_ = 5,
                              int active_party_id_ = -1, int seed_ = 0)
    {
        subsample_cols = subsample_cols_;
        depth = depth_;
        min_leaf = min_leaf_;
        max_samples_ratio = max_samples_ratio_;
        num_trees = num_trees_;
        active_party_id = active_party_id_;
        seed = seed_;
    }

    void load_estimators(vector<MPIRandomForestTree> &_estimators)
    {
        estimators = _estimators;
    }

    void clear()
    {
        estimators.clear();
    }

    vector<MPIRandomForestTree> get_estimators()
    {
        return estimators;
    }

    void fit(vector<MPIRandomForestParty> &parties, vector<float> &y)
    {
        try
        {
            throw runtime_error("you should use `fit(MPIRandomForestParty active_party, int parties_num, vector<float> &y)`");
        }
        catch (runtime_error e)
        {
            cerr << e.what() << "\n";
        }
    }

    void fit(MPIRandomForestParty &party, int parties_num)
    {
        party_for_training = &party;
        parties_num_for_training = parties_num;

        int row_count;
        vector<float> base_pred;
        if (party.party_id == active_party_id)
        {
            try
            {
                if ((active_party_id < 0) || (active_party_id > parties_num))
                {
                    throw invalid_argument("invalid active_party_id");
                }
            }
            catch (std::exception &e)
            {
                std::cout << e.what() << std::endl;
            }

            row_count = party.y.size();
        }

        for (int i = 0; i < num_trees; i++)
        {
            if (party.party_id == active_party_id)
            {
                for (int j = 0; j < row_count; j++)
                {
                    party.y_encrypted[j] = party.pk.encrypt<float>(party.y[j]);
                }

                MPIRandomForestTree tree = MPIRandomForestTree();
                tree.fit(&party, parties_num, min_leaf, depth, max_samples_ratio, active_party_id, false, seed);
                for (int p = 0; p < parties_num; p++)
                {
                    if (p != active_party_id)
                    {
                        party.world.send(p, TAG_DEPTH, -1);
                    }
                }
                estimators.push_back(tree);
            }
            else
            {
                party.run_as_passive();
            }

            party.world.barrier();
        }
    }

    vector<float> predict_raw(vector<vector<float>> &X_new)
    {
        int row_count = X_new.size();
        vector<float> y_pred(row_count, 0);
        int is_leaf_flag;
        int temp_node_party_id = 0;
        int temp_record_id = 0;
        bool temp_is_left;

        for (int i = 0; i < num_trees; i++)
        {
            for (int j = 0; j < row_count; j++)
            {

                if (party_for_training->party_id == active_party_id)
                {
                    queue<MPIRandomForestNode *> que;
                    que.push(&estimators[i].dtree);

                    MPIRandomForestNode *temp_node;
                    while (!que.empty())
                    {
                        temp_node = que.front();
                        que.pop();

                        if (temp_node->is_leaf())
                        {
                            is_leaf_flag = 1;
                        }
                        else
                        {
                            is_leaf_flag = 0;
                        }

                        for (int p = 0; p < parties_num_for_training; p++)
                        {
                            if (p != active_party_id)
                            {
                                party_for_training->world.send(p, TAG_ISLEAF, is_leaf_flag);
                            }
                        }

                        if (is_leaf_flag == 1)
                        {
                            y_pred[j] += temp_node->val / float(num_trees);
                            break;
                        }
                        else
                        {

                            for (int p = 0; p < parties_num_for_training; p++)
                            {
                                if (p != active_party_id)
                                {
                                    party_for_training->world.send(p, TAG_NODE_PARTY_ID, temp_node->party_id);
                                }
                            }

                            if (temp_node->party_id == active_party_id)
                            {
                                temp_is_left = party_for_training->is_left(temp_node->record_id, X_new[j]);
                            }
                            else
                            {
                                party_for_training->world.send(temp_node->party_id, TAG_RECORD_ID, temp_node->record_id);
                                party_for_training->world.recv(temp_node->party_id, TAG_ISLEFT, temp_is_left);
                            }

                            if (temp_is_left)
                            {
                                que.push(temp_node->left);
                            }
                            else
                            {
                                que.push(temp_node->right);
                            }
                        }
                    }
                }
                else
                {
                    while (true)
                    {
                        party_for_training->world.recv(active_party_id, TAG_ISLEAF, is_leaf_flag);
                        if (is_leaf_flag == 1)
                        {
                            break;
                        }
                        else
                        {
                            party_for_training->world.recv(active_party_id, TAG_NODE_PARTY_ID, temp_node_party_id);
                            if (temp_node_party_id == party_for_training->party_id)
                            {
                                party_for_training->world.recv(active_party_id, TAG_RECORD_ID, temp_record_id);
                                temp_is_left = party_for_training->is_left(temp_record_id, X_new[j]);
                                party_for_training->world.send(active_party_id, TAG_ISLEFT, temp_is_left);
                            }
                        }
                    }
                }
                party_for_training->world.barrier();
            }
            party_for_training->world.barrier();
        }
        party_for_training->world.barrier();

        return y_pred;
    }

    vector<float> predict_proba(vector<vector<float>> &x)
    {
        vector<float> predicted_probas;
        vector<float> raw_score = predict_raw(x);

        if (party_for_training->party_id == active_party_id)
        {
            int row_count = x.size();
            predicted_probas.resize(row_count);
            for (int i = 0; i < row_count; i++)
                predicted_probas[i] = sigmoid(raw_score[i]);
        }

        return predicted_probas;
    }
};