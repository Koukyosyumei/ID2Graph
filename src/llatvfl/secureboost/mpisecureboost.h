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

struct MPISecureBoostBase : TreeModelBase<MPISecureBoostParty>
{
    float subsample_cols;
    float min_child_weight;
    int depth;
    int min_leaf;
    float learning_rate;
    int boosting_rounds;
    float lam;
    float gamma;
    float eps;
    int active_party_id;
    int completelly_secure_round;
    float init_value;
    bool save_loss;

    vector<float> init_pred;
    vector<MPISecureBoostTree> estimators;
    vector<float> logging_loss;

    MPISecureBoostParty party_for_training;
    int parties_num_for_training;

    MPISecureBoostBase(float subsample_cols_ = 0.8,
                       float min_child_weight_ = -1 * numeric_limits<float>::infinity(),
                       int depth_ = 5, int min_leaf_ = 5,
                       float learning_rate_ = 0.4, int boosting_rounds_ = 5,
                       float lam_ = 1.5, float gamma_ = 1, float eps_ = 0.1,
                       int active_party_id_ = -1, int completelly_secure_round_ = 0,
                       float init_value_ = 1.0, bool save_loss_ = true)
    {
        subsample_cols = subsample_cols_;
        min_child_weight = min_child_weight_;
        depth = depth_;
        min_leaf = min_leaf_;
        learning_rate = learning_rate_;
        boosting_rounds = boosting_rounds_;
        lam = lam_;
        gamma = gamma_;
        eps = eps_;
        active_party_id = active_party_id_;
        completelly_secure_round = completelly_secure_round_;
        init_value = init_value_;
        save_loss = save_loss_;
    }

    virtual vector<float> get_grad(vector<float> &y_pred, vector<float> &y) = 0;
    virtual vector<float> get_hess(vector<float> &y_pred, vector<float> &y) = 0;
    virtual float get_loss(vector<float> &y_pred, vector<float> &y) = 0;
    virtual vector<float> get_init_pred(vector<float> &y) = 0;

    void load_estimators(vector<MPISecureBoostTree> &_estimators)
    {
        estimators = _estimators;
    }

    void clear()
    {
        estimators.clear();
        logging_loss.clear();
    }

    vector<MPISecureBoostTree> get_estimators()
    {
        return estimators;
    }

    void fit(vector<MPISecureBoostParty> &parties, vector<float> &y)
    {
        try
        {
            throw runtime_error("you should use `fit(MPISecureBoostParty active_party, int parties_num, vector<float> &y)`");
        }
        catch (runtime_error e)
        {
            cerr << e.what() << "\n";
        }
    }

    void fit(MPISecureBoostParty &party, int parties_num)
    {
        party_for_training = party;
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
            if (estimators.size() == 0)
            {
                init_pred = get_init_pred(party.y);
                copy(init_pred.begin(), init_pred.end(), back_inserter(base_pred));
            }
            else
            {
                base_pred.resize(row_count);
                for (int j = 0; j < row_count; j++)
                    base_pred[j] = 0;

                for (int i = 0; i < estimators.size(); i++)
                {
                    vector<float> pred_temp = estimators[i].get_train_prediction();
                    for (int j = 0; j < row_count; j++)
                        base_pred[j] += learning_rate * pred_temp[j];
                }
            }
        }

        for (int i = 0; i < boosting_rounds; i++)
        {
            if (party.party_id == active_party_id)
            {
                vector<float> plain_grad = get_grad(base_pred, party.y);
                vector<float> plain_hess = get_hess(base_pred, party.y);
                party.set_plain_gradients_and_hessians(plain_grad, plain_hess);

                for (int j = 0; j < row_count; j++)
                {
                    party.gradient[j] = party.pk.encrypt<float>(party.plain_gradient[j]);
                    party.hessian[j] = party.pk.encrypt<float>(party.plain_hessian[j]);
                }

                MPISecureBoostTree boosting_tree = MPISecureBoostTree();
                boosting_tree.fit(party, parties_num, party.y, min_child_weight, lam,
                                  gamma, eps, min_leaf, depth, active_party_id, (completelly_secure_round > i));
                for (int p = 0; p < parties_num; p++)
                {
                    if (p != active_party_id)
                    {
                        party.world.send(p, TAG_DEPTH, -1);
                    }
                }

                vector<float> pred_temp = boosting_tree.get_train_prediction();
                for (int j = 0; j < row_count; j++)
                    base_pred[j] += learning_rate * pred_temp[j];

                estimators.push_back(boosting_tree);

                if (save_loss)
                {
                    logging_loss.push_back(get_loss(base_pred, party.y));
                }
            }
            else
            {
                party.run_as_passive();
            }

            party.world.barrier();
        }
    }

    vector<float> predict_raw(vector<vector<float>> &X)
    {
        int row_count = X.size();
        // int estimators_num = estimators.size();
        vector<float> y_pred(row_count, init_value);
        int is_leaf_flag;
        int temp_node_party_id = 0;
        int temp_record_id = 0;
        bool temp_is_left;

        for (int i = 0; i < boosting_rounds; i++)
        {
            for (int j = 0; j < row_count; j++)
            {

                if (party_for_training.party_id == active_party_id)
                {
                    queue<MPISecureBoostNode *> que;
                    que.push(&estimators[i].dtree);

                    MPISecureBoostNode *temp_node;
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
                                party_for_training.world.send(p, TAG_ISLEAF, is_leaf_flag);
                            }
                        }

                        if (is_leaf_flag == 1)
                        {
                            y_pred[j] = learning_rate * temp_node->val;
                            break;
                        }
                        else
                        {

                            for (int p = 0; p < parties_num_for_training; p++)
                            {
                                if (p != active_party_id)
                                {
                                    party_for_training.world.send(p, TAG_NODE_PARTY_ID, temp_node->party_id);
                                }
                            }

                            if (temp_node->party_id == active_party_id)
                            {
                                temp_is_left = party_for_training.is_left(temp_node->record_id, X[j]);
                            }
                            else
                            {
                                party_for_training.world.send(temp_node->party_id, TAG_RECORD_ID, temp_node->record_id);
                                party_for_training.world.recv(temp_node->party_id, TAG_ISLEFT, temp_is_left);
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
                        party_for_training.world.recv(active_party_id, TAG_ISLEAF, is_leaf_flag);
                        if (is_leaf_flag == 1)
                        {
                            break;
                        }
                        else
                        {
                            party_for_training.world.recv(active_party_id, TAG_NODE_PARTY_ID, temp_node_party_id);
                            if (temp_node_party_id == party_for_training.party_id)
                            {
                                party_for_training.world.recv(active_party_id, TAG_RECORD_ID, temp_record_id);
                                temp_is_left = party_for_training.is_left(temp_record_id, X[j]);
                                party_for_training.world.send(active_party_id, TAG_ISLEFT, temp_is_left);
                            }
                        }
                    }
                }
                party_for_training.world.barrier();
            }
        }
        party_for_training.world.barrier();

        return y_pred;
    }
};

struct MPISecureBoostClassifier : public MPISecureBoostBase
{
    using MPISecureBoostBase::MPISecureBoostBase;

    float get_loss(vector<float> &y_pred, vector<float> &y)
    {
        float loss = 0;
        float n = y_pred.size();
        for (int i = 0; i < n; i++)
        {
            if (y[i] == 1)
            {
                loss += log(1 + exp(-1 * y_pred[i])) / n;
            }
            else
            {
                loss += log(1 + exp(y_pred[i])) / n;
            }
        }
        return loss;
    }

    vector<float> get_grad(vector<float> &y_pred, vector<float> &y)
    {
        int element_num = y_pred.size();
        vector<float> grad(element_num);
        for (int i = 0; i < element_num; i++)
            grad[i] = sigmoid(y_pred[i]) - y[i];
        return grad;
    }

    vector<float> get_hess(vector<float> &y_pred, vector<float> &y)
    {
        int element_num = y_pred.size();
        vector<float> hess(element_num);
        for (int i = 0; i < element_num; i++)
        {
            float temp_proba = sigmoid(y_pred[i]);
            hess[i] = temp_proba * (1 - temp_proba);
        }
        return hess;
    }

    vector<float> get_init_pred(vector<float> &y)
    {
        vector<float> init_pred(y.size(), init_value);
        return init_pred;
    }

    vector<float> predict_proba(vector<vector<float>> &x)
    {
        vector<float> predicted_probas;
        vector<float> raw_score = predict_raw(x);

        if (party_for_training.party_id == active_party_id)
        {
            int row_count = x.size();
            predicted_probas.resize(row_count);
            for (int i = 0; i < row_count; i++)
                predicted_probas[i] = sigmoid(raw_score[i]);
        }

        return predicted_probas;
    }
};