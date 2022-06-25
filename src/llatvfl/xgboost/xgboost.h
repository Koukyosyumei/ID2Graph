#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include <cmath>
#include "../core/model.h"
#include "tree.h"
using namespace std;

struct XGBoostBase : TreeModelBase<XGBoostParty>
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
    int n_job;
    bool save_loss;

    vector<float> init_pred;
    vector<XGBoostTree> estimators;
    vector<float> logging_loss;

    XGBoostBase(float subsample_cols_ = 0.8,
                float min_child_weight_ = -1 * numeric_limits<float>::infinity(),
                int depth_ = 5, int min_leaf_ = 5,
                float learning_rate_ = 0.4, int boosting_rounds_ = 5,
                float lam_ = 1.5, float gamma_ = 1, float eps_ = 0.1,
                int active_party_id_ = -1, int completelly_secure_round_ = 0,
                float init_value_ = 1.0, int n_job_ = 1, bool save_loss_ = true)
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
        n_job = n_job_;
        save_loss = save_loss_;
    }

    virtual vector<float> get_grad(vector<float> &y_pred, vector<float> &y) = 0;
    virtual vector<float> get_hess(vector<float> &y_pred, vector<float> &y) = 0;
    virtual float get_loss(vector<float> &y_pred, vector<float> &y) = 0;
    virtual vector<float> get_init_pred(vector<float> &y) = 0;

    void load_estimators(vector<XGBoostTree> _estimators)
    {
        estimators = _estimators;
    }

    vector<XGBoostTree> get_estimators()
    {
        return estimators;
    }

    void fit(vector<XGBoostParty> &parties, vector<float> &y)
    {
        int row_count = y.size();
        vector<float> base_pred;
        if (estimators.size() == 0)
        {
            init_pred = get_init_pred(y);
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

        for (int i = 0; i < boosting_rounds; i++)
        {
            vector<float> grad = get_grad(base_pred, y);
            vector<float> hess = get_hess(base_pred, y);

            XGBoostTree boosting_tree = XGBoostTree();
            boosting_tree.fit(&parties, y, grad, hess, min_child_weight,
                              lam, gamma, eps, min_leaf, depth, active_party_id, (completelly_secure_round > i), n_job);
            vector<float> pred_temp = boosting_tree.get_train_prediction();
            for (int j = 0; j < row_count; j++)
                base_pred[j] += learning_rate * pred_temp[j];

            estimators.push_back(boosting_tree);

            if (save_loss)
            {
                logging_loss.push_back(get_loss(base_pred, y));
            }
        }
    }

    vector<float> predict_raw(vector<vector<float>> &X)
    {
        int row_count = X.size();
        vector<float> y_pred;
        copy(init_pred.begin(), init_pred.end(), back_inserter(y_pred));
        int estimators_num = estimators.size();
        for (int i = 0; i < estimators_num; i++)
        {
            vector<float> y_pred_temp = estimators[i].predict(X);
            for (int j = 0; j < row_count; j++)
                y_pred[j] += learning_rate * y_pred_temp[j];
        }

        return y_pred;
    }
};

struct XGBoostClassifier : public XGBoostBase
{
    using XGBoostBase::XGBoostBase;

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
        vector<float> raw_score = predict_raw(x);
        int row_count = x.size();
        vector<float> predicted_probas(row_count);
        for (int i = 0; i < row_count; i++)
            predicted_probas[i] = sigmoid(raw_score[i]);
        return predicted_probas;
    }
};
