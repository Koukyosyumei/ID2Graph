#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include <cmath>
#include "../core/model.h"
#include "tree.h"
using namespace std;

struct RandomForestClassifier : TreeModelBase<RandomForestParty>
{
    float subsample_cols;
    int depth;
    int min_leaf;
    float max_samples_ratio;
    int num_trees;
    bool use_ispure;
    int active_party_id;
    int n_job;
    int seed;

    vector<RandomForestTree> estimators;

    RandomForestClassifier(float subsample_cols_ = 0.8, int depth_ = 5, int min_leaf_ = 1,
                           float max_samples_ratio_ = 1.0, int num_trees_ = 5,
                           int active_party_id_ = -1, int n_job_ = 1, int seed_ = 0)
    {
        subsample_cols = subsample_cols_;
        depth = depth_;
        min_leaf = min_leaf_;
        max_samples_ratio = max_samples_ratio_;
        num_trees = num_trees_;
        active_party_id = active_party_id_;
        n_job = n_job_;
        seed = seed_;
    }

    void load_estimators(vector<RandomForestTree> _estimators)
    {
        estimators = _estimators;
    }

    vector<RandomForestTree> get_estimators()
    {
        return estimators;
    }

    void fit(vector<RandomForestParty> &parties, vector<float> &y)
    {
        int row_count = y.size();

        for (int i = 0; i < num_trees; i++)
        {
            RandomForestTree boosting_tree = RandomForestTree();
            boosting_tree.fit(&parties, y, min_leaf, depth, max_samples_ratio, active_party_id, n_job, seed);
            estimators.push_back(boosting_tree);
            seed += 1;
        }
    }

    // retuen the average score of all trees (sklearn-style)
    vector<float> predict_raw(vector<vector<float>> &X)
    {
        int row_count = X.size();
        vector<float> y_pred(row_count, 0);
        int estimators_num = estimators.size();
        for (int i = 0; i < estimators_num; i++)
        {
            vector<float> y_pred_temp = estimators[i].predict(X);
            for (int j = 0; j < row_count; j++)
                y_pred[j] += y_pred_temp[j] / float(estimators_num);
        }

        return y_pred;
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
