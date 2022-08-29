#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "../core/model.h"
#include "../attack/pipe.h"
#include "tree_backdoor.h"
using namespace std;

struct RandomForestBackDoorClassifier : TreeModelBase<RandomForestBackDoorParty>
{
    float subsample_cols;
    int num_classes;
    int depth;
    int min_leaf;
    float max_samples_ratio;
    int num_trees;
    float mi_bound;
    int active_party_id;
    int n_job;
    int seed;

    int attack_start_round;
    int attack_start_depth;
    int target_party_id;
    int skip_round;
    float epsilon_random_unfolding;
    int seconds_wait4timeout;
    int max_timeout_num_patience;

    float upsilon_Y;

    vector<RandomForestBackDoorTree> estimators;
    vector<int> estimated_clusters;
    vector<int> matched_target_labels_idxs;

    RandomForestBackDoorClassifier(int num_classes_ = 2, float subsample_cols_ = 0.8, int depth_ = 5, int min_leaf_ = 1,
                                   float max_samples_ratio_ = 1.0, int num_trees_ = 5,
                                   float mi_bound_ = numeric_limits<float>::infinity(),
                                   int active_party_id_ = -1, int n_job_ = 1, int seed_ = 0,
                                   int attack_start_round_ = 3,
                                   int attack_start_depth_ = -1, int target_party_id_ = 1,
                                   int skip_round_ = 0, float epsilon_random_unfolding_ = 0.0,
                                   int seconds_wait4timeout_ = 10, int max_timeout_num_patience_ = 5)
    {
        num_classes = num_classes_;
        subsample_cols = subsample_cols_;
        depth = depth_;
        min_leaf = min_leaf_;
        max_samples_ratio = max_samples_ratio_;
        num_trees = num_trees_;
        mi_bound = mi_bound_;
        active_party_id = active_party_id_;
        n_job = n_job_;
        seed = seed_;

        attack_start_round = attack_start_round_;
        attack_start_depth = attack_start_depth_;
        target_party_id = target_party_id_;
        skip_round = skip_round_;
        epsilon_random_unfolding = epsilon_random_unfolding_;
        seconds_wait4timeout = seconds_wait4timeout_;
        max_timeout_num_patience = max_timeout_num_patience_;

        if (mi_bound < 0)
        {
            mi_bound = numeric_limits<float>::infinity();
        }
    }

    void load_estimators(vector<RandomForestBackDoorTree> &_estimators)
    {
        estimators = _estimators;
    }

    void clear()
    {
        estimators.clear();
    }

    vector<RandomForestBackDoorTree> get_estimators()
    {
        return estimators;
    }

    void fit(vector<RandomForestBackDoorParty> &parties, vector<float> &y)
    {
        int row_count = y.size();

        vector<float> prior(num_classes, 0);
        for (int j = 0; j < row_count; j++)
        {
            prior[y[j]] += 1;
        }
        for (int c = 0; c < num_classes; c++)
        {
            prior[c] /= float(row_count);
        }

        upsilon_Y = *min_element(prior.begin(), prior.end());
        float mi_delta = sqrt(upsilon_Y * mi_bound / 2);

        for (int i = 0; i < num_trees; i++)
        {
            RandomForestBackDoorTree tree = RandomForestBackDoorTree();
            tree.fit(&parties, y, num_classes, min_leaf, depth, prior, max_samples_ratio, mi_delta, active_party_id, n_job, seed);
            estimators.push_back(tree);
            seed += 1;

            if (i == attack_start_round - 1)
            {
                QuickAttackPipeline qap = QuickAttackPipeline(num_classes, attack_start_depth, 1, skip_round,
                                                              epsilon_random_unfolding, seconds_wait4timeout,
                                                              max_timeout_num_patience);

                vector<float> class_cnts(num_classes);
                for (int c = 0; c < num_classes; c++)
                {
                    class_cnts[c] = count(y.begin(), y.end(), c);
                }

                estimated_clusters = qap.attack<RandomForestBackDoorClassifier>(*this, parties[1].x);
                matched_target_labels_idxs = qap.match_prior_and_estimatedclusters(class_cnts, estimated_clusters, 1);
                parties[1].set_matched_target_labels_idxs_(matched_target_labels_idxs);
            }
        }
    }

    // retuen the average score of all trees (sklearn-style)
    vector<vector<float>> predict_raw(vector<vector<float>> &X)
    {
        int row_count = X.size();
        vector<vector<float>> y_pred(row_count, vector<float>(num_classes, 0));
        int estimators_num = estimators.size();
        for (int i = 0; i < estimators_num; i++)
        {
            vector<vector<float>> y_pred_temp = estimators[i].predict(X);
            for (int j = 0; j < row_count; j++)
            {
                for (int c = 0; c < num_classes; c++)
                {
                    y_pred[j][c] += y_pred_temp[j][c] / float(estimators_num);
                }
            }
        }

        return y_pred;
    }

    vector<vector<float>> predict_proba(vector<vector<float>> &x)
    {
        return predict_raw(x);
    }
};
