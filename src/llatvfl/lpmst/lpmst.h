#pragma once
#include <algorithm>
#include "rrp.h"
#include "../randomforest/randomforest.h"
#include "../xgboost/xgboost.h"
#include "../secureboost/secureboost.h"
using namespace std;

struct LPMST
{
    int M = 1;
    float epsilon = 1.0;
    int seed = 0;

    RRWithPrior rrp;

    LPMST(){};
    LPMST(int M_ = 1.0, float epsilon_ = 1.0, int seed_ = 0)
    {
        M = M_;
        epsilon = epsilon_;
        seed = seed_;
    }

    void fit(XGBoostClassifier &clf, vector<XGBoostParty> &parties, vector<float> &y)
    {
        _fit<XGBoostClassifier, XGBoostParty>(clf, parties, y);
    }
    void fit(SecureBoostClassifier &clf, vector<SecureBoostParty> &parties, vector<float> &y)
    {
        _fit<SecureBoostClassifier, SecureBoostParty>(clf, parties, y);
    }
    void fit(RandomForestClassifier &clf, vector<RandomForestParty> &parties, vector<float> &y)
    {
        _fit<RandomForestClassifier, RandomForestParty>(clf, parties, y);
    }

    template <typename ModelType, typename PartyType>
    void _fit(ModelType &model, vector<PartyType> &parties, vector<float> &y)
    {
        int y_size = y.size();
        int class_num = *max_element(y.begin(), y.end()) + 1;
        vector<float> prior_dist(class_num, 1.0 / float(class_num));

        if (M == 1)
        {
            rrp = RRWithPrior(epsilon, prior_dist, seed);
            vector<float> y_hat(y_size);
            for (int i = 0; i < y_size; i++)
            {
                y_hat[i] = rrp.rrtop_k(y[i]);
            }

            model.fit(parties, y_hat);
        }
        // TODO: support LP-MST (M >= 2)
    }
};