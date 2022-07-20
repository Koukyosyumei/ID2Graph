#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <numeric>
#include <string>
#include <cassert>
#include <future>
#include <utility>
#include <chrono>
#include <unistd.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include "llatvfl/randomforest/mpirandomforest.h"
#include "llatvfl/paillier/keygenerator.h"
#include "llatvfl/utils/metric.h"
#include "gtest/gtest.h"
using namespace std;

TEST(MPIRandomForest, MPIRandomForestClassifierTest)
{
    const int min_leaf = 1;
    const int depth = 2;
    const int num_trees = 1;
    const float subsample_cols = 1.0;
    const float max_samples_ratio = 1.0;
    const int key_bitsize = 512;
    const int active_party_id = 0;

    // --- Load Data --- //
    int num_row = 8;
    int num_col = 2;
    int num_party = 2;

    vector<float> y = {1, 0, 1, 0, 1, 1, 0, 1};
    vector<vector<float>> X = {{12, 1},
                               {32, 1},
                               {15, 0},
                               {24, 0},
                               {20, 1},
                               {25, 1},
                               {17, 0},
                               {16, 1}};
    vector<vector<int>> feature_idxs = {{0}, {1}};

    boost::mpi::environment env(true);
    boost::mpi::communicator world;
    int my_rank = world.rank();
    MPIRandomForestParty my_party;

    for (int i = 0; i < num_party; i++)
    {
        if (i == my_rank)
        {
            int num_col = feature_idxs[i].size();
            vector<vector<float>> x(num_row, vector<float>(num_col));
            for (int j = 0; j < num_col; j++)
            {
                for (int k = 0; k < num_row; k++)
                {
                    x[k][j] = X[k][feature_idxs[i][j]];
                }
            }
            my_party = MPIRandomForestParty(world, x, feature_idxs[i], my_rank, depth, num_trees, min_leaf, subsample_cols);
        }
    }

    MPIRandomForestClassifier clf = MPIRandomForestClassifier(subsample_cols, depth, min_leaf,
                                                              max_samples_ratio, num_trees,
                                                              active_party_id);

    if (my_rank == 0)
    {
        PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
        pair<PaillierPublicKey, PaillierSecretKey> keypair = keygenerator.generate_keypair();
        PaillierPublicKey pk = keypair.first;
        PaillierSecretKey sk = keypair.second;

        my_party.set_publickey(pk);
        my_party.set_secretkey(sk);

        for (int j = 1; j < num_party; j++)
        {
            world.send(j, TAG_PUBLICKEY, pk);
        }

        my_party.y = y;
    }
    else
    {
        PaillierPublicKey pk;
        world.recv(0, TAG_PUBLICKEY, pk);
        my_party.set_publickey(pk);
        my_party.pk.init_distribution();
    }

    world.barrier();

    clf.fit(my_party, num_party);
    vector<float> predict_raw = clf.predict_raw(X);
    vector<float> predict_proba = clf.predict_proba(X);

    if (my_rank == 0)
    {
        ASSERT_NEAR(clf.estimators[0].dtree.giniimp, 0.46875, 1e-6);
        ASSERT_NEAR(clf.estimators[0].dtree.score, 0.16875, 1e-6);
        ASSERT_EQ(clf.estimators[0].dtree.best_party_id, 0);
        ASSERT_EQ(clf.estimators[0].dtree.best_col_id, 0);
        ASSERT_EQ(clf.estimators[0].dtree.best_threshold_id, 2);

        vector<int> test_idxs_left = {0, 2, 7};
        vector<int> test_idxs_right = {1, 3, 4, 5, 6};
        vector<int> idxs_left = clf.estimators[0].dtree.left->idxs;
        sort(idxs_left.begin(), idxs_left.end());
        ASSERT_EQ(idxs_left.size(), test_idxs_left.size());
        for (int i = 0; i < idxs_left.size(); i++)
        {
            ASSERT_EQ(idxs_left[i], test_idxs_left[i]);
        }

        vector<int> idxs_right = clf.estimators[0].dtree.right->idxs;
        sort(idxs_right.begin(), idxs_right.end());
        ASSERT_EQ(idxs_right.size(), test_idxs_right.size());
        for (int i = 0; i < idxs_right.size(); i++)
        {
            ASSERT_EQ(idxs_right[i], test_idxs_right[i]);
        }

        ASSERT_EQ(clf.estimators[0].dtree.right->depth, 1);
        ASSERT_EQ(clf.estimators[0].dtree.left->is_leaf(), 1);
        ASSERT_EQ(clf.estimators[0].dtree.right->is_leaf(), 0);

        vector<int> test_idxs_right_left = {3, 6};
        vector<int> test_idxs_right_right = {1, 4, 5};
        vector<int> idxs_right_left = clf.estimators[0].dtree.right->left->idxs;
        vector<int> idxs_right_right = clf.estimators[0].dtree.right->right->idxs;
        sort(idxs_right_left.begin(), idxs_right_left.end());
        sort(idxs_right_right.begin(), idxs_right_right.end());
        for (int i = 0; i < test_idxs_right_left.size(); i++)
        {
            ASSERT_EQ(test_idxs_right_left[i], idxs_right_left[i]);
        }
        for (int i = 0; i < test_idxs_right_right.size(); i++)
        {
            ASSERT_EQ(test_idxs_right_right[i], idxs_right_right[i]);
        }

        vector<float> test_predict_raw = {1, 2.0 / 3.0, 1, 0, 2.0 / 3.0, 2.0 / 3.0, 0, 1};
        for (int i = 0; i < predict_raw.size(); i++)
        {
            ASSERT_EQ(test_predict_raw[i], predict_raw[i]);
        }

        vector<float> test_predict_proba = {0.7310585786300049, 0.6607563732194243,
                                            0.7310585786300049, 0.5,
                                            0.6607563732194243, 0.6607563732194243,
                                            0.5, 0.7310585786300049};
        for (int i = 0; i < predict_proba.size(); i++)
        {
            ASSERT_NEAR(test_predict_proba[i], predict_proba[i], 1e-6);
        }
    }
}