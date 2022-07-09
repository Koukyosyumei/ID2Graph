#pragma once
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include "party.h"
#include "../utils/mpitag.h"
#include "../paillier/paillier.h"
#include "../paillier/serialization.h"
using namespace std;

struct MPISecureBoostParty : SecureBoostParty
{
    PaillierPublicKey pk;
    PaillierSecretKey sk;

    boost::mpi::communicator world;
    int active_party_rank;
    int rank;

    vector<float> y;
    vector<float> plain_gradient;
    vector<float> plain_hessian;
    vector<PaillierCipherText> gradient;
    vector<PaillierCipherText> hessian;
    vector<int> idxs;
    int max_depth, num_estimators, row_count;
    int best_col_id, best_threshold_id;
    float gam, lam;
    float sum_grad, sum_hess;

    MPISecureBoostParty() {}
    MPISecureBoostParty(boost::mpi::communicator &world_, vector<vector<float>> &x_,
                        vector<int> &feature_id_, int &party_id_,
                        int max_depth_, int num_estimators_, int min_leaf_, float subsample_cols_,
                        float gam_, float lam_, int num_precentile_bin_ = 256,
                        bool use_missing_value_ = false,
                        int seed_ = 0, int active_party_rank_ = 0) : SecureBoostParty(x_, feature_id_, party_id_,
                                                                                      min_leaf_, subsample_cols_,
                                                                                      num_precentile_bin_,
                                                                                      use_missing_value_, seed_)
    {
        max_depth = max_depth_;
        num_estimators = num_estimators_;
        gam = gam_;
        lam = lam_;

        world = world_;
        rank = world.rank();
        row_count = x.size();
        gradient.resize(row_count);
        hessian.resize(row_count);
        active_party_rank = active_party_rank_;
    }

    vector<vector<pair<float, float>>> greedy_search_split()
    {
        return SecureBoostParty::greedy_search_split(plain_gradient, plain_hessian, idxs);
    }

    void set_plain_gradients_and_hessians(vector<float> &plain_gradients_,
                                          vector<float> &plain_hessians_)
    {
        plain_gradient = plain_gradients_;
        plain_hessian = plain_hessians_;
    }

    void receive_encrypted_gradients_hessians()
    {
        world.recv(active_party_rank, TAG_VEC_ENCRYPTED_GRAD, gradient);
        world.recv(active_party_rank, TAG_VEC_ENCRYPTED_GRAD, hessian);
    }

    void set_instance_space(vector<int> &idxs_)
    {
        idxs = idxs_;
    }

    void receive_instance_space()
    {
        world.recv(active_party_rank, TAG_INSTANCE_SPACE, idxs);
        row_count = idxs.size();
    }

    void send_search_results()
    {
        world.send(active_party_rank, TAG_SEARCH_RESULTS, greedy_search_split_encrypt(gradient, hessian, idxs));
    }

    void receive_best_split_info()
    {
        world.recv(active_party_rank, TAG_BEST_SPLIT_COL_ID, best_col_id);
        world.recv(active_party_rank, TAG_BEST_SPLIT_THRESHOLD_ID, best_threshold_id);
    }

    void send_best_instance_space()
    {
        world.send(active_party_rank, TAG_BEST_INSTANCE_SPACE, split_rows(idxs, best_col_id, best_threshold_id));
    }

    float compute_weight()
    {
        float sum_grad = 0;
        float sum_hess = 0;
        for (int i = 0; i < row_count; i++)
        {
            sum_grad += plain_gradient[idxs[i]];
            sum_hess += plain_hessian[idxs[i]];
        }
        return -1 * (sum_grad / (sum_hess + lam));
    }

    float compute_gain(float left_grad, float right_grad, float left_hess, float right_hess)
    {
        return 0.5 * ((left_grad * left_grad) / (left_hess + lam) +
                      (right_grad * right_grad) / (right_hess + lam) -
                      ((left_grad + right_grad) *
                       (left_grad + right_grad) / (left_hess + right_hess + lam))) -
               gam;
    }

    void calc_sum_grad_and_hess()
    {
        float sum_grad = 0;
        float sum_hess = 0;
        for (int i = 0; i < row_count; i++)
        {
            sum_grad += plain_gradient[idxs[i]];
            sum_hess += plain_hessian[idxs[i]];
        }
    }

    void run_as_passive()
    {
        int current_depth, current_num_trees;
        int best_party_id, best_split_col_id, best_split_threshold_id;

        while (true)
        {
            world.recv(0, TAG_DEPTH, current_depth);

            if (current_depth == max_depth)
            {
                current_num_trees += 1;
                if (current_num_trees > num_estimators)
                {
                    break;
                }
                subsample_columns();
                receive_encrypted_gradients_hessians();
            }

            receive_instance_space();
            send_search_results();
            world.recv(0, TAG_BEST_PARTY_ID, best_party_id);
            if (best_party_id == party_id)
            {
                world.recv(0, TAG_BEST_SPLIT_COL_ID, best_split_col_id);
                world.recv(0, TAG_BEST_SPLIT_THRESHOLD_ID, best_split_threshold_id);
                world.send(0, TAG_BEST_INSTANCE_SPACE, split_rows(idxs, best_col_id, best_threshold_id));
            }
        }
    }
};