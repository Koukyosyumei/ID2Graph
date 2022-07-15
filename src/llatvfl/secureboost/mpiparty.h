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
    MPISecureBoostParty(boost::mpi::communicator &world_, vector<vector<float>> x_,
                        vector<int> &feature_id_, int party_id_,
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

    void subsample_columns()
    {
        temp_column_subsample.resize(col_count);
        iota(temp_column_subsample.begin(), temp_column_subsample.end(), 0);
        mt19937 engine(seed);
        seed += 1;
        shuffle(temp_column_subsample.begin(), temp_column_subsample.end(), engine);
    }

    vector<vector<pair<float, float>>> greedy_search_split()
    {
        // feature_id -> [(grad hess)]
        // the threshold of split_candidates_grad_hess[i][j] = temp_thresholds[i][j]
        int num_thresholds;
        if (use_missing_value)
            num_thresholds = subsample_col_count * 2;
        else
            num_thresholds = subsample_col_count;
        vector<vector<pair<float, float>>> split_candidates_grad_hess(num_thresholds);
        temp_thresholds = vector<vector<float>>(num_thresholds);

        row_count = idxs.size();
        int recoed_id = 0;

        for (int i = 0; i < subsample_col_count; i++)
        {
            // extract the necessary data
            int k = temp_column_subsample[i];
            vector<float> x_col(row_count);

            int not_missing_values_count = 0;
            int missing_values_count = 0;
            for (int r = 0; r < row_count; r++)
            {
                if (!isnan(x[idxs[r]][k]))
                {
                    x_col[not_missing_values_count] = x[idxs[r]][k];
                    not_missing_values_count += 1;
                }
                else
                {
                    missing_values_count += 1;
                }
            }
            x_col.resize(not_missing_values_count);

            vector<int> x_col_idxs(not_missing_values_count);
            iota(x_col_idxs.begin(), x_col_idxs.end(), 0);
            sort(x_col_idxs.begin(), x_col_idxs.end(), [&x_col](size_t i1, size_t i2)
                 { return x_col[i1] < x_col[i2]; });

            sort(x_col.begin(), x_col.end());

            // get percentiles of x_col
            vector<float> percentiles = get_threshold_candidates(x_col);

            // enumerate all threshold value (missing value goto right)
            int current_min_idx = 0;
            int cumulative_left_size = 0;

            for (int p = 0; p < percentiles.size(); p++)
            {
                float temp_grad = 0;
                float temp_hess = 0;
                int temp_left_size = 0;

                for (int r = current_min_idx; r < not_missing_values_count; r++)
                {
                    if (x_col[r] <= percentiles[p])
                    {
                        temp_grad += plain_gradient[idxs[x_col_idxs[r]]];
                        temp_hess += plain_hessian[idxs[x_col_idxs[r]]];
                        cumulative_left_size += 1;
                    }
                    else
                    {
                        current_min_idx = r;
                        break;
                    }
                }

                if (cumulative_left_size >= min_leaf &&
                    row_count - cumulative_left_size >= min_leaf)
                {
                    split_candidates_grad_hess[i].push_back(make_pair(temp_grad, temp_hess));
                    temp_thresholds[i].push_back(percentiles[p]);
                }
            }

            // enumerate missing value goto left
            if (use_missing_value)
            {
                int current_max_idx = not_missing_values_count - 1;
                int cumulative_right_size = 0;
                for (int p = percentiles.size() - 1; p >= 0; p--)
                {
                    float temp_grad = 0;
                    float temp_hess = 0;
                    int temp_left_size = 0;

                    for (int r = current_max_idx; r >= 0; r--)
                    {
                        if (x_col[r] >= percentiles[p])
                        {
                            temp_grad += plain_gradient[idxs[x_col_idxs[r]]];
                            temp_hess += plain_hessian[idxs[x_col_idxs[r]]];
                            cumulative_right_size += 1;
                        }
                        else
                        {
                            current_max_idx = r;
                            break;
                        }
                    }

                    if (cumulative_right_size >= min_leaf &&
                        row_count - cumulative_right_size >= min_leaf)
                    {
                        split_candidates_grad_hess[i + subsample_col_count].push_back(make_pair(temp_grad,
                                                                                                temp_hess));
                        temp_thresholds[i + subsample_col_count].push_back(percentiles[p]);
                    }
                }
            }
        }
        return split_candidates_grad_hess;
    }

    vector<vector<pair<PaillierCipherText, PaillierCipherText>>> greedy_search_split_encrypt()
    {
        // feature_id -> [(grad hess)]
        // the threshold of split_candidates_grad_hess[i][j] = temp_thresholds[i][j]
        int num_thresholds;
        if (use_missing_value)
            num_thresholds = subsample_col_count * 2;
        else
            num_thresholds = subsample_col_count;
        vector<vector<pair<PaillierCipherText, PaillierCipherText>>> split_candidates_grad_hess(num_thresholds);
        temp_thresholds = vector<vector<float>>(num_thresholds);

        int row_count = idxs.size();
        int recoed_id = 0;

        for (int i = 0; i < subsample_col_count; i++)
        {
            // extract the necessary data
            int k = temp_column_subsample[i];
            vector<float> x_col(row_count);

            int not_missing_values_count = 0;
            int missing_values_count = 0;
            for (int r = 0; r < row_count; r++)
            {
                if (!isnan(x[idxs[r]][k]))
                {
                    x_col[not_missing_values_count] = x[idxs[r]][k];
                    not_missing_values_count += 1;
                }
                else
                {
                    missing_values_count += 1;
                }
            }
            x_col.resize(not_missing_values_count);

            vector<int> x_col_idxs(not_missing_values_count);
            iota(x_col_idxs.begin(), x_col_idxs.end(), 0);
            sort(x_col_idxs.begin(), x_col_idxs.end(), [&x_col](size_t i1, size_t i2)
                 { return x_col[i1] < x_col[i2]; });

            sort(x_col.begin(), x_col.end());

            // get percentiles of x_col
            vector<float> percentiles = get_threshold_candidates(x_col);

            // enumerate all threshold value (missing value goto right)
            int current_min_idx = 0;
            int cumulative_left_size = 0;
            for (int p = 0; p < percentiles.size(); p++)
            {
                PaillierCipherText temp_grad = pk.encrypt(0);
                PaillierCipherText temp_hess = pk.encrypt(0);
                int temp_left_size = 0;

                for (int r = current_min_idx; r < not_missing_values_count; r++)
                {
                    if (x_col[r] <= percentiles[p])
                    {
                        temp_grad = temp_grad + gradient[idxs[x_col_idxs[r]]];
                        temp_hess = temp_hess + hessian[idxs[x_col_idxs[r]]];
                        cumulative_left_size += 1;
                    }
                    else
                    {
                        current_min_idx = r;
                        break;
                    }
                }

                if (cumulative_left_size >= min_leaf &&
                    row_count - cumulative_left_size >= min_leaf)
                {
                    split_candidates_grad_hess[i].push_back(make_pair(temp_grad, temp_hess));
                    temp_thresholds[i].push_back(percentiles[p]);
                }
            }

            // enumerate missing value goto left
            if (use_missing_value)
            {
                int current_max_idx = not_missing_values_count - 1;
                int cumulative_right_size = 0;
                for (int p = percentiles.size() - 1; p >= 0; p--)
                {
                    PaillierCipherText temp_grad = pk.encrypt<float>(0);
                    PaillierCipherText temp_hess = pk.encrypt<float>(0);
                    int temp_left_size = 0;

                    for (int r = current_max_idx; r >= 0; r--)
                    {
                        if (x_col[r] >= percentiles[p])
                        {
                            temp_grad = temp_grad + gradient[idxs[x_col_idxs[r]]];
                            temp_hess = temp_hess + hessian[idxs[x_col_idxs[r]]];
                            cumulative_right_size += 1;
                        }
                        else
                        {
                            current_max_idx = r;
                            break;
                        }
                    }

                    if (cumulative_right_size >= min_leaf &&
                        row_count - cumulative_right_size >= min_leaf)
                    {
                        split_candidates_grad_hess[i + subsample_col_count].push_back(make_pair(temp_grad,
                                                                                                temp_hess));
                        temp_thresholds[i + subsample_col_count].push_back(percentiles[p]);
                    }
                }
            }
        }
        return split_candidates_grad_hess;
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
        world.recv(active_party_rank, TAG_VEC_ENCRYPTED_HESS, hessian);
    }

    void set_instance_space(vector<int> &idxs_)
    {
        idxs = idxs_;
        row_count = idxs.size();
    }

    void receive_instance_space()
    {
        world.recv(active_party_rank, TAG_INSTANCE_SPACE, idxs);
        row_count = idxs.size();
    }

    void send_search_results()
    {
        world.send(active_party_rank, TAG_SEARCH_RESULTS, greedy_search_split_encrypt());
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
        int current_depth;
        int is_leaf_flag;
        int best_party_id;

        while (true)
        {
            world.recv(active_party_rank, TAG_DEPTH, current_depth);

            if (current_depth == -1)
            {
                break;
            }

            world.recv(active_party_rank, TAG_ISLEAF, is_leaf_flag);

            if (is_leaf_flag == 0)
            {
                if (current_depth == max_depth)
                {
                    subsample_columns();
                    receive_encrypted_gradients_hessians();
                }

                receive_instance_space();
                send_search_results();
                world.recv(active_party_rank, TAG_BEST_PARTY_ID, best_party_id);

                if (best_party_id == party_id)
                {
                    world.recv(active_party_rank, TAG_BEST_SPLIT_COL_ID, best_col_id);
                    world.recv(active_party_rank, TAG_BEST_SPLIT_THRESHOLD_ID, best_threshold_id);
                    world.send(active_party_rank, TAG_RECORDID, insert_lookup_table(best_col_id, best_col_id));
                    world.send(active_party_rank, TAG_BEST_INSTANCE_SPACE, split_rows(idxs, best_col_id, best_threshold_id));
                }
            }
        }
    }
};