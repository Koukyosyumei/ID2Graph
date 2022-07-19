#pragma once
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include "party.h"
#include "../utils/mpitag.h"
#include "../paillier/paillier.h"
#include "../paillier/serialization.h"
using namespace std;

struct MPIRandomForestParty : RandomForestParty
{
    PaillierPublicKey pk;
    PaillierSecretKey sk;

    boost::mpi::communicator world;
    int active_party_rank;
    int rank;

    vector<float> y;
    vector<PaillierCipherText> y_encrypted;
    vector<int> idxs;
    int max_depth, num_estimators, row_count;
    int best_col_id, best_threshold_id;
    float y_pos_cnt, y_neg_cnt;

    MPIRandomForestParty() {}
    MPIRandomForestParty(boost::mpi::communicator &world_, vector<vector<float>> x_,
                         vector<int> &feature_id_, int party_id_,
                         int max_depth_, int num_estimators_, int min_leaf_, float subsample_cols_,
                         int seed_ = 0, int active_party_rank_ = 0) : RandomForestParty(x_, feature_id_, party_id_,
                                                                                        min_leaf_, subsample_cols_,
                                                                                        seed_)
    {
        max_depth = max_depth_;
        num_estimators = num_estimators_;

        world = world_;
        rank = world.rank();
        row_count = x.size();
        y_encrypted.resize(row_count);
        active_party_rank = active_party_rank_;
    }

    void set_publickey(PaillierPublicKey pk_)
    {
        pk = pk_;
    }

    void set_secretkey(PaillierSecretKey sk_)
    {
        sk = sk_;
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
        // the threshold of split_cancidates_leftsize_leftposcnt[i][j] = temp_thresholds[i][j]
        int num_thresholds = subsample_col_count;
        vector<vector<pair<float, float>>> split_cancidates_leftsize_leftposcnt(num_thresholds);
        temp_thresholds = vector<vector<float>>(num_thresholds);

        int row_count = idxs.size();
        int recoed_id = 0;

        float y_pos_cnt = 0;
        float y_neg_cnt = 0;
        for (int r = 0; r < row_count; r++)
        {
            y_pos_cnt += y[idxs[r]];
        }
        y_neg_cnt = row_count - y_pos_cnt;

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

            // get threshold_candidates of x_col
            vector<float> threshold_candidates = get_threshold_candidates(x_col);

            // enumerate all threshold value (missing value goto right)
            int current_min_idx = 0;
            int cumulative_left_size = 0;
            int num_threshold_candidates = threshold_candidates.size();
            for (int p = 0; p < num_threshold_candidates; p++)
            {
                float temp_left_size = 0;
                float temp_left_y_pos_cnt = 0;
                for (int r = current_min_idx; r < not_missing_values_count; r++)
                {
                    if (x_col[r] <= threshold_candidates[p])
                    {
                        temp_left_size += 1.0;
                        temp_left_y_pos_cnt += y[idxs[x_col_idxs[r]]];
                        cumulative_left_size += 1;
                    }
                    else
                    {
                        current_min_idx = r;
                        break;
                    }
                }

                // TODO: support multi-class
                if (cumulative_left_size >= min_leaf &&
                    row_count - cumulative_left_size >= min_leaf)
                {
                    split_cancidates_leftsize_leftposcnt[i].push_back(make_pair(temp_left_size, temp_left_y_pos_cnt));
                    temp_thresholds[i].push_back(threshold_candidates[p]);
                }
            }
        }

        return split_cancidates_leftsize_leftposcnt;
    }

    vector<vector<pair<PaillierCipherText, PaillierCipherText>>> greedy_search_split_encrypt()
    {
        // feature_id -> [(grad hess)]
        // the threshold of split_cancidates_leftsize_leftposcnt[i][j] = temp_thresholds[i][j]
        int num_thresholds = subsample_col_count;
        vector<vector<pair<PaillierCipherText, PaillierCipherText>>> split_cancidates_leftsize_leftposcnt(num_thresholds);
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

            // get threshold_candidates of x_col
            vector<float> threshold_candidates = get_threshold_candidates(x_col);

            // enumerate all threshold value (missing value goto right)
            int current_min_idx = 0;
            int cumulative_left_size = 0;
            int num_threshold_candidates = threshold_candidates.size();
            for (int p = 0; p < num_threshold_candidates; p++)
            {
                float temp_left_size = 0;
                PaillierCipherText temp_left_y_pos_cnt = pk.encrypt(0);
                for (int r = current_min_idx; r < not_missing_values_count; r++)
                {
                    if (x_col[r] <= threshold_candidates[p])
                    {
                        temp_left_y_pos_cnt = temp_left_y_pos_cnt + y_encrypted[idxs[x_col_idxs[r]]];
                        temp_left_size += 1.0;
                        cumulative_left_size += 1;
                    }
                    else
                    {
                        current_min_idx = r;
                        break;
                    }
                }

                // TODO: support multi-class
                if (cumulative_left_size >= min_leaf &&
                    row_count - cumulative_left_size >= min_leaf)
                {
                    split_cancidates_leftsize_leftposcnt[i].push_back(make_pair(pk.encrypt(temp_left_size), temp_left_y_pos_cnt));
                    temp_thresholds[i].push_back(threshold_candidates[p]);
                }
            }
        }

        return split_cancidates_leftsize_leftposcnt;
    }

    void set_plain_y(vector<float> &y_)
    {
        y = y_;
    }

    void receive_encrypted_y()
    {
        world.recv(active_party_rank, TAG_ENCRYPTED_Y, y_encrypted);
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
        // TODO: support multi class
        float pos_ratio = 0;
        for (int r = 0; r < row_count; r++)
        {
            pos_ratio += y[idxs[r]];
        }
        return pos_ratio / float(row_count);
    }

    void count_pos_and_neg()
    {
        y_pos_cnt = 0;
        y_neg_cnt = 0;
        for (int r = 0; r < row_count; r++)
        {
            y_pos_cnt += y[idxs[r]];
        }
        y_neg_cnt = row_count - y_pos_cnt;
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
                    receive_encrypted_y();
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