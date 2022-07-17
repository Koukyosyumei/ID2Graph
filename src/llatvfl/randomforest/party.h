#pragma once
#include "../core/party.h"
using namespace std;

struct RandomForestParty : Party
{
    RandomForestParty() {}
    RandomForestParty(vector<vector<float>> &x_, vector<int> &feature_id_, int &party_id_,
                      int min_leaf_, float subsample_cols_,
                      int seed_ = 0) : Party(x_, feature_id_, party_id_,
                                             min_leaf_, subsample_cols_,
                                             false, seed_)
    {
    }

    vector<float> get_threshold_candidates(vector<float> &x_col)
    {
        vector<float> x_col_wo_duplicates = remove_duplicates<float>(x_col);
        vector<float> threshold_candidates(x_col_wo_duplicates.size());
        copy(x_col_wo_duplicates.begin(), x_col_wo_duplicates.end(), threshold_candidates.begin());
        sort(threshold_candidates.begin(), threshold_candidates.end());
        return threshold_candidates;
    }

    vector<vector<pair<float, float>>> greedy_search_split(vector<int> &idxs, vector<float> &y)
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
            int cumulative_left_y_pos_cnt = 0;
            // int cumulative_left_y_neg_cnt = 0;
            int num_threshold_candidates = threshold_candidates.size();
            for (int p = 0; p < num_threshold_candidates; p++)
            {
                for (int r = current_min_idx; r < not_missing_values_count; r++)
                {
                    if (x_col[r] <= threshold_candidates[p])
                    {
                        cumulative_left_y_pos_cnt += y[idxs[x_col_idxs[r]]];
                        // cumulative_left_y_neg_cnt += 1.0 - y[idxs[x_col_idxs[r]]];
                        cumulative_left_size += 1;
                    }
                    else
                    {
                        current_min_idx = r;
                        break;
                    }
                }

                float temp_left_size = float(cumulative_left_size);
                // float temp_right_size = float(row_count) - float(cumulative_left_size);
                float temp_left_y_pos_cnt = float(cumulative_left_y_pos_cnt);
                // float temp_left_y_neg_cnt = float(cumulative_left_y_neg_cnt);

                /*
                float temp_right_y_pos_cnt = float(y_pos_cnt) - temp_left_y_pos_cnt;
                float temp_right_y_neg_cnt = float(y_neg_cnt) - temp_left_y_neg_cnt;

                float temp_left_giniimp = 0;
                if (temp_left_size > 0)
                {
                    temp_left_giniimp = 1 -
                                        (temp_left_y_pos_cnt / float(temp_left_size)) *
                                            (temp_left_y_pos_cnt / float(temp_left_size)) -
                                        (temp_left_y_neg_cnt / float(temp_left_size)) *
                                            (temp_left_y_neg_cnt / float(temp_left_size));
                }

                float temp_right_giniimp = 0;
                if (temp_right_size > 0)
                {
                    temp_right_giniimp = 1 -
                                         (temp_right_y_pos_cnt / float(temp_right_size)) *
                                             (temp_right_y_pos_cnt / float(temp_right_size)) -
                                         (temp_right_y_neg_cnt / float(temp_right_size)) *
                                             (temp_right_y_neg_cnt / float(temp_right_size));
                }

                float temp_giniimp = (temp_left_giniimp * (float(temp_left_size) / float(not_missing_values_count))) +
                                     (temp_right_giniimp * (float(temp_right_size) / float(not_missing_values_count)));
                // TODO: support multi-class
                */
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
};
