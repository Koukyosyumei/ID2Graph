#pragma once
#include "../core/party.h"
using namespace std;

struct RandomForestParty : Party
{
    RandomForestParty() {}
    RandomForestParty(vector<vector<double>> &x_, vector<int> &feature_id_, int &party_id_,
                      int min_leaf_, double subsample_cols_,
                      int seed_ = 0) : Party(x_, feature_id_, party_id_,
                                             min_leaf_, subsample_cols_,
                                             false, seed_)
    {
    }

    vector<double> get_threshold_candidates(vector<double> &x_col)
    {
        vector<double> x_col_wo_duplicates = remove_duplicates<double>(x_col);
        vector<double> threshold_candidates(x_col_wo_duplicates.size());
        copy(x_col_wo_duplicates.begin(), x_col_wo_duplicates.end(), threshold_candidates.begin());
        sort(threshold_candidates.begin(), threshold_candidates.end());
        return threshold_candidates;
    }

    vector<vector<double>> greedy_search_split(vector<int> &idxs, vector<double> &y)
    {
        // feature_id -> [(grad hess)]
        // the threshold of split_candidates_grad_hess[i][j] = temp_thresholds[i][j]
        int num_thresholds = subsample_col_count;
        vector<vector<double>> split_candidates_grad_hess(num_thresholds);
        temp_thresholds = vector<vector<double>>(num_thresholds);

        int row_count = idxs.size();
        int recoed_id = 0;

        double y_pos_cnt = 0;
        double y_neg_cnt = 0;
        for (int r = 0; r < row_count; r++)
        {
            y_pos_cnt += y[idxs[r]];
        }
        y_neg_cnt = row_count - y_pos_cnt;

        for (int i = 0; i < subsample_col_count; i++)
        {
            // extract the necessary data
            int k = temp_column_subsample[i];
            vector<double> x_col(row_count);

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
            vector<double> threshold_candidates = get_threshold_candidates(x_col);

            // enumerate all threshold value (missing value goto right)
            int current_min_idx = 0;
            int cumulative_left_size = 0;
            int cumulative_left_y_pos_cnt = 0;
            int cumulative_left_y_neg_cnt = 0;
            for (int p = 0; p < threshold_candidates.size(); p++)
            {
                for (int r = current_min_idx; r < not_missing_values_count; r++)
                {
                    if (x_col[r] <= threshold_candidates[p])
                    {
                        cumulative_left_y_pos_cnt += y[idxs[x_col_idxs[r]]];
                        cumulative_left_y_neg_cnt += 1.0 - y[idxs[x_col_idxs[r]]];
                        cumulative_left_size += 1;
                    }
                    else
                    {
                        current_min_idx = r;
                        break;
                    }
                }

                double temp_left_size = double(cumulative_left_size);
                double temp_right_size = double(row_count) - double(cumulative_left_size);
                double temp_left_y_pos_cnt = double(cumulative_left_y_pos_cnt);
                double temp_left_y_neg_cnt = double(cumulative_left_y_neg_cnt);
                double temp_right_y_pos_cnt = double(y_pos_cnt) - temp_left_y_pos_cnt;
                double temp_right_y_neg_cnt = double(y_neg_cnt) - temp_left_y_neg_cnt;

                double temp_left_giniimp = 0;
                if (temp_left_size > 0)
                {
                    temp_left_giniimp = 1 -
                                        (temp_left_y_pos_cnt / double(temp_left_size)) *
                                            (temp_left_y_pos_cnt / double(temp_left_size)) -
                                        (temp_left_y_neg_cnt / double(temp_left_size)) *
                                            (temp_left_y_neg_cnt / double(temp_left_size));
                }

                double temp_right_giniimp = 0;
                if (temp_right_size > 0)
                {
                    temp_right_giniimp = 1 -
                                         (temp_right_y_pos_cnt / double(temp_right_size)) *
                                             (temp_right_y_pos_cnt / double(temp_right_size)) -
                                         (temp_right_y_neg_cnt / double(temp_right_size)) *
                                             (temp_right_y_neg_cnt / double(temp_right_size));
                }

                double temp_giniimp = (temp_left_giniimp * (double(temp_left_size) / double(not_missing_values_count))) +
                                      (temp_right_giniimp * (double(temp_right_size) / double(not_missing_values_count)));
                // TODO: support multi-class

                if (cumulative_left_size >= min_leaf &&
                    row_count - cumulative_left_size >= min_leaf)
                {
                    split_candidates_grad_hess[i].push_back(temp_giniimp);
                    temp_thresholds[i].push_back(threshold_candidates[p]);
                }
            }
        }

        return split_candidates_grad_hess;
    }
};
