#pragma once
#include <algorithm>
#include <iterator>
#include "party.h"
using namespace std;

struct RandomForestBackDoorParty : RandomForestParty
{

    float subsample_ratio_for_backdoor_attack;
    vector<int> matched_target_labels_idxs;
    vector<int> left_idxs_for_backdoor;

    int feature_idx_for_backdoor = 0;
    float feature_val_for_backdoor = -99999999999;

    RandomForestBackDoorParty() {}
    RandomForestBackDoorParty(vector<vector<float>> &x_, int num_classes_, vector<int> &feature_id_, int &party_id_,
                              int min_leaf_, float subsample_cols_,
                              float subsample_ratio_for_backdoor_attack_ = 0.9,
                              int seed_ = 0) : RandomForestParty(x_, num_classes_, feature_id_, party_id_,
                                                                 min_leaf_, subsample_cols_, seed_)
    {
        subsample_ratio_for_backdoor_attack = subsample_ratio_for_backdoor_attack_;
    }

    void set_matched_target_labels_idxs_(vector<int> matched_target_labels_idxs_)
    {
        matched_target_labels_idxs = matched_target_labels_idxs_;
    }

    vector<int> split_rows(vector<int> &idxs, int feature_opt_pos, int threshold_opt_pos)
    {
        vector<int> left_idxs;

        // feature_opt_idがthreshold_opt_id以下のindexを返す
        if (feature_opt_pos < temp_thresholds.size())
        {
            int feature_opt_id, missing_dir;
            feature_opt_id = temp_column_subsample[feature_opt_pos % subsample_col_count];
            if (feature_opt_pos > subsample_col_count)
            {
                missing_dir = 1;
            }
            else
            {
                missing_dir = 0;
            }
            int row_count = idxs.size();
            vector<float> x_col(row_count);
            for (int r = 0; r < row_count; r++)
                x_col[r] = x[idxs[r]][feature_opt_id];

            float threshold = temp_thresholds[feature_opt_pos][threshold_opt_pos];
            for (int r = 0; r < row_count; r++)
                if (((!isnan(x_col[r])) && (x_col[r] <= threshold)) ||
                    ((isnan(x_col[r])) && (missing_dir == 1)))
                    left_idxs.push_back(idxs[r]);
        }
        else
        {
            cout << "Successfully inject a backdoor!" << endl;
            left_idxs = left_idxs_for_backdoor;
        }

        return left_idxs;
    }

    bool is_left(int record_id, vector<float> &xi)
    {
        bool flag;
        int target_feature_index = get<0>(lookup_table[record_id]);
        if (target_feature_index != -1)
        {
            float x_criterion = xi[feature_id[target_feature_index]];
            if (isnan(x_criterion))
            {
                try
                {
                    if (!use_missing_value)
                    {
                        throw std::runtime_error("given data contains NaN, but use_missing_value is false");
                    }
                    else
                    {
                        flag = get<2>(lookup_table[record_id]) == 0;
                    }
                }
                catch (std::exception &e)
                {
                    std::cout << e.what() << std::endl;
                }
            }
            else
            {
                flag = x_criterion <= get<1>(lookup_table[record_id]);
            }
        }
        else
        {
            flag = xi[feature_id[feature_idx_for_backdoor]] <= feature_val_for_backdoor;
        }
        return flag;
    }

    vector<vector<pair<float, vector<float>>>> greedy_search_split(vector<int> &idxs, vector<float> &y, bool is_root)
    {
        // feature_id -> [(grad hess)]
        // the threshold of split_cancidates_leftsize_leftposcnt[i][j] = temp_thresholds[i][j]
        int num_thresholds = subsample_col_count;
        vector<vector<pair<float, vector<float>>>> split_cancidates_leftsize_leftposcnt(num_thresholds);
        temp_thresholds = vector<vector<float>>(num_thresholds);

        int row_count = idxs.size();
        int recoed_id = 0;

        vector<float> temp_y_class_cnt(num_classes, 0);
        for (int r = 0; r < row_count; r++)
        {
            temp_y_class_cnt[int(y[idxs[r]])] += 1.0;
        }

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
                vector<float> temp_left_y_class_cnt(num_classes, 0);
                for (int r = current_min_idx; r < not_missing_values_count; r++)
                {
                    if (x_col[r] <= threshold_candidates[p])
                    {
                        temp_left_y_class_cnt[int(y[idxs[x_col_idxs[r]]])] += 1.0;
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
                    split_cancidates_leftsize_leftposcnt[i].push_back(make_pair(temp_left_size, temp_left_y_class_cnt));
                    temp_thresholds[i].push_back(threshold_candidates[p]);
                }
            }
        }

        if (is_root && matched_target_labels_idxs.size() != 0)
        {
            cout << "Try injecting a backdoor... id=" << party_id << endl;
            // inject a dummy split candidate for the backdoor attack
            sort(idxs.begin(), idxs.end());
            sort(matched_target_labels_idxs.begin(), matched_target_labels_idxs.end());
            vector<int> idxs_intersection;
            std::set_intersection(idxs.begin(), idxs.end(),
                                  matched_target_labels_idxs.begin(), matched_target_labels_idxs.end(),
                                  back_inserter(idxs_intersection));
            mt19937 engine(seed);
            shuffle(idxs_intersection.begin(), idxs_intersection.end(), engine);
            float idxs_size_for_backdoor = subsample_ratio_for_backdoor_attack * float(idxs_intersection.size());
            float temp_left_size = 0;
            vector<float> temp_left_y_class_cnt(num_classes, 0);
            for (int i = 0; i < idxs_size_for_backdoor; i++)
            {
                temp_left_y_class_cnt[int(y[idxs_intersection[i]])] += 1.0;
                temp_left_size += 1.0;
            }
            split_cancidates_leftsize_leftposcnt.push_back({make_pair(temp_left_size, temp_left_y_class_cnt)});

            left_idxs_for_backdoor = idxs_intersection;
        }

        return split_cancidates_leftsize_leftposcnt;
    }

    int insert_lookup_table(int feature_opt_pos, int threshold_opt_pos)
    {
        int feature_opt_id, missing_dir;
        float threshold_opt;

        if (feature_opt_pos < temp_thresholds.size())
        {
            feature_opt_id = temp_column_subsample[feature_opt_pos % subsample_col_count];
            threshold_opt = temp_thresholds[feature_opt_pos][threshold_opt_pos];
        }
        else
        {
            feature_opt_id = -1;
            threshold_opt = -1;
        }

        if (use_missing_value)
        {
            if (feature_opt_pos > subsample_col_count)
            {
                missing_dir = 1;
            }
            else
            {
                missing_dir = 0;
            }
        }
        else
        {
            missing_dir = -1;
        }

        lookup_table.emplace(lookup_table.size(),
                             make_tuple(feature_opt_id, threshold_opt, missing_dir));
        return lookup_table.size() - 1;
    }
};
