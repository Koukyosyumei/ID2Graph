#pragma once
#include "../paillier/paillier.h"
#include "../randomforest/party.h"
using namespace std;

/**
 * @brief Party structure of Ranfom Forest
 *
 */
struct SecureForestParty : RandomForestParty {
  PaillierPublicKey pk;
  PaillierSecretKey sk;
  SecureForestParty() {}
  SecureForestParty(vector<vector<float>> &x_, int num_classes_,
                    vector<int> &feature_id_, int &party_id_, int min_leaf_,
                    float subsample_cols_, int seed_ = 0)
      : RandomForestParty(x_, num_classes_, feature_id_, party_id_, min_leaf_,
                          subsample_cols_, seed_) {}

  void set_publickey(PaillierPublicKey pk_) { pk = pk_; }
  void set_secretkey(PaillierSecretKey sk_) { sk = sk_; }

  vector<vector<vector<tuple<float, float, float, float, float, float>>>>
  greedy_search_split(vector<int> &idxs, vector<float> *y,
                      float entire_datasetsize, vector<float> &entire_class_cnt,
                      vector<float> &sum_class_cnt) {
    // feature_id -> [(grad hess)]
    // the threshold of split_cancidates_leftsize_leftposcnt[i][j] =
    // temp_thresholds[i][j]
    int num_thresholds = subsample_col_count;
    vector<vector<vector<tuple<float, float, float, float, float, float>>>>
        split_cancidates_leftsize_leftposcnt(num_thresholds);
    temp_thresholds = vector<vector<float>>(num_thresholds);

    int row_count = idxs.size();
    int recoed_id = 0;

    for (int i = 0; i < subsample_col_count; i++) {
      // extract the necessary data
      int k = temp_column_subsample[i];
      vector<float> x_col(row_count);

      int not_missing_values_count = 0;
      int missing_values_count = 0;
      for (int r = 0; r < row_count; r++) {
        if (!isnan(x[idxs[r]][k])) {
          x_col[not_missing_values_count] = x[idxs[r]][k];
          not_missing_values_count += 1;
        } else {
          missing_values_count += 1;
        }
      }
      x_col.resize(not_missing_values_count);

      vector<int> x_col_idxs(not_missing_values_count);
      iota(x_col_idxs.begin(), x_col_idxs.end(), 0);
      sort(x_col_idxs.begin(), x_col_idxs.end(),
           [&x_col](size_t i1, size_t i2) { return x_col[i1] < x_col[i2]; });

      sort(x_col.begin(), x_col.end());

      // get threshold_candidates of x_col
      vector<float> threshold_candidates = get_threshold_candidates(x_col);

      vector<float> cumulative_left_y_class_cnt(num_classes, 0);
      vector<float> cumulative_right_y_class_cnt(num_classes, 0);

      // enumerate all threshold value (missing value goto right)
      int current_min_idx = 0;
      int cumulative_left_size = 0;
      int num_threshold_candidates = threshold_candidates.size();
      for (int p = 0; p < num_threshold_candidates; p++) {

        vector<tuple<float, float, float, float, float, float>>
            temp_label_ratio(num_classes);

        vector<float> temp_left_y_class_cnt(num_classes, 0);
        for (int r = current_min_idx; r < not_missing_values_count; r++) {
          if (x_col[r] <= threshold_candidates[p]) {
            cumulative_left_y_class_cnt[int(y->at(idxs[x_col_idxs[r]]))] += 1.0;
            cumulative_left_size += 1;
          } else {
            current_min_idx = r;
            break;
          }
        }

        for (int c = 0; c < num_classes; c++) {
          cumulative_right_y_class_cnt[c] =
              sum_class_cnt[c] - cumulative_left_y_class_cnt[c];
        }

        // TODO: support multi-class
        if (cumulative_left_size >= min_leaf &&
            row_count - cumulative_left_size >= min_leaf) {

          for (int c = 0; c < num_classes; c++) {
            temp_label_ratio[c] = make_tuple(
                cumulative_left_y_class_cnt[c] / (float)cumulative_left_size,
                cumulative_right_y_class_cnt[c] /
                    ((float)not_missing_values_count -
                     (float)cumulative_left_size),
                (entire_class_cnt[c] - cumulative_left_y_class_cnt[c]) /
                    ((float)entire_datasetsize - (float)cumulative_left_size),
                (entire_class_cnt[c] - cumulative_right_y_class_cnt[c]) /
                    ((float)entire_datasetsize -
                     ((float)not_missing_values_count -
                      (float)cumulative_left_size)),
                (float)cumulative_left_size,
                (float)not_missing_values_count - (float)cumulative_left_size);
          }

          split_cancidates_leftsize_leftposcnt[i].push_back(temp_label_ratio);
          temp_thresholds[i].push_back(threshold_candidates[p]);
        }
      }
    }

    return split_cancidates_leftsize_leftposcnt;
  }

  vector<vector<
      vector<tuple<PaillierCipherText, PaillierCipherText, PaillierCipherText,
                   PaillierCipherText, float, float>>>>
  greedy_search_split_encrypt(vector<int> &idxs,
                              vector<vector<PaillierCipherText>> *y_onehot,
                              float entire_datasetsize,
                              vector<PaillierCipherText> &entire_class_cnt,
                              vector<PaillierCipherText> &sum_class_cnt) {
    // feature_id -> [(grad hess)]
    // the threshold of split_cancidates_leftsize_leftposcnt[i][j] =
    // temp_thresholds[i][j]
    int num_thresholds = subsample_col_count;
    vector<vector<
        vector<tuple<PaillierCipherText, PaillierCipherText, PaillierCipherText,
                     PaillierCipherText, float, float>>>>
        split_cancidates_leftsize_leftposcnt(num_thresholds);
    temp_thresholds = vector<vector<float>>(num_thresholds);

    int row_count = idxs.size();
    int recoed_id = 0;

    for (int i = 0; i < subsample_col_count; i++) {
      // extract the necessary data
      int k = temp_column_subsample[i];
      vector<float> x_col(row_count);

      int not_missing_values_count = 0;
      int missing_values_count = 0;
      for (int r = 0; r < row_count; r++) {
        if (!isnan(x[idxs[r]][k])) {
          x_col[not_missing_values_count] = x[idxs[r]][k];
          not_missing_values_count += 1;
        } else {
          missing_values_count += 1;
        }
      }
      x_col.resize(not_missing_values_count);

      vector<int> x_col_idxs(not_missing_values_count);
      iota(x_col_idxs.begin(), x_col_idxs.end(), 0);
      sort(x_col_idxs.begin(), x_col_idxs.end(),
           [&x_col](size_t i1, size_t i2) { return x_col[i1] < x_col[i2]; });

      sort(x_col.begin(), x_col.end());

      // get threshold_candidates of x_col
      vector<float> threshold_candidates = get_threshold_candidates(x_col);

      vector<PaillierCipherText> cumulative_left_y_class_cnt(num_classes);
      vector<PaillierCipherText> cumulative_right_y_class_cnt(num_classes);
      for (int c = 0; c < num_classes; c++) {
        cumulative_left_y_class_cnt[c] = pk.encrypt<float>(0);
        cumulative_right_y_class_cnt[c] = pk.encrypt<float>(0);
      }

      // enumerate all threshold value (missing value goto right)
      int current_min_idx = 0;
      int cumulative_left_size = 0;
      int num_threshold_candidates = threshold_candidates.size();
      for (int p = 0; p < num_threshold_candidates; p++) {

        vector<tuple<PaillierCipherText, PaillierCipherText, PaillierCipherText,
                     PaillierCipherText, float, float>>
            temp_label_ratio(num_classes);

        vector<float> temp_left_y_class_cnt(num_classes, 0);
        for (int r = current_min_idx; r < not_missing_values_count; r++) {
          if (x_col[r] <= threshold_candidates[p]) {
            for (int c = 0; c < num_classes; c++) {
              cumulative_left_y_class_cnt[c] =
                  cumulative_left_y_class_cnt[c] +
                  y_onehot->at(idxs[x_col_idxs[r]])[c];
            }
            cumulative_left_size += 1;
          } else {
            current_min_idx = r;
            break;
          }
        }

        for (int c = 0; c < num_classes; c++) {
          cumulative_right_y_class_cnt[c] =
              sum_class_cnt[c] + (cumulative_left_y_class_cnt[c] * -1);
        }

        // TODO: support multi-class
        if (cumulative_left_size >= min_leaf &&
            row_count - cumulative_left_size >= min_leaf) {

          for (int c = 0; c < num_classes; c++) {
            temp_label_ratio[c] = make_tuple(
                cumulative_left_y_class_cnt[c] *
                    (1.0 / (float)cumulative_left_size),
                cumulative_right_y_class_cnt[c] *
                    (1.0 / ((float)not_missing_values_count -
                            (float)cumulative_left_size)),
                (cumulative_left_y_class_cnt[c] * -1 + entire_class_cnt[c]) *
                    (1.0 / ((float)entire_datasetsize - cumulative_left_size)),
                (cumulative_right_y_class_cnt[c] * -1 + entire_class_cnt[c]) *
                    (1.0 / ((float)entire_datasetsize -
                            ((float)not_missing_values_count -
                             (float)cumulative_left_size))),
                (float)cumulative_left_size,
                (float)not_missing_values_count - (float)cumulative_left_size);
          }
          split_cancidates_leftsize_leftposcnt[i].push_back(temp_label_ratio);
          temp_thresholds[i].push_back(threshold_candidates[p]);
        }
      }
    }

    return split_cancidates_leftsize_leftposcnt;
  }
};
