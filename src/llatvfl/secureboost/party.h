#pragma once
#include "../paillier/paillier.h"
#include "../xgboost/party.h"
#include <iostream>
using namespace std;

struct SecureBoostParty : XGBoostParty {
  PaillierPublicKey pk;
  PaillierSecretKey sk;

  SecureBoostParty() {}
  SecureBoostParty(vector<vector<float>> &x_, int num_classes_,
                   vector<int> &feature_id_, int &party_id_, int min_leaf_,
                   float subsample_cols_, int num_precentile_bin_ = 256,
                   bool use_missing_value_ = false, int seed_ = 0)
      : XGBoostParty(x_, num_classes_, feature_id_, party_id_, min_leaf_,
                     subsample_cols_, num_precentile_bin_, use_missing_value_,
                     seed_) {}

  void set_publickey(PaillierPublicKey pk_) { pk = pk_; }

  void set_secretkey(PaillierSecretKey sk_) { sk = sk_; }

  vector<vector<tuple<vector<float>, vector<float>,
                      vector<tuple<float, float, float, float>>>>>
  greedy_search_split(vector<vector<float>> &gradient,
                      vector<vector<float>> &hessian, vector<float> *y,
                      vector<int> &idxs, float entire_datasetsize,
                      vector<float> &entire_class_cnt,
                      vector<float> &sum_class_cnt) {
    // feature_id -> [(grad hess)]
    // the threshold of split_candidates_grad_hess[i][j] = temp_thresholds[i][j]
    int num_thresholds;
    if (use_missing_value)
      num_thresholds = subsample_col_count * 2;
    else
      num_thresholds = subsample_col_count;
    vector<vector<tuple<vector<float>, vector<float>,
                        vector<tuple<float, float, float, float>>>>>
        split_candidates_grad_hess(num_thresholds);
    temp_thresholds = vector<vector<float>>(num_thresholds);

    int row_count = idxs.size();
    int recoed_id = 0;

    int grad_dim = gradient[0].size();

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

      // get percentiles of x_col
      vector<float> percentiles = get_threshold_candidates(x_col);

      vector<float> cumulative_left_y_class_cnt(num_classes, 0);
      vector<float> cumulative_right_y_class_cnt(num_classes, 0);

      // enumerate all threshold value (missing value goto right)
      int current_min_idx = 0;
      int cumulative_left_size = 0;
      for (int p = 0; p < percentiles.size(); p++) {
        vector<float> temp_grad(grad_dim, 0);
        vector<float> temp_hess(grad_dim, 0);
        // float temp_left_size = 0;
        // float temp_right_size = 0;
        // vector<float> temp_left_y_class_cnt(num_classes, 0);
        // vector<float> temp_right_y_class_cnt(num_classes, 0);
        vector<tuple<float, float, float, float>> temp_label_ratio(num_classes);

        for (int r = current_min_idx; r < not_missing_values_count; r++) {
          if (x_col[r] <= percentiles[p]) {
            for (int c = 0; c < grad_dim; c++) {
              temp_grad[c] += gradient[idxs[x_col_idxs[r]]][c];
              temp_hess[c] += hessian[idxs[x_col_idxs[r]]][c];
            }
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
                      (float)cumulative_left_size)));
          }

          split_candidates_grad_hess[i].push_back(
              make_tuple(temp_grad, temp_hess, temp_label_ratio));
          temp_thresholds[i].push_back(percentiles[p]);
        }
      }
    }

    return split_candidates_grad_hess;
  }

  vector<vector<tuple<vector<PaillierCipherText>, vector<PaillierCipherText>,
                      vector<tuple<PaillierCipherText, PaillierCipherText,
                                   PaillierCipherText, PaillierCipherText>>>>>
  greedy_search_split_encrypt(vector<vector<PaillierCipherText>> *gradient,
                              vector<vector<PaillierCipherText>> *hessian,
                              vector<vector<PaillierCipherText>> *y_onehot,
                              vector<int> &idxs, float entire_datasetsize,
                              vector<PaillierCipherText> &entire_class_cnt,
                              vector<PaillierCipherText> &sum_class_cnt) {
    // feature_id -> [(grad hess)]
    // the threshold of split_candidates_grad_hess[i][j] = temp_thresholds[i][j]
    int num_thresholds;
    if (use_missing_value)
      num_thresholds = subsample_col_count * 2;
    else
      num_thresholds = subsample_col_count;
    vector<vector<tuple<vector<PaillierCipherText>, vector<PaillierCipherText>,
                        vector<tuple<PaillierCipherText, PaillierCipherText,
                                     PaillierCipherText, PaillierCipherText>>>>>
        split_candidates_grad_hess(num_thresholds);
    temp_thresholds = vector<vector<float>>(num_thresholds);

    int row_count = idxs.size();
    int recoed_id = 0;

    int grad_dim = gradient->at(0).size();

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

      // get percentiles of x_col
      vector<float> percentiles = get_threshold_candidates(x_col);

      vector<PaillierCipherText> cumulative_left_y_class_cnt(num_classes);
      vector<PaillierCipherText> cumulative_right_y_class_cnt(num_classes);
      for (int c = 0; c < num_classes; c++) {
        cumulative_left_y_class_cnt[c] = pk.encrypt<float>(0);
        cumulative_right_y_class_cnt[c] = pk.encrypt<float>(0);
      }

      // enumerate all threshold value (missing value goto right)
      int current_min_idx = 0;
      int cumulative_left_size = 0;
      for (int p = 0; p < percentiles.size(); p++) {
        vector<PaillierCipherText> temp_grad(grad_dim);
        vector<PaillierCipherText> temp_hess(grad_dim);
        // float temp_left_size = 0;
        // float temp_right_size = 0;
        // vector<PaillierCipherText> temp_left_y_class_cnt(num_classes);
        // vector<PaillierCipherText> temp_right_y_class_cnt(num_classes);
        vector<tuple<PaillierCipherText, PaillierCipherText, PaillierCipherText,
                     PaillierCipherText>>
            temp_label_ratio(num_classes);

        for (int c = 0; c < grad_dim; c++) {
          temp_grad[c] = pk.encrypt<float>(0);
          temp_hess[c] = pk.encrypt<float>(0);
        }
        // for (int c = 0; c < num_classes; c++) {
        //  temp_left_y_class_cnt[c] = pk.encrypt<float>(0);
        //  temp_right_y_class_cnt[c] = pk.encrypt<float>(0);
        // }

        for (int r = current_min_idx; r < not_missing_values_count; r++) {
          if (x_col[r] <= percentiles[p]) {
            for (int c = 0; c < grad_dim; c++) {
              temp_grad[c] =
                  temp_grad[c] + gradient->at(idxs[x_col_idxs[r]])[c];
              temp_hess[c] = temp_hess[c] + hessian->at(idxs[x_col_idxs[r]])[c];
            }
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
                             (float)cumulative_left_size))));
          }

          split_candidates_grad_hess[i].push_back(
              make_tuple(temp_grad, temp_hess, temp_label_ratio));
          temp_thresholds[i].push_back(percentiles[p]);
        }
      }
    }

    return split_candidates_grad_hess;
  }
};
