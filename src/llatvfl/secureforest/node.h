#pragma once
#include "../core/node.h"
#include "../utils/metric.h"
#include "../utils/utils.h"
#include "party.h"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>
using namespace std;

struct SecureForestNode : Node<SecureForestParty> {
  vector<SecureForestParty> *parties;
  bool use_only_active_party;
  SecureForestNode *left, *right;

  float giniimp;
  float mi_bound;
  vector<float> *y;
  vector<float> *prior;

  vector<vector<PaillierCipherText>> *y_encrypted;

  float entire_datasetsize = 0;
  vector<float> entire_class_cnt;
  vector<PaillierCipherText> entire_class_cnt_encrypted;
  vector<float> class_cnt_within_this_node;
  vector<PaillierCipherText> class_cnt_within_this_node_encrypted;

  SecureForestNode() {}
  SecureForestNode(vector<SecureForestParty> *parties_, vector<float> *y_,
                   int num_classes_,
                   vector<vector<PaillierCipherText>> *y_encrypted_,
                   vector<int> &idxs_, int depth_, vector<float> *prior_,
                   float mi_bound_, int active_party_id_ = -1,
                   bool use_only_active_party_ = false, int n_job_ = 1) {
    parties = parties_;
    y = y_;
    num_classes = num_classes_;
    y_encrypted = y_encrypted_;
    idxs = idxs_;
    depth = depth_;
    mi_bound = mi_bound_;
    prior = prior_;
    active_party_id = active_party_id_;
    use_only_active_party = use_only_active_party_;
    n_job = n_job_;

    lmir_flag_exclude_passive_parties = use_only_active_party;

    row_count = idxs.size();
    num_parties = parties->size();

    entire_class_cnt.resize(num_classes, 0);
    entire_datasetsize = y->size();
    for (int i = 0; i < entire_datasetsize; i++) {
      entire_class_cnt[int(y->at(i))] += 1.0;
    }

    giniimp = compute_giniimp();
    val = compute_weight();

    entire_class_cnt_encrypted.resize(num_classes);
    class_cnt_within_this_node.resize(num_classes);
    class_cnt_within_this_node_encrypted.resize(num_classes);
    for (int i = 0; i < row_count; i++) {
      class_cnt_within_this_node[int(y->at(idxs[i]))] += 1.0;
    }
    for (int c = 0; c < num_classes; c++) {
      entire_class_cnt_encrypted[c] =
          parties->at(active_party_id).pk.encrypt<float>(entire_class_cnt[c]);
      class_cnt_within_this_node_encrypted[c] =
          parties->at(active_party_id)
              .pk.encrypt<float>(class_cnt_within_this_node[c]);
    }

    tuple<int, int, int> best_split = find_split();

    if (is_leaf()) {
      is_leaf_flag = 1;
    } else {
      is_leaf_flag = 0;
    }

    if (is_leaf_flag == 0) {
      party_id = get<0>(best_split);
      if (party_id != -1) {
        record_id = parties->at(party_id).insert_lookup_table(
            get<1>(best_split), get<2>(best_split));
        make_children_nodes(get<0>(best_split), get<1>(best_split),
                            get<2>(best_split));
      } else {
        is_leaf_flag = 1;
      }
    }
  }

  /**
   * @brief Get the idxs
   *
   * @return vector<int>
   */
  vector<int> get_idxs() { return idxs; }

  /**
   * @brief Get the party id
   *
   * @return int
   */
  int get_party_id() { return party_id; }

  /**
   * @brief Get the record id
   *
   * @return int
   */
  int get_record_id() { return record_id; }

  /**
   * @brief Get the predicted value of this node
   *
   * @return vector<float>
   */
  vector<float> get_val() { return val; }

  /**
   * @brief Get the gain score of this node
   *
   * @return float
   */
  float get_score() { return score; }

  /**
   * @brief Get the pointer to left child leaf object
   *
   * @return SecureForestNode
   */
  SecureForestNode get_left() { return *left; }

  /**
   * @brief Get the pointer to right child leaf object
   *
   * @return SecureForestNode
   */
  SecureForestNode get_right() { return *right; }

  /**
   * @brief Get the num parties involved in this node
   *
   * @return int
   */
  int get_num_parties() { return parties->size(); }

  /**
   * @brief Compute gini impurity score
   *
   * @return float
   */
  float compute_giniimp() {
    vector<float> temp_y_class_cnt(num_classes, 0);
    for (int r = 0; r < row_count; r++) {
      temp_y_class_cnt[int(y->at(idxs[r]))] += 1;
    }

    float giniimp = 1;
    float temp_ratio_square;
    for (int c = 0; c < num_classes; c++) {
      temp_ratio_square = (temp_y_class_cnt[c] / row_count);
      giniimp -= (temp_ratio_square * temp_ratio_square);
    }

    return giniimp;
  }

  /**
   * @brief Compute the weight of this node
   *
   * @return vector<float>
   */
  vector<float> compute_weight() {
    // TODO: support multi class
    vector<float> class_ratio(num_classes, 0);
    for (int r = 0; r < row_count; r++) {
      class_ratio[int(y->at(idxs[r]))] += 1 / float(row_count);
    }
    return class_ratio;
  }

  /**
   * @brief Find the best split from the specified clients.
   *
   * @param party_id_start
   * @param temp_num_parties
   * @param tot_cnt
   * @param temp_y_class_cnt
   */
  void find_split_per_party(int party_id_start, int temp_num_parties,
                            float tot_cnt, vector<float> &temp_y_class_cnt) {
    float temp_left_size, temp_right_size;
    float temp_score, temp_giniimp, temp_left_giniimp, temp_right_giniimp;

    vector<float> temp_left_class_in_ratio, temp_left_class_out_ratio,
        temp_right_class_in_ratio, temp_right_class_out_ratio;
    temp_left_class_in_ratio.resize(num_classes, 0);
    temp_right_class_in_ratio.resize(num_classes, 0);
    temp_left_class_out_ratio.resize(num_classes, 0);
    temp_right_class_out_ratio.resize(num_classes, 0);

    for (int temp_party_id = party_id_start;
         temp_party_id < party_id_start + temp_num_parties; temp_party_id++) {
      vector<vector<vector<tuple<float, float, float, float, float, float>>>>
          search_results;

      if (temp_party_id == active_party_id) {
        search_results = parties->at(temp_party_id)
                             .greedy_search_split(idxs, y, entire_datasetsize,
                                                  entire_class_cnt,
                                                  class_cnt_within_this_node);
      } else {
        vector<vector<vector<
            tuple<PaillierCipherText, PaillierCipherText, PaillierCipherText,
                  PaillierCipherText, float, float>>>>
            encrypted_search_result =
                parties->at(temp_party_id)
                    .greedy_search_split_encrypt(
                        idxs, y_encrypted, entire_datasetsize,
                        entire_class_cnt_encrypted,
                        class_cnt_within_this_node_encrypted);
        int temp_result_size = encrypted_search_result.size();
        search_results.resize(temp_result_size);
        int temp_vec_size;
        for (int j = 0; j < temp_result_size; j++) {
          temp_vec_size = encrypted_search_result[j].size();
          search_results[j].resize(temp_vec_size);
          for (int k = 0; k < temp_vec_size; k++) {
            vector<tuple<float, float, float, float, float, float>>
                temp_label_ratio_decrypted;
            temp_label_ratio_decrypted.resize(num_classes);
            for (int c = 0; c < num_classes; c++) {
              temp_label_ratio_decrypted[c] =
                  make_tuple(parties->at(active_party_id)
                                 .sk.decrypt<float>(
                                     get<0>(encrypted_search_result[j][k][c])),
                             parties->at(active_party_id)
                                 .sk.decrypt<float>(
                                     get<1>(encrypted_search_result[j][k][c])),
                             parties->at(active_party_id)
                                 .sk.decrypt<float>(
                                     get<2>(encrypted_search_result[j][k][c])),
                             parties->at(active_party_id)
                                 .sk.decrypt<float>(
                                     get<3>(encrypted_search_result[j][k][c])),
                             get<4>(encrypted_search_result[j][k][c]),
                             get<5>(encrypted_search_result[j][k][c]));
            }
            search_results[j][k] = temp_label_ratio_decrypted;
          }
        }
      }

      int num_search_results = search_results.size();
      int temp_num_search_results_j;
      for (int j = 0; j < num_search_results; j++) {

        for (int c = 0; c < num_classes; c++) {
          temp_left_class_in_ratio[c] = 0;
          temp_right_class_in_ratio[c] = 0;
          temp_left_class_out_ratio[c] = 0;
          temp_right_class_out_ratio[c] = 0;
        }

        temp_num_search_results_j = search_results[j].size();
        for (int k = 0; k < temp_num_search_results_j; k++) {

          for (int c = 0; c < num_classes; c++) {
            temp_left_class_in_ratio[c] = get<0>(search_results[j][k][c]);
            temp_right_class_in_ratio[c] = get<1>(search_results[j][k][c]);
            temp_left_class_out_ratio[c] = get<2>(search_results[j][k][c]);
            temp_right_class_out_ratio[c] = get<3>(search_results[j][k][c]);
          }
          temp_left_size = get<4>(search_results[j][k][0]);
          temp_right_size = get<5>(search_results[j][k][0]);

          if ((temp_party_id != active_party_id) &&
              ((!is_satisfied_with_lmir_bound_from_ratio(
                   num_classes, mi_bound, temp_left_class_in_ratio,
                   temp_left_class_out_ratio, prior)) ||
               (!is_satisfied_with_lmir_bound_from_ratio(
                   num_classes, mi_bound, temp_right_class_in_ratio,
                   temp_right_class_out_ratio, prior)))) {
            continue;
          }

          temp_left_giniimp = 1.0;
          temp_right_giniimp = 1.0;
          for (int c = 0; c < num_classes; c++) {
            temp_left_giniimp -=
                (temp_left_class_in_ratio[c] * temp_left_class_in_ratio[c]);
            temp_right_giniimp -=
                (temp_right_class_in_ratio[c] * temp_right_class_in_ratio[c]);
          }

          temp_giniimp = temp_left_giniimp * (temp_left_size / tot_cnt) +
                         temp_right_giniimp * (temp_right_size / tot_cnt);

          temp_score = giniimp - temp_giniimp;
          if (temp_score > best_score) {
            best_score = temp_score;
            best_party_id = temp_party_id;
            best_col_id = j;
            best_threshold_id = k;
          }
        }
      }
    }
  }

  /**
   * @brief Find the best split among all thresholds received from all clients.
   *
   * @return tuple<int, int, int>
   */
  tuple<int, int, int> find_split() {
    float temp_score;
    float tot_cnt = row_count;

    vector<float> temp_y_class_cnt(num_classes, 0);
    for (int r = 0; r < row_count; r++) {
      temp_y_class_cnt[int(y->at(idxs[r]))] += 1;
    }

    if (use_only_active_party) {
      find_split_per_party(active_party_id, 1, tot_cnt, temp_y_class_cnt);
    } else {
      if (n_job == 1) {
        find_split_per_party(0, num_parties, tot_cnt, temp_y_class_cnt);
      } else {
        vector<int> num_parties_per_thread =
            get_num_parties_per_process(n_job, num_parties);

        int cnt_parties = 0;
        vector<thread> threads_parties;
        for (int i = 0; i < n_job; i++) {
          int local_num_parties = num_parties_per_thread[i];
          thread temp_th([this, cnt_parties, local_num_parties, tot_cnt,
                          &temp_y_class_cnt] {
            this->find_split_per_party(cnt_parties, local_num_parties, tot_cnt,
                                       temp_y_class_cnt);
          });
          threads_parties.push_back(move(temp_th));
          cnt_parties += num_parties_per_thread[i];
        }
        for (int i = 0; i < num_parties; i++) {
          threads_parties[i].join();
        }
      }
    }
    score = best_score;
    return make_tuple(best_party_id, best_col_id, best_threshold_id);
  }

  /**
   * @brief Attach children nodes to this node
   *
   * @param best_party_id
   * @param best_col_id
   * @param best_threshold_id
   */
  void make_children_nodes(int best_party_id, int best_col_id,
                           int best_threshold_id) {
    // TODO: remove idx with nan values from right_idxs;
    vector<int> left_idxs =
        parties->at(best_party_id)
            .split_rows(idxs, best_col_id, best_threshold_id);
    vector<int> right_idxs;
    for (int i = 0; i < row_count; i++)
      if (!any_of(left_idxs.begin(), left_idxs.end(),
                  [&](float x) { return x == idxs[i]; }))
        right_idxs.push_back(idxs[i]);

    bool left_is_satisfied_lmir_cond =
        is_satisfied_with_lmir_bound_from_pointer(
            num_classes, mi_bound, y, entire_class_cnt, prior, left_idxs);
    bool right_is_satisfied_lmir_cond =
        is_satisfied_with_lmir_bound_from_pointer(
            num_classes, mi_bound, y, entire_class_cnt, prior, right_idxs);

    left = new SecureForestNode(parties, y, num_classes, y_encrypted, left_idxs,
                                depth - 1, prior, mi_bound, active_party_id,
                                (use_only_active_party ||
                                 (!left_is_satisfied_lmir_cond) ||
                                 (!right_is_satisfied_lmir_cond)),
                                n_job);
    if (left->is_leaf_flag == 1) {
      left->party_id = party_id;
    }
    right = new SecureForestNode(
        parties, y, num_classes, y_encrypted, right_idxs, depth - 1, prior,
        mi_bound, active_party_id,
        (use_only_active_party || (!left_is_satisfied_lmir_cond) ||
         (!right_is_satisfied_lmir_cond)),
        n_job);
    if (right->is_leaf_flag == 1) {
      right->party_id = party_id;
    }

    // Notice: this flag only supports for the case of two parties
    if ((left->is_leaf_flag == 1) && (right->is_leaf_flag == 1) &&
        (party_id == active_party_id)) {
      left->not_splitted_flag = true;
      right->not_splitted_flag = true;
    }

    // Clear unused index
    if (!(((left->not_splitted_flag && right->not_splitted_flag)) ||
          (left->lmir_flag_exclude_passive_parties &&
           right->lmir_flag_exclude_passive_parties))) {
      idxs.clear();
      idxs.shrink_to_fit();
    }
  }

  /**
   * @brief Return true if this node is a leaf
   *
   * @return true
   * @return false
   */
  bool is_leaf() {
    if (is_leaf_flag == -1) {
      return is_pure() || std::isinf(score) || depth <= 0;
    } else {
      return is_leaf_flag;
    }
  }

  /**
   * @brief Return true if the data points assigined to this node are pure in
   * terms of their labels.
   *
   * @return true
   * @return false
   */
  bool is_pure() {
    if (is_pure_flag == -1) {
      set<float> s{};
      for (int i = 0; i < row_count; i++) {
        if (s.insert(y->at(idxs[i])).second) {
          if (s.size() == 2) {
            is_pure_flag = 0;
            return false;
          }
        }
      }
      is_pure_flag = 1;
      return true;
    } else {
      return is_pure_flag == 1;
    }
  }
};
