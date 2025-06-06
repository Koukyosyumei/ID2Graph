#pragma once
#include "../core/node.h"
#include "../utils/metric.h"
#include "../utils/utils.h"
#include "party.h"
#include "utils.h"
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

/**
 * @brief Node structure of XGBoost
 *
 */
struct XGBoostNode : Node<XGBoostParty> {
  vector<XGBoostParty> *parties;
  vector<float> *y;
  vector<float> *prior;
  vector<vector<float>> *gradient, *hessian;
  float min_child_weight, lam, gamma, eps, mi_bound;
  float best_entropy;
  bool use_only_active_party;
  XGBoostNode *left, *right;

  int num_classes;

  float entire_datasetsize = 0;
  vector<float> entire_class_cnt;
  int num_communicated_ciphertext = 0;

  XGBoostNode() {}
  XGBoostNode(vector<XGBoostParty> *parties_, vector<float> *y_,
              int num_classes_, vector<vector<float>> *gradient_,
              vector<vector<float>> *hessian_, vector<int> &idxs_,
              vector<float> *prior_, float min_child_weight_, float lam_,
              float gamma_, float eps_, int depth_, float mi_bound_,
              int active_party_id_ = -1, bool use_only_active_party_ = false,
              int n_job_ = 1) {
    parties = parties_;
    y = y_;
    num_classes = num_classes_;
    gradient = gradient_;
    hessian = hessian_;
    idxs = idxs_;
    prior = prior_;
    min_child_weight = min_child_weight_;
    mi_bound = mi_bound_;
    lam = lam_;
    gamma = gamma_;
    eps = eps_;
    depth = depth_;
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

    try {
      if (use_only_active_party && active_party_id > parties->size()) {
        throw invalid_argument("invalid active_party_id");
      }
    } catch (std::exception &e) {
      std::cerr << e.what() << std::endl;
    }

    val = compute_weight();

    if (is_leaf()) {
      is_leaf_flag = 1;
    } else {
      is_leaf_flag = 0;
    }

    if (is_leaf_flag == 0) {
      tuple<int, int, int> best_split = find_split();
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
   * @brief Get the value assigned to this node.
   *
   * @return float
   */
  vector<float> get_val() { return val; }

  /**
   * @brief Get the evaluation score of this node.
   *
   * @return float
   */
  float get_score() { return score; }

  /**
   * @brief Get the pointer to the left node.
   *
   * @return XGBoostNode
   */
  XGBoostNode get_left() { return *left; }

  /**
   * @brief Get the pointer to the right node.
   *
   * @return XGBoostNode
   */
  XGBoostNode get_right() { return *right; }

  /**
   * @brief Get the num of parties used for this node.
   *
   * @return int
   */
  int get_num_parties() { return parties->size(); }

  /**
   * @brief Compute the weight (val) of this node.
   *
   * @return vector<float>
   */
  vector<float> compute_weight() {
    return xgboost_compute_weight_from_pointer(row_count, gradient, hessian,
                                               idxs, lam);
  }

  /**
   * @brief Compute gain of this node.
   *
   * @param left_grad
   * @param right_grad
   * @param left_hess
   * @param right_hess
   * @return float
   */
  float compute_gain(vector<float> &left_grad, vector<float> &right_grad,
                     vector<float> &left_hess, vector<float> &right_hess) {
    return xgboost_compute_gain(left_grad, right_grad, left_hess, right_hess,
                                gamma, lam);
  }

  /**
   * @brief Find the best split from the specified clients.
   *
   * @param party_id_start
   * @param temp_num_parties
   * @param sum_grad
   * @param sum_hess
   * @param tot_cnt
   * @param temp_y_class_cnt
   */
  void find_split_per_party(int party_id_start, int temp_num_parties,
                            vector<float> &sum_grad, vector<float> &sum_hess,
                            float tot_cnt, vector<float> &temp_y_class_cnt) {

    vector<float> temp_left_class_cnt, temp_right_class_cnt;
    temp_left_class_cnt.resize(num_classes, 0);
    temp_right_class_cnt.resize(num_classes, 0);

    int grad_dim = sum_grad.size();

    for (int temp_party_id = party_id_start;
         temp_party_id < party_id_start + temp_num_parties; temp_party_id++) {

      vector<vector<tuple<vector<float>, vector<float>, float, vector<float>>>>
          search_results =
              parties->at(temp_party_id)
                  .greedy_search_split_from_pointer(gradient, hessian, y, idxs);

      float temp_score, temp_entropy;
      vector<float> temp_left_grad(grad_dim, 0);
      vector<float> temp_left_hess(grad_dim, 0);
      vector<float> temp_right_grad(grad_dim, 0);
      vector<float> temp_right_hess(grad_dim, 0);
      float temp_left_size, temp_right_size;
      bool skip_flag = false;

      for (int j = 0; j < search_results.size(); j++) {
        temp_score = 0;
        temp_entropy = 0;
        temp_left_size = 0;
        temp_right_size = 0;

        for (int c = 0; c < grad_dim; c++) {
          temp_left_grad[c] = 0;
          temp_left_hess[c] = 0;
        }

        for (int c = 0; c < num_classes; c++) {
          temp_left_class_cnt[c] = 0;
          temp_right_class_cnt[c] = 0;
        }

        for (int k = 0; k < search_results[j].size(); k++) {
          if (temp_party_id != active_party_id) {
            num_communicated_ciphertext += 2 * num_classes;

            if ((mi_bound != numeric_limits<float>::infinity()) &&
                (mi_bound != -1)) {
              num_communicated_ciphertext += num_classes;
            }
          }

          for (int c = 0; c < grad_dim; c++) {
            temp_left_grad[c] += get<0>(search_results[j][k])[c];
            temp_left_hess[c] += get<1>(search_results[j][k])[c];
          }
          temp_left_size += get<2>(search_results[j][k]);
          temp_right_size = tot_cnt - temp_left_size;

          for (int c = 0; c < num_classes; c++) {
            temp_left_class_cnt[c] += get<3>(search_results[j][k])[c];
            temp_right_class_cnt[c] =
                temp_y_class_cnt[c] - temp_left_class_cnt[c];
          }

          if ((temp_party_id != active_party_id) &&
              ((!is_satisfied_with_lmir_bound_with_precalculation(
                   num_classes, mi_bound, temp_left_size, y->size(),
                   entire_class_cnt, prior, temp_left_class_cnt)) ||
               (!is_satisfied_with_lmir_bound_with_precalculation(
                   num_classes, mi_bound, temp_right_size, y->size(),
                   entire_class_cnt, prior, temp_right_class_cnt)))) {
            continue;
          }

          skip_flag = false;
          for (int c = 0; c < grad_dim; c++) {
            if (temp_left_hess[c] < min_child_weight ||
                sum_hess[c] - temp_left_hess[c] < min_child_weight) {
              skip_flag = true;
            }
          }
          if (skip_flag) {
            continue;
          }

          for (int c = 0; c < grad_dim; c++) {
            temp_right_grad[c] = sum_grad[c] - temp_left_grad[c];
            temp_right_hess[c] = sum_hess[c] - temp_left_hess[c];
          }

          temp_score = compute_gain(temp_left_grad, temp_right_grad,
                                    temp_left_hess, temp_right_hess);

          if (temp_score > best_score) {
            best_score = temp_score;
            best_entropy = temp_entropy;
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
    vector<float> sum_grad(gradient->at(0).size(), 0);
    vector<float> sum_hess(hessian->at(0).size(), 0);
    for (int i = 0; i < row_count; i++) {
      for (int c = 0; c < sum_grad.size(); c++) {
        sum_grad[c] += gradient->at(idxs[i]).at(c);
        sum_hess[c] += hessian->at(idxs[i]).at(c);
      }
    }

    float tot_cnt = row_count;
    vector<float> temp_y_class_cnt(num_classes, 0);
    for (int r = 0; r < row_count; r++) {
      temp_y_class_cnt[int(y->at(idxs[r]))] += 1;
    }

    float temp_score, temp_left_grad, temp_left_hess;

    if (use_only_active_party) {
      find_split_per_party(active_party_id, 1, sum_grad, sum_hess, tot_cnt,
                           temp_y_class_cnt);
    } else {
      if (n_job == 1) {
        find_split_per_party(0, num_parties, sum_grad, sum_hess, tot_cnt,
                             temp_y_class_cnt);
      } else {
        vector<int> num_parties_per_thread =
            get_num_parties_per_process(n_job, num_parties);

        int cnt_parties = 0;
        vector<thread> threads_parties;
        for (int i = 0; i < n_job; i++) {
          int local_num_parties = num_parties_per_thread[i];
          thread temp_th([this, cnt_parties, local_num_parties, &sum_grad,
                          &sum_hess, tot_cnt, &temp_y_class_cnt] {
            this->find_split_per_party(cnt_parties, local_num_parties, sum_grad,
                                       sum_hess, tot_cnt, temp_y_class_cnt);
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
   * @brief Generate the children nodes.
   *
   * @param best_party_id The index of the best party.
   * @param best_col_id The index of the best feature.
   * @param best_threshold_id The index of the best threshold.
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

    left = new XGBoostNode(
        parties, y, num_classes, gradient, hessian, left_idxs, prior,
        min_child_weight, lam, gamma, eps, depth - 1, mi_bound, active_party_id,
        (use_only_active_party || (!left_is_satisfied_lmir_cond) ||
         (!right_is_satisfied_lmir_cond)),
        n_job);
    if (left->is_leaf_flag == 1) {
      left->party_id = party_id;
    }
    right = new XGBoostNode(
        parties, y, num_classes, gradient, hessian, right_idxs, prior,
        min_child_weight, lam, gamma, eps, depth - 1, mi_bound, active_party_id,
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

    num_communicated_ciphertext += left->num_communicated_ciphertext;
    num_communicated_ciphertext += right->num_communicated_ciphertext;
  }

  /**
   * @brief Return true if this node is a leaf.
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
   * @brief Return true if the node is pure; the assigned labels to this node
   * consist of a unique label.
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
