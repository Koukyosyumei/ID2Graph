#pragma once
#include "../randomforest/randomforest.h"
#include <algorithm>
#include <limits>
#include <queue>

inline bool mismatch_preds(RandomForestNode *node, vector<float> *original_y) {
  std::vector<float> noised_val = node->val;
  vector<float> original_val(node->num_classes, 0);

  for (int r = 0; r < node->row_count; r++) {
    original_val[int(original_y->at(node->idxs[r]))] +=
        1 / float(node->row_count);
  }

  int noised_argmax, original_argmax;
  float max_val = 0;
  for (int i = 0; i < noised_val.size(); i++) {
    if (noised_val[i] > max_val) {
      max_val = noised_val[i];
      noised_argmax = i;
    }
  }
  max_val = 0;
  for (int i = 0; i < original_val.size(); i++) {
    if (original_val[i] > max_val) {
      max_val = original_val[i];
      original_argmax = i;
    }
  }

  return noised_argmax != original_argmax;
}

inline void mark_contaminated_nodes(RandomForestNode *node,
                                    vector<RandomForestNode *> &target_nodes,
                                    vector<float> *original_y) {
  if (node->is_leaf()) {
    node->is_all_subsequent_children_contaminated =
        mismatch_preds(node, original_y);
    return;
  } else {
    mark_contaminated_nodes(node->left, target_nodes, original_y);
    mark_contaminated_nodes(node->right, target_nodes, original_y);
    if (node->left->is_all_subsequent_children_contaminated ||
        node->right->is_all_subsequent_children_contaminated) {
      if (mismatch_preds(node, original_y)) {
        node->is_all_subsequent_children_contaminated = true;
      } else {
        target_nodes.push_back(node);
      }
    }
  }
}

inline void selfrepair_tree(RandomForestTree &tree, vector<float> *original_y) {
  // queue<RandomForestNode *> que;
  // que.push(&tree.dtree);

  // RandomForestNode *temp_node;
  std::vector<RandomForestNode *> root_of_problems;
  mark_contaminated_nodes(&tree.dtree, root_of_problems, original_y);

  /*
  while (!que.empty()) {
    temp_node = que.front();
    que.pop();

    if (!temp_node->is_leaf_flag) {
      if (mismatch_preds(temp_node->left, original_y) ||
          mismatch_preds(temp_node->right, original_y)) {
        root_of_problems.push_back(temp_node);
      } else {
        que.push(temp_node->left);
        que.push(temp_node->right);
      }
    }
  }
  */

  for (RandomForestNode *node : root_of_problems) {

    node->best_party_id = -1;
    node->best_col_id = -1;
    node->best_threshold_id = -1;
    node->best_score = -1 * numeric_limits<float>::infinity();
    node->is_leaf_flag = -1;
    node->is_pure_flag = -1;

    node->use_only_active_party = true;
    node->lmir_flag_exclude_passive_parties = true;
    node->y = original_y;
    node->entire_class_cnt.resize(node->num_classes, 0);
    node->entire_datasetsize = node->y->size();
    for (int i = 0; i < node->entire_datasetsize; i++) {
      node->entire_class_cnt[int(node->y->at(i))] += 1.0;
    }

    node->giniimp = node->compute_giniimp();
    node->val = node->compute_weight();
    tuple<int, int, int> best_split = node->find_split();

    if (node->is_leaf()) {
      node->is_leaf_flag = 1;
    } else {
      node->is_leaf_flag = 0;
    }

    if (node->is_leaf_flag == 0) {
      node->party_id = get<0>(best_split);
      if (node->party_id != -1) {
        node->record_id =
            node->parties->at(node->party_id)
                .insert_lookup_table(get<1>(best_split), get<2>(best_split));
        node->make_children_nodes(get<0>(best_split), get<1>(best_split),
                                  get<2>(best_split));
      } else {
        node->is_leaf_flag = 1;
      }
    }
  }
}

inline void selfrepair_forest(RandomForestClassifier &clf,
                              std::vector<float> *original_y) {
  for (RandomForestTree &tree : clf.estimators) {
    selfrepair_tree(tree, original_y);
  }
}
