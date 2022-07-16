#pragma once
#include "../xgboost/node.h"
#include "../paillier/paillier.h"
#include "mpiparty.h"
using namespace std;

struct MPISecureBoostNode : Node<MPISecureBoostParty>
{
    MPISecureBoostParty *active_party;
    int parties_num, max_depth;
    float min_child_weight, lam, gamma, eps;
    bool use_only_active_party;
    MPISecureBoostNode *left, *right;

    MPISecureBoostNode() {}
    MPISecureBoostNode(MPISecureBoostParty *active_party_, int parties_num_,
                       vector<int> &idxs_, int max_depth_, float min_child_weight_, float lam_,
                       float gamma_, float eps_, int depth_, int active_party_id_ = 0,
                       bool use_only_active_party_ = false)
    {
        active_party = active_party_;
        parties_num = parties_num_;
        idxs = idxs_;
        max_depth = max_depth_;
        min_child_weight = min_child_weight_;
        lam = lam_;
        gamma = gamma_;
        eps = eps_;
        depth = depth_;
        active_party_id = active_party_id_;
        use_only_active_party = use_only_active_party_;

        y = active_party->y;

        row_count = idxs.size();
        num_parties = parties_num;

        active_party->set_instance_space(idxs);
        val = compute_weight();

        if (is_leaf())
        {
            is_leaf_flag = 1;
        }
        else
        {
            is_leaf_flag = 0;
        }

        for (int i = 0; i < parties_num; i++)
        {
            if (i != active_party_id)
            {
                active_party->world.send(i, TAG_DEPTH, depth);
                active_party->world.send(i, TAG_ISLEAF, is_leaf_flag);
            }
        }

        if (is_leaf_flag == 0)
        {
            tuple<int, int, int> best_split = find_split();
            party_id = get<0>(best_split);
            make_children_nodes(get<0>(best_split), get<1>(best_split), get<2>(best_split));
        }
    }

    vector<int> get_idxs()
    {
        return idxs;
    }

    int get_party_id()
    {
        return party_id;
    }

    int get_record_id()
    {
        return record_id;
    }

    float get_val()
    {
        return val;
    }

    float get_score()
    {
        return score;
    }

    int get_num_parties()
    {
        return parties_num;
    }

    MPISecureBoostNode get_left()
    {
        return *left;
    }

    MPISecureBoostNode get_right()
    {
        return *right;
    }

    float compute_weight()
    {
        return active_party->compute_weight();
    }

    tuple<int, int, int> find_split()
    {
        float temp_score, temp_left_grad, temp_left_hess;
        vector<vector<pair<float, float>>> search_results;
        vector<vector<pair<PaillierCipherText, PaillierCipherText>>> encrypted_search_result;

        if (use_only_active_party)
        {
            active_party->calc_sum_grad_and_hess();
            search_results = active_party->greedy_search_split();

            for (int j = 0; j < search_results.size(); j++)
            {
                float temp_score;
                float temp_left_grad = 0;
                float temp_left_hess = 0;
                for (int k = 0; k < search_results[j].size(); k++)
                {
                    temp_left_grad += search_results[j][k].first;
                    temp_left_hess += search_results[j][k].second;

                    if (temp_left_hess < min_child_weight ||
                        active_party->sum_hess - temp_left_hess < min_child_weight)
                        continue;

                    temp_score = active_party->compute_gain(temp_left_grad, active_party->sum_grad - temp_left_grad,
                                                            temp_left_hess, active_party->sum_hess - temp_left_hess);

                    if (temp_score > best_score)
                    {
                        best_score = temp_score;
                        best_party_id = active_party_id;
                        best_col_id = j;
                        best_threshold_id = k;
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < parties_num; i++)
            {
                if (i == active_party_id)
                {
                    active_party->calc_sum_grad_and_hess();
                }
                else
                {
                    if (max_depth == depth)
                    {
                        active_party->world.send(i, TAG_VEC_ENCRYPTED_GRAD, active_party->gradient);
                        active_party->world.send(i, TAG_VEC_ENCRYPTED_HESS, active_party->hessian);
                    }

                    active_party->world.send(i, TAG_INSTANCE_SPACE, idxs);
                }
            }

            for (int i = 0; i < parties_num; i++)
            {
                if (i == active_party_id)
                {
                    search_results = active_party->greedy_search_split();
                }
                else
                {
                    active_party->world.recv(i, TAG_SEARCH_RESULTS, encrypted_search_result);

                    int temp_result_size = encrypted_search_result.size();
                    search_results.resize(temp_result_size);
                    int temp_vec_size;
                    for (int j = 0; j < temp_result_size; j++)
                    {
                        temp_vec_size = encrypted_search_result[j].size();
                        search_results[j].resize(temp_vec_size);
                        for (int k = 0; k < temp_vec_size; k++)
                        {
                            search_results[j][k] = make_pair(active_party->sk.decrypt<float>(
                                                                 encrypted_search_result[j][k].first),
                                                             active_party->sk.decrypt<float>(
                                                                 encrypted_search_result[j][k].second));
                        }
                    }
                }

                for (int j = 0; j < search_results.size(); j++)
                {
                    float temp_score;
                    float temp_left_grad = 0;
                    float temp_left_hess = 0;
                    for (int k = 0; k < search_results[j].size(); k++)
                    {
                        temp_left_grad += search_results[j][k].first;
                        temp_left_hess += search_results[j][k].second;

                        if (temp_left_hess < min_child_weight ||
                            active_party->sum_hess - temp_left_hess < min_child_weight)
                            continue;

                        temp_score = active_party->compute_gain(temp_left_grad, active_party->sum_grad - temp_left_grad,
                                                                temp_left_hess, active_party->sum_hess - temp_left_hess);

                        if (temp_score > best_score)
                        {
                            best_score = temp_score;
                            best_party_id = i;
                            best_col_id = j;
                            best_threshold_id = k;
                        }
                    }
                }
            }
        }
        score = best_score;
        return make_tuple(best_party_id, best_col_id, best_threshold_id);
    }

    void make_children_nodes(int best_party_id, int best_col_id, int best_threshold_id)
    {
        for (int i = 0; i < parties_num; i++)
        {
            if (i != active_party_id)
            {
                active_party->world.send(i, TAG_BEST_PARTY_ID, best_party_id);
            }
        }

        // TODO: remove idx with nan values from right_idxs;
        vector<int> left_idxs;
        if (best_party_id == active_party_id)
        {
            record_id = active_party->insert_lookup_table(best_col_id, best_threshold_id);
            left_idxs = active_party->split_rows(idxs, best_col_id, best_threshold_id);
        }
        else
        {
            active_party->world.send(best_party_id, TAG_BEST_SPLIT_COL_ID, best_col_id);
            active_party->world.send(best_party_id, TAG_BEST_SPLIT_THRESHOLD_ID, best_threshold_id);
            active_party->world.recv(best_party_id, TAG_RECORDID, record_id);
            active_party->world.recv(best_party_id, TAG_BEST_INSTANCE_SPACE, left_idxs);
        }

        vector<int> right_idxs;
        for (int i = 0; i < row_count; i++)
            if (!any_of(left_idxs.begin(), left_idxs.end(), [&](float x)
                        { return x == idxs[i]; }))
                right_idxs.push_back(idxs[i]);

        left = new MPISecureBoostNode(active_party, parties_num,
                                      left_idxs, max_depth, min_child_weight,
                                      lam, gamma, eps, depth - 1, active_party_id, use_only_active_party);
        if (left->is_leaf_flag == 1)
        {
            left->party_id = party_id;
        }
        right = new MPISecureBoostNode(active_party, parties_num,
                                       right_idxs, max_depth, min_child_weight,
                                       lam, gamma, eps, depth - 1, active_party_id, use_only_active_party);
        if (right->is_leaf_flag == 1)
        {
            right->party_id = party_id;
        }
    }

    bool is_leaf()
    {
        if (is_leaf_flag == -1)
        {
            return is_pure() || std::isinf(score) || depth <= 0;
        }
        else
        {
            return is_leaf_flag;
        }
    }

    bool is_pure()
    {
        set<float> s{};
        for (int i = 0; i < row_count; i++)
        {
            if (s.insert(active_party->y[idxs[i]]).second)
            {
                if (s.size() == 2)
                    return false;
            }
        }
        return true;
    }
};