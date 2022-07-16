#pragma once
#include "../xgboost/node.h"
#include "../paillier/paillier.h"
#include "party.h"
using namespace std;

struct SecureBoostNode : Node<SecureBoostParty>
{
    vector<SecureBoostParty> *parties;
    vector<PaillierCipherText> gradient, hessian;
    vector<float> vanila_gradient, vanila_hessian;
    float min_child_weight, lam, gamma, eps;
    bool use_only_active_party;
    SecureBoostNode *left, *right;

    SecureBoostNode() {}
    SecureBoostNode(vector<SecureBoostParty> *parties_, vector<float> &y_,
                    vector<PaillierCipherText> &gradient_,
                    vector<PaillierCipherText> &hessian_,
                    vector<float> &vanila_gradient_,
                    vector<float> &vanila_hessian_,
                    vector<int> &idxs_, float min_child_weight_, float lam_,
                    float gamma_, float eps_, int depth_, int active_party_id_ = 0,
                    bool use_only_active_party_ = false, int n_job_ = 1)
    {
        parties = parties_;
        y = y_;
        gradient = gradient_;
        hessian = hessian_;
        vanila_gradient = vanila_gradient_;
        vanila_hessian = vanila_hessian_;
        idxs = idxs_;
        min_child_weight = min_child_weight_;
        lam = lam_;
        gamma = gamma_;
        eps = eps_;
        depth = depth_;
        active_party_id = active_party_id_;
        use_only_active_party = use_only_active_party_;
        n_job = n_job_;

        row_count = idxs.size();
        num_parties = parties->size();

        val = compute_weight();
        // tuple<int, int, int> best_split = find_split();

        if (is_leaf())
        {
            is_leaf_flag = 1;
        }
        else
        {
            is_leaf_flag = 0;
        }

        if (is_leaf_flag == 0)
        {
            tuple<int, int, int> best_split = find_split();
            party_id = get<0>(best_split);
            record_id = parties->at(party_id).insert_lookup_table(get<1>(best_split), get<2>(best_split));
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

    SecureBoostNode get_left()
    {
        return *left;
    }

    SecureBoostNode get_right()
    {
        return *right;
    }

    int get_num_parties()
    {
        return parties->size();
    }

    float compute_weight()
    {
        float sum_grad = 0;
        float sum_hess = 0;
        for (int i = 0; i < row_count; i++)
        {
            sum_grad += vanila_gradient[idxs[i]];
            sum_hess += vanila_hessian[idxs[i]];
        }
        return -1 * (sum_grad / (sum_hess + lam));
    }

    float compute_gain(float left_grad, float right_grad, float left_hess, float right_hess)
    {
        return 0.5 * ((left_grad * left_grad) / (left_hess + lam) +
                      (right_grad * right_grad) / (right_hess + lam) -
                      ((left_grad + right_grad) *
                       (left_grad + right_grad) / (left_hess + right_hess + lam))) -
               gamma;
    }

    void find_split_per_party(int party_id_start, int temp_num_parties, float sum_grad, float sum_hess)
    {
        for (int party_id = party_id_start; party_id < party_id_start + temp_num_parties; party_id++)
        {

            vector<vector<pair<float, float>>> search_results;
            if (party_id == active_party_id)
            {
                search_results = parties->at(party_id).greedy_search_split(vanila_gradient, vanila_hessian, idxs);
            }
            else
            {
                vector<vector<pair<PaillierCipherText, PaillierCipherText>>> encrypted_search_result =
                    parties->at(party_id).greedy_search_split_encrypt(gradient, hessian, idxs);
                int temp_result_size = encrypted_search_result.size();
                search_results.resize(temp_result_size);
                int temp_vec_size;
                for (int j = 0; j < temp_result_size; j++)
                {
                    temp_vec_size = encrypted_search_result[j].size();
                    search_results[j].resize(temp_vec_size);
                    for (int k = 0; k < temp_vec_size; k++)
                    {
                        search_results[j][k] = make_pair(parties->at(active_party_id)
                                                             .sk.decrypt<float>(
                                                                 encrypted_search_result[j][k].first),
                                                         parties->at(active_party_id)
                                                             .sk.decrypt<float>(
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
                        sum_hess - temp_left_hess < min_child_weight)
                        continue;

                    temp_score = compute_gain(temp_left_grad, sum_grad - temp_left_grad,
                                              temp_left_hess, sum_hess - temp_left_hess);

                    if (temp_score > best_score)
                    {
                        best_score = temp_score;
                        best_party_id = party_id;
                        best_col_id = j;
                        best_threshold_id = k;
                    }
                }
            }
        }
    }

    tuple<int, int, int> find_split()
    {
        float sum_grad = 0;
        float sum_hess = 0;
        for (int i = 0; i < row_count; i++)
        {
            sum_grad += vanila_gradient[idxs[i]];
            sum_hess += vanila_hessian[idxs[i]];
        }

        float temp_score, temp_left_grad, temp_left_hess;

        if (use_only_active_party)
        {
            find_split_per_party(active_party_id, 1, sum_grad, sum_hess);
        }
        else
        {
            if (n_job == 1)
            {
                find_split_per_party(0, num_parties, sum_grad, sum_hess);
            }
            else
            {
                vector<int> num_parties_per_thread = get_num_parties_per_process(n_job, num_parties);
                int cnt_parties = 0;
                vector<thread> threads_parties;
                for (int i = 0; i < n_job; i++)
                {
                    int local_num_parties = num_parties_per_thread[i];
                    thread temp_th([this, cnt_parties, local_num_parties, sum_grad, sum_hess]
                                   { this->find_split_per_party(cnt_parties, local_num_parties, sum_grad, sum_hess); });
                    threads_parties.push_back(move(temp_th));
                    cnt_parties += num_parties_per_thread[i];
                }
                for (int i = 0; i < num_parties; i++)
                {
                    threads_parties[i].join();
                }
            }
        }
        score = best_score;
        return make_tuple(best_party_id, best_col_id, best_threshold_id);
    }

    void make_children_nodes(int best_party_id, int best_col_id, int best_threshold_id)
    {
        // TODO: remove idx with nan values from right_idxs;
        vector<int> left_idxs = parties->at(best_party_id).split_rows(idxs, best_col_id, best_threshold_id);
        vector<int> right_idxs;
        for (int i = 0; i < row_count; i++)
            if (!any_of(left_idxs.begin(), left_idxs.end(), [&](float x)
                        { return x == idxs[i]; }))
                right_idxs.push_back(idxs[i]);

        left = new SecureBoostNode(parties, y, gradient, hessian,
                                   vanila_gradient, vanila_hessian,
                                   left_idxs, min_child_weight,
                                   lam, gamma, eps, depth - 1, active_party_id, use_only_active_party);
        if (left->is_leaf_flag == 1)
        {
            left->party_id = party_id;
        }
        right = new SecureBoostNode(parties, y, gradient, hessian,
                                    vanila_gradient, vanila_hessian,
                                    right_idxs, min_child_weight,
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
            if (s.insert(y[idxs[i]]).second)
            {
                if (s.size() == 2)
                    return false;
            }
        }
        return true;
    }
};