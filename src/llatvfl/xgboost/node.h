#pragma once
#include <cmath>
#include <numeric>
#include <vector>
#include <iterator>
#include <limits>
#include <algorithm>
#include <thread>
#include <set>
#include <tuple>
#include <random>
#include <ctime>
#include <string>
#include <queue>
#include <unordered_map>
#include <stdexcept>
#include "party.h"
#include "../core/node.h"
#include "../utils/metric.h"
#include "../utils/utils.h"
using namespace std;

struct XGBoostNode : Node<XGBoostParty>
{
    vector<XGBoostParty> *parties;
    vector<float> gradient, hessian;
    float min_child_weight, lam, gamma, eps, weight_entropy;
    bool use_only_active_party;
    XGBoostNode *left, *right;

    XGBoostNode() {}
    XGBoostNode(vector<XGBoostParty> *parties_, vector<float> &y_, vector<float> &gradient_,
                vector<float> &hessian_, vector<int> &idxs_,
                float min_child_weight_, float lam_, float gamma_, float eps_,
                int depth_, float weight_entropy_, int active_party_id_ = -1, bool use_only_active_party_ = false, int n_job_ = 1)
    {
        parties = parties_;
        y = y_;
        gradient = gradient_;
        hessian = hessian_;
        idxs = idxs_;
        min_child_weight = min_child_weight_;
        lam = lam_;
        gamma = gamma_;
        eps = eps_;
        depth = depth_;
        weight_entropy = weight_entropy_;
        active_party_id = active_party_id_;
        use_only_active_party = use_only_active_party_;
        n_job = n_job_;

        try
        {
            if (use_only_active_party && active_party_id > parties->size())
            {
                throw invalid_argument("invalid active_party_id");
            }
        }
        catch (std::exception &e)
        {
            std::cerr << e.what() << std::endl;
        }

        row_count = idxs.size();
        num_parties = parties->size();

        val = compute_weight();

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
            if (party_id != -1)
            {
                record_id = parties->at(party_id).insert_lookup_table(get<1>(best_split), get<2>(best_split));
                make_children_nodes(get<0>(best_split), get<1>(best_split), get<2>(best_split));
            }
            else
            {
                is_leaf_flag = 1;
            }
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

    XGBoostNode get_left()
    {
        return *left;
    }

    XGBoostNode get_right()
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
            sum_grad += gradient[idxs[i]];
            sum_hess += hessian[idxs[i]];
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

    void find_split_per_party(int party_id_start, int temp_num_parties, float sum_grad, float sum_hess, float tot_cnt, float pos_cnt)
    {
        for (int temp_party_id = party_id_start; temp_party_id < party_id_start + temp_num_parties; temp_party_id++)
        {
            vector<vector<tuple<float, float, float, float>>> search_results =
                parties->at(temp_party_id).greedy_search_split(gradient, hessian, y, idxs);

            cout << "---" << endl;
            for (int j = 0; j < search_results.size(); j++)
            {
                float temp_score;
                float temp_left_grad = 0;
                float temp_left_hess = 0;
                float temp_left_size = 0;
                float temp_left_poscnt = 0;
                float temp_right_size, temp_right_poscnt;

                for (int k = 0; k < search_results[j].size(); k++)
                {
                    temp_left_grad += get<0>(search_results[j][k]);
                    temp_left_hess += get<1>(search_results[j][k]);
                    temp_left_size += get<2>(search_results[j][k]);
                    temp_left_poscnt += get<3>(search_results[j][k]);
                    temp_right_size = tot_cnt - temp_left_size;
                    temp_right_poscnt = pos_cnt - temp_left_poscnt;

                    if (temp_left_hess < min_child_weight ||
                        sum_hess - temp_left_hess < min_child_weight)
                        continue;

                    temp_score = compute_gain(temp_left_grad, sum_grad - temp_left_grad,
                                              temp_left_hess, sum_hess - temp_left_hess);
                    cout << temp_score;
                    if (weight_entropy != 0)
                    {
                        float entropy = (calc_entropy(temp_left_size, temp_left_poscnt) * (temp_left_size / tot_cnt) +
                                         calc_entropy(temp_right_size, temp_right_poscnt) * (temp_right_size / tot_cnt));
                        temp_score += weight_entropy * entropy;
                        cout << " " << entropy << " " << weight_entropy * entropy << " " << temp_score << endl;
                    }

                    if (temp_score > best_score)
                    {
                        best_score = temp_score;
                        best_party_id = temp_party_id;
                        best_col_id = j;
                        best_threshold_id = k;
                    }
                }
            }
            cout << "---" << endl;
        }
    }

    tuple<int, int, int> find_split()
    {
        float sum_grad = 0;
        float sum_hess = 0;
        for (int i = 0; i < row_count; i++)
        {
            sum_grad += gradient[idxs[i]];
            sum_hess += hessian[idxs[i]];
        }

        float tot_cnt = row_count;
        float pos_cnt = 0;
        for (int i = 0; i < row_count; i++)
        {
            pos_cnt += float(y[idxs[i]]);
        }

        float temp_score, temp_left_grad, temp_left_hess;

        if (use_only_active_party)
        {
            find_split_per_party(active_party_id, 1, sum_grad, sum_hess, tot_cnt, pos_cnt);
        }
        else
        {
            if (n_job == 1)
            {
                find_split_per_party(0, num_parties, sum_grad, sum_hess, tot_cnt, pos_cnt);
            }
            else
            {
                vector<int> num_parties_per_thread = get_num_parties_per_process(n_job, num_parties);

                int cnt_parties = 0;
                vector<thread> threads_parties;
                for (int i = 0; i < n_job; i++)
                {
                    int local_num_parties = num_parties_per_thread[i];
                    thread temp_th([this, cnt_parties, local_num_parties, sum_grad, sum_hess, tot_cnt, pos_cnt]
                                   { this->find_split_per_party(cnt_parties, local_num_parties, sum_grad, sum_hess, tot_cnt, pos_cnt); });
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

        left = new XGBoostNode(parties, y, gradient, hessian, left_idxs, min_child_weight,
                               lam, gamma, eps, depth - 1, weight_entropy, active_party_id, use_only_active_party);
        if (left->is_leaf_flag == 1)
        {
            left->party_id = party_id;
        }
        right = new XGBoostNode(parties, y, gradient, hessian, right_idxs, min_child_weight,
                                lam, gamma, eps, depth - 1, weight_entropy, active_party_id, use_only_active_party);
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