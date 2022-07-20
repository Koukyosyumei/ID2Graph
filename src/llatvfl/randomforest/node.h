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
#include "gini.h"
#include "../core/node.h"
#include "../utils/utils.h"
using namespace std;

struct RandomForestNode : Node<RandomForestParty>
{
    vector<RandomForestParty> *parties;
    RandomForestNode *left, *right;

    float giniimp;

    RandomForestNode() {}
    RandomForestNode(vector<RandomForestParty> *parties_, vector<float> &y_,
                     vector<int> &idxs_, int depth_, int active_party_id_ = -1, int n_job_ = 1)
    {
        parties = parties_;
        y = y_;
        idxs = idxs_;
        depth = depth_;
        active_party_id = active_party_id_;
        n_job = n_job_;

        row_count = idxs.size();
        num_parties = parties->size();

        giniimp = compute_giniimp();
        val = compute_weight();

        tuple<int, int, int> best_split = find_split();

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

    RandomForestNode get_left()
    {
        return *left;
    }

    RandomForestNode get_right()
    {
        return *right;
    }

    int get_num_parties()
    {
        return parties->size();
    }

    float compute_giniimp()
    {
        float temp_y_pos_cnt = 0;
        for (int r = 0; r < row_count; r++)
        {
            temp_y_pos_cnt += y[idxs[r]];
        }
        float temp_y_neg_cnt = row_count - temp_y_pos_cnt;
        float giniimp = 1 -
                        (temp_y_pos_cnt / row_count) * (temp_y_pos_cnt / row_count) -
                        (temp_y_neg_cnt / row_count) * (temp_y_neg_cnt / row_count);
        return giniimp;
    }

    float compute_weight()
    {
        // TODO: support multi class
        float pos_ratio = 0;
        for (int r = 0; r < row_count; r++)
        {
            pos_ratio += y[idxs[r]];
        }
        return pos_ratio / float(row_count);
    }

    void find_split_per_party(int party_id_start, int temp_num_parties, float tot_cnt, float pos_cnt)
    {
        float temp_left_size, temp_left_poscnt, temp_right_size, temp_right_poscnt;
        float temp_score, temp_giniimp, temp_left_giniimp, temp_right_giniimp;
        float neg_cnt = tot_cnt - pos_cnt;

        for (int temp_party_id = party_id_start; temp_party_id < party_id_start + temp_num_parties; temp_party_id++)
        {
            vector<vector<pair<float, float> > > search_results = parties->at(temp_party_id).greedy_search_split(idxs, y);

            int num_search_results = search_results.size();
            int temp_num_search_results_j;
            for (int j = 0; j < num_search_results; j++)
            {
                temp_left_size = 0;
                temp_left_poscnt = 0;

                temp_num_search_results_j = search_results[j].size();
                for (int k = 0; k < temp_num_search_results_j; k++)
                {
                    temp_left_size += search_results[j][k].first;
                    temp_left_poscnt += search_results[j][k].second;
                    temp_right_size = tot_cnt - temp_left_size;
                    temp_right_poscnt = pos_cnt - temp_left_poscnt;

                    temp_left_giniimp = calc_giniimp(temp_left_size, temp_left_poscnt);
                    temp_right_giniimp = calc_giniimp(temp_right_size, temp_right_poscnt);
                    temp_giniimp = temp_left_giniimp * (temp_left_size / tot_cnt) +
                                   temp_right_giniimp * (temp_right_size / tot_cnt);

                    temp_score = giniimp - temp_giniimp;
                    if (temp_score > best_score)
                    {
                        best_score = temp_score;
                        best_party_id = temp_party_id;
                        best_col_id = j;
                        best_threshold_id = k;
                    }
                }
            }
        }
    }

    tuple<int, int, int> find_split()
    {
        float temp_score;
        float pos_cnt = 0;
        float tot_cnt = row_count;

        for (int i = 0; i < row_count; i++)
        {
            pos_cnt += float(y[idxs[i]]);
        }

        if (n_job == 1)
        {
            find_split_per_party(0, num_parties, tot_cnt, pos_cnt);
        }
        else
        {
            vector<int> num_parties_per_thread = get_num_parties_per_process(n_job, num_parties);

            int cnt_parties = 0;
            vector<thread> threads_parties;
            for (int i = 0; i < n_job; i++)
            {
                int local_num_parties = num_parties_per_thread[i];
                thread temp_th([this, cnt_parties, local_num_parties, tot_cnt, pos_cnt]
                               { this->find_split_per_party(cnt_parties, local_num_parties, tot_cnt, pos_cnt); });
                threads_parties.push_back(move(temp_th));
                cnt_parties += num_parties_per_thread[i];
            }
            for (int i = 0; i < num_parties; i++)
            {
                threads_parties[i].join();
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

        left = new RandomForestNode(parties, y, left_idxs,
                                    depth - 1, active_party_id);
        if (left->is_leaf_flag == 1)
        {
            left->party_id = party_id;
        }
        right = new RandomForestNode(parties, y, right_idxs,
                                     depth - 1, active_party_id);
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
