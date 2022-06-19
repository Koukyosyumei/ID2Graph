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
#include <unordered_map>
#include <stdexcept>
#include "party.h"
#include "../core/node.h"
#include "../utils/utils.h"
using namespace std;

struct RandomForestNode : Node
{
    vector<RandomForestParty> *parties;
    bool use_only_active_party;
    RandomForestNode *left, *right;

    double giniimp;

    RandomForestNode() {}
    RandomForestNode(vector<RandomForestParty> *parties_, vector<double> y_,
                     vector<int> idxs_, int depth_, int active_party_id_ = -1,
                     bool use_only_active_party_ = false, int n_job_ = 1)
    {
        parties = parties_;
        y = y_;
        idxs = idxs_;
        depth = depth_;
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
            std::cout << e.what() << std::endl;
        }

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

    double get_val()
    {
        return val;
    }

    double get_score()
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

    double compute_giniimp()
    {
        double temp_y_pos_cnt;
        for (int r = 0; r < idxs.size(); r++)
        {
            temp_y_pos_cnt += y[idxs[r]];
        }
        double giniimp = 1 - (temp_y_pos_cnt / idxs.size()) * (temp_y_pos_cnt / idxs.size());
        return giniimp;
    }

    double compute_weight()
    {
        /*
        TODO:
            compute the majority classs
         */
    }

    void find_split_per_party(int party_id_start, int temp_num_parties)
    {
        for (int party_id = party_id_start; party_id < party_id_start + temp_num_parties; party_id++)
        {
            vector<vector<double>> search_results = parties->at(party_id).greedy_search_split(idxs, y);

            for (int j = 0; j < search_results.size(); j++)
            {
                double temp_score;
                double temp_giniimp;
                for (int k = 0; k < search_results[j].size(); k++)
                {
                    temp_giniimp = search_results[j][k];
                    temp_score = giniimp - temp_giniimp;

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
        double temp_score, temp_left_grad, temp_left_hess;

        if (use_only_active_party)
        {
            find_split_per_party(active_party_id, 1);
        }
        else
        {
            if (n_job == 1)
            {
                find_split_per_party(0, num_parties);
            }
            else
            {
                vector<int> num_parties_per_thread = get_num_parties_per_process(n_job, num_parties);

                int cnt_parties = 0;
                vector<thread> threads_parties;
                for (int i = 0; i < n_job; i++)
                {
                    int local_num_parties = num_parties_per_thread[i];
                    thread temp_th([this, cnt_parties, local_num_parties]
                                   { this->find_split_per_party(cnt_parties, local_num_parties); });
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
        for (int i = 0; i < idxs.size(); i++)
            if (!any_of(left_idxs.begin(), left_idxs.end(), [&](double x)
                        { return x == idxs[i]; }))
                right_idxs.push_back(idxs[i]);

        left = new RandomForestNode(parties, y, left_idxs,
                                    depth - 1, active_party_id, use_only_active_party);
        if (left->is_leaf_flag == 1)
        {
            left->party_id = party_id;
        }
        right = new RandomForestNode(parties, y, right_idxs,
                                     depth - 1, active_party_id, use_only_active_party);
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
        set<double> s{};
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

    vector<double> predict(vector<vector<double>> &x_new)
    {
        int x_new_size = x_new.size();
        vector<double> y_pred(x_new_size);
        for (int i = 0; i < x_new_size; i++)
        {
            y_pred[i] = predict_row(x_new[i]);
        }
        return y_pred;
    }

    double predict_row(vector<double> &xi)
    {
        if (is_leaf())
            return val;
        else
        {
            if (parties->at(party_id).is_left(record_id, xi))
                return left->predict_row(xi);
            else
                return right->predict_row(xi);
        }
    }
};
