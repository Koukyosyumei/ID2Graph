#pragma once
#include <numeric>
#include <vector>
#include <iostream>
#include <iterator>
#include <limits>
#include <algorithm>
#include <set>
#include <tuple>
using namespace std;

template <typename PartyType>
struct Node
{
    vector<PartyType> *parties;
    vector<float> y;
    vector<int> idxs;

    int depth;
    int active_party_id;
    int n_job;

    int party_id, record_id;
    int row_count, num_parties;
    float val, score;

    int best_party_id, best_col_id, best_threshold_id;

    float best_score = -1 * numeric_limits<float>::infinity();
    int is_leaf_flag = -1; // -1:not calculated yer, 0: is not leaf, 1: is leaf

    Node(){};

    virtual vector<int> get_idxs() = 0;
    virtual int get_party_id() = 0;
    virtual int get_record_id() = 0;
    virtual float get_val() = 0;
    virtual float get_score() = 0;
    virtual int get_num_parties() = 0;
    virtual float compute_weight() = 0;
    virtual tuple<int, int, int> find_split() = 0;
    virtual void make_children_nodes(int best_party_id, int best_col_id, int best_threshold_id) = 0;
    virtual bool is_leaf() = 0;
    virtual bool is_pure() = 0;
    // virtual vector<float> predict(vector<vector<float>> &x_new) = 0;
};