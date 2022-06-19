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

struct Node
{
    vector<double> y;
    vector<int> idxs;

    int depth;
    int active_party_id;
    int n_job;

    int party_id, record_id;
    int row_count, num_parties;
    double val, score;

    int best_party_id, best_col_id, best_threshold_id;

    double best_score = -1 * numeric_limits<double>::infinity();
    int is_leaf_flag = -1; // -1:not calculated yer, 0: is not leaf, 1: is leaf

    Node(){};

    virtual vector<int> get_idxs() = 0;
    virtual int get_party_id() = 0;
    virtual int get_record_id() = 0;
    virtual double get_val() = 0;
    virtual double get_score() = 0;
    virtual int get_num_parties() = 0;
    virtual double compute_weight() = 0;
    virtual tuple<int, int, int> find_split() = 0;
    virtual void make_children_nodes(int best_party_id, int best_col_id, int best_threshold_id) = 0;
    virtual bool is_leaf() = 0;
    virtual bool is_pure() = 0;
    virtual vector<double> predict(vector<vector<double>> &x_new) = 0;
};