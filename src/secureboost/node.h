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
#include "utils.h"
using namespace std;

struct Party
{
    vector<vector<double>> x; // a feature vector of this party
    vector<int> feature_id;   // id of the features
    int party_id;             // id of this party
    int min_leaf;
    double subsample_cols; // ratio of subsampled columuns
    int num_percentile_bin;
    bool use_missing_value;
    int seed;

    int col_count; // the number of columns
    int subsample_col_count;

    unordered_map<int, tuple<int, double, int>> lookup_table; // record_id: (feature_id, threshold, missing_value_dir)
    vector<int> temp_column_subsample;
    vector<vector<double>> temp_thresholds; // feature_id->threshold

    Party() {}
    Party(vector<vector<double>> &x_, vector<int> &feature_id_, int &party_id_,
          int min_leaf_, double subsample_cols_, int num_precentile_bin_ = 256,
          bool use_missing_value_ = false, int seed_ = 0)
    {
        validate_arguments(x_, feature_id_, party_id_, min_leaf_, subsample_cols_);
        x = x_;
        feature_id = feature_id_;
        party_id = party_id_;
        min_leaf = min_leaf_;
        subsample_cols = subsample_cols_;
        num_percentile_bin = num_precentile_bin_;
        use_missing_value = use_missing_value_;
        seed = seed_;

        col_count = x.at(0).size();
        subsample_col_count = subsample_cols * col_count;
    }

    void validate_arguments(vector<vector<double>> &x_, vector<int> &feature_id_, int &party_id_,
                            int min_leaf_, double subsample_cols_)
    {
        try
        {
            if (x_.size() == 0)
            {
                throw invalid_argument("x is empty");
            }
        }
        catch (std::exception &e)
        {
            std::cout << e.what() << std::endl;
        }

        try
        {
            if (x_[0].size() != feature_id_.size())
            {
                throw invalid_argument("the number of columns of x is different from the size of feature_id");
            }
        }
        catch (std::exception &e)
        {
            std::cout << e.what() << std::endl;
        }

        try
        {
            if (subsample_cols_ > 1 || subsample_cols_ < 0)
            {
                throw out_of_range("subsample_cols should be in [1, 0]");
            }
        }
        catch (std::exception &e)
        {
            std::cout << e.what() << std::endl;
        }
    }

    unordered_map<int, tuple<int, double, int>> get_lookup_table()
    {
        return lookup_table;
    }

    vector<double> get_percentiles(vector<double> &x_col)
    {
        if (x_col.size() > num_percentile_bin)
        {
            vector<double> probs(num_percentile_bin);
            for (int i = 1; i <= num_percentile_bin; i++)
                probs[i] = double(i) / double(num_percentile_bin);
            vector<double> percentiles_candidate = Quantile<double>(x_col, probs);
            vector<double> percentiles = remove_duplicates<double>(percentiles_candidate);
            return percentiles;
        }
        else
        {
            vector<double> x_col_wo_duplicates = remove_duplicates<double>(x_col);
            vector<double> percentiles(x_col_wo_duplicates.size());
            copy(x_col_wo_duplicates.begin(), x_col_wo_duplicates.end(), percentiles.begin());
            sort(percentiles.begin(), percentiles.end());
            return percentiles;
        }
    }

    bool is_left(int record_id, vector<double> &xi)
    {
        bool flag;
        double x_criterion = xi[feature_id[get<0>(lookup_table[record_id])]];
        if (isnan(x_criterion))
        {
            try
            {
                if (!use_missing_value)
                {
                    throw std::runtime_error("given data contains NaN, but use_missing_value is false");
                }
                else
                {
                    flag = get<2>(lookup_table[record_id]) == 0;
                }
            }
            catch (std::exception &e)
            {
                std::cout << e.what() << std::endl;
            }
        }
        else
        {
            flag = x_criterion <= get<1>(lookup_table[record_id]);
        }
        return flag;
    }

    void subsample_columns()
    {
        temp_column_subsample.resize(col_count);
        iota(temp_column_subsample.begin(), temp_column_subsample.end(), 0);
        mt19937 engine(seed);
        seed += 1;
        shuffle(temp_column_subsample.begin(), temp_column_subsample.end(), engine);
    }

    vector<vector<pair<double, double>>> greedy_search_split(vector<double> &gradient,
                                                             vector<double> &hessian,
                                                             vector<int> &idxs)
    {
        // feature_id -> [(grad hess)]
        // the threshold of split_candidates_grad_hess[i][j] = temp_thresholds[i][j]
        int num_thresholds;
        if (use_missing_value)
            num_thresholds = subsample_col_count * 2;
        else
            num_thresholds = subsample_col_count;
        vector<vector<pair<double, double>>> split_candidates_grad_hess(num_thresholds);
        temp_thresholds = vector<vector<double>>(num_thresholds);

        int row_count = idxs.size();
        int recoed_id = 0;

        for (int i = 0; i < subsample_col_count; i++)
        {
            // extract the necessary data
            int k = temp_column_subsample[i];
            vector<double> x_col(row_count);

            int not_missing_values_count = 0;
            int missing_values_count = 0;
            for (int r = 0; r < row_count; r++)
            {
                if (!isnan(x[idxs[r]][k]))
                {
                    x_col[not_missing_values_count] = x[idxs[r]][k];
                    not_missing_values_count += 1;
                }
                else
                {
                    missing_values_count += 1;
                }
            }
            x_col.resize(not_missing_values_count);

            vector<int> x_col_idxs(not_missing_values_count);
            iota(x_col_idxs.begin(), x_col_idxs.end(), 0);
            sort(x_col_idxs.begin(), x_col_idxs.end(), [&x_col](size_t i1, size_t i2)
                 { return x_col[i1] < x_col[i2]; });

            sort(x_col.begin(), x_col.end());

            // get percentiles of x_col
            vector<double> percentiles = get_percentiles(x_col);

            // enumerate all threshold value (missing value goto right)
            int current_min_idx = 0;
            int cumulative_left_size = 0;
            for (int p = 0; p < percentiles.size(); p++)
            {
                double temp_grad = 0;
                double temp_hess = 0;
                int temp_left_size = 0;

                for (int r = current_min_idx; r < not_missing_values_count; r++)
                {
                    if (x_col[r] <= percentiles[p])
                    {
                        temp_grad += gradient[idxs[x_col_idxs[r]]];
                        temp_hess += hessian[idxs[x_col_idxs[r]]];
                        cumulative_left_size += 1;
                    }
                    else
                    {
                        current_min_idx = r;
                        break;
                    }
                }

                if (cumulative_left_size >= min_leaf &&
                    row_count - cumulative_left_size >= min_leaf)
                {
                    split_candidates_grad_hess[i].push_back(make_pair(temp_grad, temp_hess));
                    temp_thresholds[i].push_back(percentiles[p]);
                }
            }

            // enumerate missing value goto left
            if (use_missing_value)
            {
                int current_max_idx = not_missing_values_count - 1;
                int cumulative_right_size = 0;
                for (int p = percentiles.size() - 1; p >= 0; p--)
                {
                    double temp_grad = 0;
                    double temp_hess = 0;
                    int temp_left_size = 0;

                    for (int r = current_max_idx; r >= 0; r--)
                    {
                        if (x_col[r] >= percentiles[p])
                        {
                            temp_grad += gradient[idxs[x_col_idxs[r]]];
                            temp_hess += hessian[idxs[x_col_idxs[r]]];
                            cumulative_right_size += 1;
                        }
                        else
                        {
                            current_max_idx = r;
                            break;
                        }
                    }

                    if (cumulative_right_size >= min_leaf &&
                        row_count - cumulative_right_size >= min_leaf)
                    {
                        split_candidates_grad_hess[i + subsample_col_count].push_back(make_pair(temp_grad,
                                                                                                temp_hess));
                        temp_thresholds[i + subsample_col_count].push_back(percentiles[p]);
                    }
                }
            }
        }

        return split_candidates_grad_hess;
    }

    vector<int> split_rows(vector<int> &idxs, int feature_opt_pos, int threshold_opt_pos)
    {
        // feature_opt_idがthreshold_opt_id以下のindexを返す
        int feature_opt_id, missing_dir;
        feature_opt_id = temp_column_subsample[feature_opt_pos % subsample_col_count];
        if (feature_opt_pos > subsample_col_count)
        {
            missing_dir = 1;
        }
        else
        {
            missing_dir = 0;
        }
        int row_count = idxs.size();
        vector<double> x_col(row_count);
        for (int r = 0; r < row_count; r++)
            x_col[r] = x[idxs[r]][feature_opt_id];

        vector<int> left_idxs;
        double threshold = temp_thresholds[feature_opt_pos][threshold_opt_pos];
        for (int r = 0; r < row_count; r++)
            if (((!isnan(x_col[r])) && (x_col[r] <= threshold)) ||
                ((isnan(x_col[r])) && (missing_dir == 1)))
                left_idxs.push_back(idxs[r]);

        return left_idxs;
    }

    int insert_lookup_table(int feature_opt_pos, int threshold_opt_pos)
    {
        int feature_opt_id, missing_dir;
        double threshold_opt;
        feature_opt_id = temp_column_subsample[feature_opt_pos % subsample_col_count];
        threshold_opt = temp_thresholds[feature_opt_pos][threshold_opt_pos];

        if (use_missing_value)
        {
            if (feature_opt_pos > subsample_col_count)
            {
                missing_dir = 1;
            }
            else
            {
                missing_dir = 0;
            }
        }
        else
        {
            missing_dir = -1;
        }

        lookup_table.emplace(lookup_table.size(),
                             make_tuple(feature_opt_id, threshold_opt, missing_dir));
        return lookup_table.size() - 1;
    }
};

struct Node
{
    vector<Party> *parties;
    vector<double> y, gradient, hessian;
    vector<int> idxs;
    double min_child_weight, lam, gamma, eps;
    int depth;
    int active_party_id;
    bool use_only_active_party;
    int n_job;

    double best_score = -1 * numeric_limits<double>::infinity();
    int best_party_id, best_col_id, best_threshold_id;

    int party_id, record_id;
    int row_count, num_parties;
    double val, score;
    Node *left, *right;
    int is_leaf_flag = -1; // -1:not calculated yer, 0: is not leaf, 1: is leaf

    Node() {}
    Node(vector<Party> *parties_, vector<double> y_, vector<double> gradient_,
         vector<double> hessian_, vector<int> idxs_,
         double min_child_weight_, double lam_, double gamma_, double eps_,
         int depth_, int active_party_id_ = -1, bool use_only_active_party_ = false, int n_job_ = 1)
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

    Node get_left()
    {
        return *left;
    }

    Node get_right()
    {
        return *right;
    }

    Party get_party(int idx)
    {
        return parties->at(idx);
    }

    int get_num_parties()
    {
        return parties->size();
    }

    double compute_weight()
    {
        double sum_grad = 0;
        double sum_hess = 0;
        for (int i = 0; i < row_count; i++)
        {
            sum_grad += gradient[idxs[i]];
            sum_hess += hessian[idxs[i]];
        }
        return -1 * (sum_grad / (sum_hess + lam));
    }

    double compute_gain(double left_grad, double right_grad, double left_hess, double right_hess)
    {
        return 0.5 * ((left_grad * left_grad) / (left_hess + lam) +
                      (right_grad * right_grad) / (right_hess + lam) -
                      ((left_grad + right_grad) *
                       (left_grad + right_grad) / (left_hess + right_hess + lam))) -
               gamma;
    }

    void find_split_per_party(int party_id_start, int temp_num_parties, double sum_grad, double sum_hess)
    {
        for (int party_id = party_id_start; party_id < party_id_start + temp_num_parties; party_id++)
        {
            vector<vector<pair<double, double>>> search_results =
                parties->at(party_id).greedy_search_split(gradient, hessian, idxs);

            for (int j = 0; j < search_results.size(); j++)
            {
                double temp_score;
                double temp_left_grad = 0;
                double temp_left_hess = 0;
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
        double sum_grad = 0;
        double sum_hess = 0;
        for (int i = 0; i < row_count; i++)
        {
            sum_grad += gradient[idxs[i]];
            sum_hess += hessian[idxs[i]];
        }

        double temp_score, temp_left_grad, temp_left_hess;

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
        for (int i = 0; i < idxs.size(); i++)
            if (!any_of(left_idxs.begin(), left_idxs.end(), [&](double x)
                        { return x == idxs[i]; }))
                right_idxs.push_back(idxs[i]);

        left = new Node(parties, y, gradient, hessian, left_idxs, min_child_weight,
                        lam, gamma, eps, depth - 1, active_party_id, use_only_active_party);
        if (left->is_leaf_flag == 1)
        {
            left->party_id = party_id;
        }
        right = new Node(parties, y, gradient, hessian, right_idxs, min_child_weight,
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

    double get_leaf_purity()
    {
        double leaf_purity = 0;
        if (is_leaf())
        {
            int cnt_idxs = idxs.size();
            if (cnt_idxs == 0)
            {
                leaf_purity = 0.0;
            }
            else
            {
                int cnt_zero = 0;
                for (int i = 0; i < idxs.size(); i++)
                {
                    if (y[idxs[i]] == 0)
                    {
                        cnt_zero += 1;
                    }
                }
                leaf_purity = max(double(cnt_zero) / double(cnt_idxs),
                                  1 - double(cnt_zero) / double(cnt_idxs));
                leaf_purity = leaf_purity * (double(cnt_idxs) / double(y.size()));
            }
        }
        else
        {
            leaf_purity = left->get_leaf_purity() + right->get_leaf_purity();
        }
        return leaf_purity;
    }

    string print(bool show_purity = false, bool binary_color = true, int target_party_id = -1)
    {
        pair<string, bool> result = recursive_print("", false, show_purity, binary_color, target_party_id);
        if (result.second)
        {
            return "";
        }
        else
        {
            return result.first;
        }
    }

    string print_leaf(bool show_purity, bool binary_color)
    {
        string node_info = to_string(get_val());
        if (show_purity)
        {
            int cnt_idxs = idxs.size();
            if (cnt_idxs == 0)
            {
                node_info += ", null";
            }
            else
            {
                int cnt_zero = 0;
                for (int i = 0; i < idxs.size(); i++)
                {
                    if (y[idxs[i]] == 0)
                    {
                        cnt_zero += 1;
                    }
                }
                double purity = max(double(cnt_zero) / double(cnt_idxs),
                                    1 - double(cnt_zero) / double(cnt_idxs));
                node_info += ", ";

                if (binary_color)
                {
                    if (purity < 0.7)
                    {
                        node_info += "\033[32m";
                    }
                    else if (purity < 0.9)
                    {
                        node_info += "\033[33m";
                    }
                    else
                    {
                        node_info += "\033[31m";
                    }
                    node_info += to_string(purity);
                    node_info += " (";
                    node_info += to_string(cnt_zero);
                    node_info += ", ";
                    node_info += to_string(cnt_idxs - cnt_zero);
                    node_info += ")";
                    node_info += "\033[0m";
                }
                else
                {
                    node_info += to_string(purity);
                }
            }
        }
        else
        {
            node_info += ", [";
            int temp_id;
            for (int i = 0; i < idxs.size(); i++)
            {
                temp_id = idxs[i];
                if (binary_color)
                {
                    if (y[temp_id] == 0)
                    {
                        node_info += "\033[32m";
                        node_info += to_string(temp_id);
                        node_info += "\033[0m";
                    }
                    else
                    {
                        node_info += to_string(temp_id);
                    }
                }
                else
                {
                    node_info += to_string(temp_id);
                }
                node_info += ", ";
            }
            node_info += "]";
        }

        return node_info;
    }

    pair<string, bool> recursive_print(string prefix, bool isleft, bool show_purity,
                                       bool binary_color, int target_party_id = -1)
    {
        string node_info;
        bool skip_flag;
        if (is_leaf())
        {
            skip_flag = depth <= 0 && target_party_id != -1 && party_id != target_party_id;
            if (skip_flag)
            {
                node_info = "";
            }
            else
            {
                node_info = print_leaf(show_purity, binary_color);
            }
            node_info = prefix + "|-- " + node_info;
            node_info += "\n";
        }
        else
        {
            node_info += to_string(get_party_id());
            node_info += ", ";
            node_info += to_string(get_record_id());
            node_info = prefix + "|-- " + node_info;

            string next_prefix = "";
            if (isleft)
            {
                next_prefix += "|    ";
            }
            else
            {
                next_prefix += "     ";
            }

            pair<string, bool> left_node_info_and_skip_flag = get_left().recursive_print(prefix + next_prefix, true,
                                                                                         show_purity, binary_color, target_party_id);
            pair<string, bool> right_node_info_and_skip_flag = get_right().recursive_print(prefix + next_prefix, false,
                                                                                           show_purity, binary_color, target_party_id);
            if (left_node_info_and_skip_flag.second && right_node_info_and_skip_flag.second)
            {
                node_info += " -> " + print_leaf(show_purity, binary_color);
                node_info += "\n";
            }
            else
            {
                node_info += "\n";
                node_info += left_node_info_and_skip_flag.first;
                node_info += right_node_info_and_skip_flag.first;
            }

            skip_flag = false;
        }

        return make_pair(node_info, skip_flag);
    }
};
