#include <cmath>
#include <numeric>
#include <vector>
#include <iterator>
#include <limits>
#include <algorithm>
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

    int col_count; // the number of columns

    unordered_map<int, pair<int, double>> lookup_table; // record_id: (feature_id, threshold)
    vector<int> temp_column_subsample;
    vector<vector<double>> temp_thresholds; // feature_id->threshold
    int seed = 0;

    Party() {}
    Party(vector<vector<double>> &x_, vector<int> &feature_id_, int &party_id_,
          int min_leaf_, double subsample_cols_, int num_precentile_bin_ = 256)
    {
        validate_arguments(x_, feature_id_, party_id_, min_leaf_, subsample_cols_);
        x = x_;
        feature_id = feature_id_;
        party_id = party_id_;
        min_leaf = min_leaf_;
        subsample_cols = subsample_cols_;
        num_percentile_bin = num_precentile_bin_;

        col_count = x.at(0).size();
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

    unordered_map<int, pair<int, double>> get_lookup_table()
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
            percentiles.erase(remove_if(begin(percentiles),
                                        end(percentiles),
                                        [](const auto &value)
                                        { return isnan(value); }),
                              end(percentiles));
            sort(percentiles.begin(), percentiles.end());
            return percentiles;
        }
    }

    bool is_left(int record_id, vector<double> &xi)
    {
        return xi[feature_id[lookup_table[record_id].first]] <= lookup_table[record_id].second;
    }

    vector<vector<pair<double, double>>> greedy_search_split(vector<double> &gradient,
                                                             vector<double> &hessian,
                                                             vector<int> &idxs)
    {
        temp_column_subsample.resize(col_count);
        iota(temp_column_subsample.begin(), temp_column_subsample.end(), 0);
        mt19937 engine(seed);
        seed += 1;
        shuffle(temp_column_subsample.begin(), temp_column_subsample.end(), engine);
        int subsample_col_count = subsample_cols * col_count;

        // feature_id -> [(grad hess)]
        // the threshold of split_candidates_grad_hess[i][j] = temp_thresholds[i][j]
        vector<vector<pair<double, double>>> split_candidates_grad_hess(temp_column_subsample.size());
        temp_thresholds = vector<vector<double>>(temp_column_subsample.size());

        int row_count = idxs.size();
        int recoed_id = 0;

        for (int i = 0; i < subsample_col_count; i++)
        {
            int k = temp_column_subsample[i];
            vector<double> x_col(row_count);
            vector<int> x_col_idxs(row_count);

            for (int r = 0; r < row_count; r++)
                x_col[r] = x[idxs[r]][k];

            vector<double> percentiles = get_percentiles(x_col);

            iota(x_col_idxs.begin(), x_col_idxs.end(), 0);
            sort(x_col_idxs.begin(), x_col_idxs.end(), [&x_col](size_t i1, size_t i2)
                 { return x_col[i1] < x_col[i2]; });

            sort(x_col.begin(), x_col.end());

            int current_min_idx = 0;
            int cumulative_left_size = 0;
            for (int p = 0; p < percentiles.size(); p++)
            {
                double temp_grad = 0;
                double temp_hess = 0;
                int temp_left_size = 0;

                for (int r = current_min_idx; r < row_count; r++)
                {
                    if (x_col[r] <= percentiles[p])
                    {
                        if (!isnan(x_col[r]))
                        {
                            temp_grad += gradient[idxs[x_col_idxs[r]]];
                            temp_hess += hessian[idxs[x_col_idxs[r]]];
                            cumulative_left_size += 1;
                        }
                        else
                        {
                            cout << "detect nan" << endl;
                        }
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
        }

        return split_candidates_grad_hess;
    }

    vector<int> split_rows(vector<int> &idxs, int feature_opt_id, int threshold_opt_id)
    {
        // feature_opt_idがthreshold_opt_id以下のindexを返す
        int row_count = idxs.size();
        vector<double> x_col(row_count);
        for (int r = 0; r < row_count; r++)
            x_col[r] = x[idxs[r]][temp_column_subsample[feature_opt_id]];

        vector<int> left_idxs;
        double threshold = temp_thresholds[feature_opt_id][threshold_opt_id];
        for (int r = 0; r < row_count; r++)
            if ((!isnan(x_col[r])) && (x_col[r] <= threshold))
                left_idxs.push_back(idxs[r]);

        return left_idxs;
    }

    int insert_lookup_table(int feature_opt_id, int threshold_opt_id)
    {
        lookup_table.emplace(lookup_table.size(),
                             make_pair(temp_column_subsample[feature_opt_id],
                                       temp_thresholds[feature_opt_id][threshold_opt_id]));
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

    int party_id, record_id;
    int row_count, num_parties;
    double val, score;
    Node *left, *right;
    int is_leaf_flag = -1; // -1:not calculated yer, 0: is not leaf, 1: is leaf

    Node() {}
    Node(vector<Party> *parties_, vector<double> y_, vector<double> gradient_,
         vector<double> hessian_, vector<int> idxs_,
         double min_child_weight_, double lam_, double gamma_, double eps_,
         int depth_, int active_party_id_ = -1, bool use_only_active_party_ = false)
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

    tuple<int, int, int> find_split()
    {
        double sum_grad = 0;
        double sum_hess = 0;
        for (int i = 0; i < row_count; i++)
        {
            sum_grad += gradient[idxs[i]];
            sum_hess += hessian[idxs[i]];
        }

        double best_score = -1 * numeric_limits<double>::infinity();
        double temp_score, temp_left_grad, temp_left_hess;
        int best_party_id, best_col_id, best_threshold_id;

        if (use_only_active_party)
        {
            vector<vector<pair<double, double>>> search_results =
                parties->at(active_party_id).greedy_search_split(gradient, hessian, idxs);
            for (int j = 0; j < search_results.size(); j++)
            {
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
                        best_party_id = active_party_id;
                        best_col_id = j;
                        best_threshold_id = k;
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < num_parties; i++)
            {
                vector<vector<pair<double, double>>> search_results =
                    parties->at(i).greedy_search_split(gradient, hessian, idxs);

                for (int j = 0; j < search_results.size(); j++)
                {
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
        // TODO: remove idx with nan values from right_idxs;
        vector<int> left_idxs = parties->at(best_party_id).split_rows(idxs, best_col_id, best_threshold_id);
        vector<int> right_idxs;
        for (int i = 0; i < idxs.size(); i++)
            if (!any_of(left_idxs.begin(), left_idxs.end(), [&](double x)
                        { return x == idxs[i]; }))
                right_idxs.push_back(idxs[i]);

        left = new Node(parties, y, gradient, hessian, left_idxs, min_child_weight,
                        lam, gamma, eps, depth - 1, active_party_id, use_only_active_party);
        right = new Node(parties, y, gradient, hessian, right_idxs, min_child_weight,
                         lam, gamma, eps, depth - 1, active_party_id, use_only_active_party);
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

    string print(bool show_purity = false, bool binary_color = true)
    {
        return recursive_print("", false, show_purity, binary_color);
    }

    string recursive_print(string prefix, bool isleft, bool show_purity, bool binary_color)
    {
        string node_info;
        if (is_leaf())
        {
            node_info = to_string(get_val());
            vector<int> temp_idxs = get_idxs();
            if (show_purity)
            {
                int cnt_idxs = temp_idxs.size();
                if (cnt_idxs == 0)
                {
                    node_info += ", null";
                }
                else
                {
                    int cnt_zero = 0;
                    for (int i = 0; i < temp_idxs.size(); i++)
                    {
                        if (y[temp_idxs[i]] == 0)
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
                        node_info += to_string(cnt_idxs);
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
                for (int i = 0; i < temp_idxs.size(); i++)
                {
                    temp_id = temp_idxs[i];
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
        }
        else
        {
            node_info += to_string(get_party_id());
            node_info += ", ";
            node_info += to_string(get_record_id());
        }

        node_info = prefix + "|--" + node_info;
        node_info += "\n";

        if (!is_leaf())
        {
            string next_prefix = "";
            if (isleft)
            {
                next_prefix += "|    ";
            }
            else
            {
                next_prefix += "     ";
            }
            node_info += get_left().recursive_print(prefix + next_prefix, true, show_purity, binary_color);
            node_info += get_right().recursive_print(prefix + next_prefix, false, show_purity, binary_color);
        }

        return node_info;
    }
};
