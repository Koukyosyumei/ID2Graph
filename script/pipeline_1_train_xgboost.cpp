#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <numeric>
#include <string>
#include <cassert>
#include <chrono>
#include <unistd.h>
#include "../src/llatvfl/attack/attack.h"
#include "../src/llatvfl/louvain/louvain.h"
#include "../src/llatvfl/utils/metric.h"
using namespace std;

const int min_leaf = 1;
const int max_bin = 32;
const float learning_rate = 0.3;
const float lam = 0.0;
const float const_gamma = 0.0;
const float eps = 1.0;
const float min_child_weight = -1 * numeric_limits<float>::infinity();
const float subsample_cols = 0.8;

string folderpath;
string fileprefix;
int boosting_rounds = 20;
int completelly_secure_round = 0;
int depth = 3;
int n_job = 1;
bool use_missing_value = false;
bool is_weighted_graph = false;
int skip_round = 0;
float eta = 0.3;

void parse_args(int argc, char *argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "f:p:r:c:e:h:j:mw")) != -1)
    {
        switch (opt)
        {
        case 'f':
            folderpath = string(optarg);
            break;
        case 'p':
            fileprefix = string(optarg);
            break;
        case 'r':
            boosting_rounds = stoi(string(optarg));
            break;
        case 'c':
            completelly_secure_round = stoi(string(optarg));
            break;
        case 'e':
            eta = stof(string(optarg));
            break;
        case 'h':
            depth = stoi(string(optarg));
            break;
        case 'j':
            n_job = stoi(string(optarg));
            break;
        case 'm':
            use_missing_value = true;
            break;
        case 'w':
            is_weighted_graph = true;
            break;
        default:
            printf("unknown parameter %s is specified", optarg);
            printf("Usage: %s [-f] [-p] [-r] [-c] [-j] [-m] [-w] ...\n", argv[0]);
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    parse_args(argc, argv);

    // --- Load Data --- //
    int num_row_train, num_row_val, num_col, num_party;
    int num_nan_cell = 0;
    if (scanf("%d %d %d", &num_row_train, &num_col, &num_party) != 3)
    {
        try
        {
            throw runtime_error("bad input");
        }
        catch (std::runtime_error e)
        {
            cerr << e.what() << "/n";
        }
    }
    vector<vector<float>> X_train(num_row_train, vector<float>(num_col));
    vector<float> y_train(num_row_train);
    vector<XGBoostParty> parties(num_party);

    int temp_count_feature = 0;
    for (int i = 0; i < num_party; i++)
    {
        int num_col = 0;
        if (scanf("%d", &num_col) != 1)
        {
            try
            {
                throw runtime_error("bad input");
            }
            catch (std::runtime_error e)
            {
                cerr << e.what() << "/n";
            }
        }
        vector<int> feature_idxs(num_col);
        vector<vector<float>> x(num_row_train, vector<float>(num_col));
        for (int j = 0; j < num_col; j++)
        {
            feature_idxs[j] = temp_count_feature;
            for (int k = 0; k < num_row_train; k++)
            {
                if (scanf("%f", &x[k][j]) != 1)
                {
                    try
                    {
                        throw runtime_error("bad input");
                    }
                    catch (std::runtime_error e)
                    {
                        cerr << e.what() << "/n";
                    }
                }
                if (use_missing_value && x[k][j] == -1)
                {
                    x[k][j] = nan("");
                    num_nan_cell += 1;
                }
                X_train[k][temp_count_feature] = x[k][j];
            }
            temp_count_feature += 1;
        }
        XGBoostParty party(x, feature_idxs, i, min_leaf, subsample_cols, max_bin, use_missing_value);
        parties[i] = party;
    }
    for (int j = 0; j < num_row_train; j++)
    {
        if (scanf("%f", &y_train[j]) != 1)
        {
            try
            {
                throw runtime_error("bad input");
            }
            catch (std::runtime_error e)
            {
                cerr << e.what() << "/n";
            }
        }
    }

    if (scanf("%d", &num_row_val) != 1)
    {
        try
        {
            throw runtime_error("bad input");
        }
        catch (std::runtime_error e)
        {
            cerr << e.what() << "/n";
        }
    }
    vector<vector<float>> X_val(num_row_val, vector<float>(num_col));
    vector<float> y_val(num_row_val);
    for (int i = 0; i < num_col; i++)
    {
        for (int j = 0; j < num_row_val; j++)
        {
            if (scanf("%f", &X_val[j][i]) != 1)
            {
                try
                {
                    throw runtime_error("bad input");
                }
                catch (std::runtime_error e)
                {
                    cerr << e.what() << "/n";
                }
            }
            if (use_missing_value && X_val[j][i] == -1)
            {
                X_val[j][i] = nan("");
            }
        }
    }
    for (int j = 0; j < num_row_val; j++)
    {
        if (scanf("%f", &y_val[j]) != 1)
        {
            try
            {
                throw runtime_error("bad input");
            }
            catch (std::runtime_error e)
            {
                cerr << e.what() << "/n";
            }
        }
    }

    std::ofstream result_file;
    string result_filepath = folderpath + "/" + fileprefix + "_result.ans";
    result_file.open(result_filepath, std::ios::out);
    result_file << "train size," << num_row_train << "\n";
    result_file << "val size," << num_row_val << "\n";
    result_file << "column size," << num_col << "\n";
    result_file << "party size," << num_party << "\n";
    result_file << "num of nan," << num_nan_cell << "\n";

    // --- Check Initialization --- //
    XGBoostClassifier clf = XGBoostClassifier(subsample_cols,
                                              min_child_weight,
                                              depth, min_leaf,
                                              learning_rate,
                                              boosting_rounds,
                                              lam, const_gamma, eps,
                                              0, completelly_secure_round,
                                              0.5, n_job, true);

    printf("Start training seed=%s\n", fileprefix.c_str());
    chrono::system_clock::time_point start, end;
    start = chrono::system_clock::now();
    clf.fit(parties, y_train);
    end = chrono::system_clock::now();
    float elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("Training is complete %f [ms] seed=%s\n", elapsed, fileprefix.c_str());

    for (int i = 0; i < clf.logging_loss.size(); i++)
    {
        result_file << "round " << i + 1 << ": " << clf.logging_loss[i] << "\n";
    }

    for (int i = 0; i < clf.estimators.size(); i++)
    {
        result_file << "Tree-" << i + 1 << ": " << clf.estimators[i].get_leaf_purity() << "\n";
        // result_file << clf.estimators[i].print(true, true).c_str() << "\n";
    }

    vector<float> predict_proba_train = clf.predict_proba(X_train);
    vector<int> y_true_train(y_train.begin(), y_train.end());
    result_file << "Train AUC," << roc_auc_score(predict_proba_train, y_true_train) << "\n";
    vector<float> predict_proba_val = clf.predict_proba(X_val);
    vector<int> y_true_val(y_val.begin(), y_val.end());
    result_file << "Val AUC," << roc_auc_score(predict_proba_val, y_true_val) << "\n";
    result_file.close();

    printf("Start graph extraction seed=%s\n", fileprefix.c_str());
    start = chrono::system_clock::now();
    SparseMatrixDOK<float> adj_matrix = extract_adjacency_matrix_from_forest(&clf, 1, is_weighted_graph, skip_round, eta);
    Graph g = Graph(adj_matrix);
    end = chrono::system_clock::now();
    elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("Graph extraction is complete %f [ms] seed=%s\n", elapsed, fileprefix.c_str());

    printf("Start community detection seed=%s\n", fileprefix.c_str());
    start = chrono::system_clock::now();
    Louvain louvain = Louvain();
    louvain.fit(g);
    end = chrono::system_clock::now();
    elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("Community detection is complete %f [ms] seed=%s\n", elapsed, fileprefix.c_str());

    std::ofstream com_file;
    string filepath = folderpath + "/" + fileprefix + "_communities.out";
    com_file.open(filepath, std::ios::out);
    com_file << louvain.g.nodes.size() << "\n";
    com_file << g.num_nodes << "\n";
    for (int i = 0; i < louvain.g.nodes.size(); i++)
    {
        for (int j = 0; j < louvain.g.nodes[i].size(); j++)
        {
            com_file << louvain.g.nodes[i][j] << " ";
        }
        com_file << "\n";
    }
    com_file.close();
}
