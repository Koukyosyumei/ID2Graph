#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <numeric>
#include <string>
#include <cassert>
#include <future>
#include <utility>
#include <chrono>
#include <unistd.h>
#include "llatvfl/attack/attack.h"
#include "llatvfl/lpmst/lpmst.h"
#include "llatvfl/louvain/louvain.h"
#include "llatvfl/utils/metric.h"
using namespace std;

const int n_job = 1;
const float subsample_cols = 0.8;
const float max_samples_ratio = 0.8;
const int max_timeout_num_patience = 15;

string folderpath;
string fileprefix;
int num_trees = 20;
int depth = 3;
int min_leaf = 1;
int skip_round = 0;
float eta = 0.3;
float mi_bound = numeric_limits<float>::infinity();
float epsilon_ldp = -1;
float epsilon_random_unfolding = 1.0;
int seconds_wait4timeout = 300;
int attack_start_depth = -1;
bool save_adj_mat = false;
bool save_tree_html = false;
int m_lpmst = 2;

void parse_args(int argc, char *argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "f:p:r:h:j:c:e:l:o:z:b:w:x:gq")) != -1)
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
            num_trees = stoi(string(optarg));
            break;
        case 'h':
            depth = stoi(string(optarg));
            break;
        case 'j':
            min_leaf = stoi(string(optarg));
            break;
        case 'c':
            skip_round = stoi(string(optarg));
            break;
        case 'e':
            eta = stof(string(optarg));
            break;
        case 'l':
            epsilon_random_unfolding = stof(string(optarg));
            break;
        case 'o':
            epsilon_ldp = stof(string(optarg));
            break;
        case 'z':
            seconds_wait4timeout = stoi(string(optarg));
            break;
        case 'b':
            mi_bound = stof(string(optarg));
            break;
        case 'w':
            attack_start_depth = stoi(string(optarg));
            break;
        case 'x':
            m_lpmst = stoi(string(optarg));
            break;
        case 'g':
            save_adj_mat = true;
            break;
        case 'q':
            save_tree_html = true;
            break;
        default:
            printf("unknown parameter %s is specified", optarg);
            printf("Usage: %s [-f] [-p] [-r] [-h] [-j] [-c] [-e] [-l] [-o] [-z] [-b] [-w] [-x] [-g] [-q] ...\n", argv[0]);
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    parse_args(argc, argv);

    // --- Load Data --- //
    int num_classes, num_row_train, num_row_val, num_col, num_party;
    if (scanf("%d %d %d %d", &num_classes, &num_row_train, &num_col, &num_party) != 4)
    {
        try
        {
            throw runtime_error("bad input");
        }
        catch (std::runtime_error e)
        {
            cerr << e.what() << "\n";
        }
    }
    vector<vector<float>> X_train(num_row_train, vector<float>(num_col));
    vector<float> y_train(num_row_train);
    vector<RandomForestParty> parties(num_party);

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
                cerr << e.what() << "\n";
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
                        cerr << e.what() << "\n";
                    }
                }
                X_train[k][temp_count_feature] = x[k][j];
            }
            temp_count_feature += 1;
        }
        RandomForestParty party(x, num_classes, feature_idxs, i, min_leaf, subsample_cols);
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
                cerr << e.what() << "\n";
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
            cerr << e.what() << "\n";
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
                    cerr << e.what() << "\n";
                }
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
                cerr << e.what() << "\n";
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

    // --- Check Initialization --- //
    RandomForestClassifier clf = RandomForestClassifier(num_classes, subsample_cols, depth, min_leaf,
                                                        max_samples_ratio, num_trees,
                                                        mi_bound, 0, n_job, 0);
    printf("Start training trial=%s\n", fileprefix.c_str());
    chrono::system_clock::time_point start, end;
    start = chrono::system_clock::now();
    if (epsilon_ldp > 0)
    {
        LPMST lp_1st(m_lpmst, epsilon_ldp, 0);
        lp_1st.fit(clf, parties, y_train);
    }
    else
    {
        clf.fit(parties, y_train);
    }
    end = chrono::system_clock::now();
    float elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("Training is complete %f [ms] trial=%s\n", elapsed, fileprefix.c_str());

    for (int i = 0; i < clf.estimators.size(); i++)
    {
        result_file << "round " << i + 1 << ": " << 0 << "\n";
    }

    for (int i = 0; i < clf.estimators.size(); i++)
    {
        result_file << "Tree-" << i + 1 << ": " << clf.estimators[i].get_leaf_purity() << "\n";
        result_file << clf.estimators[i].print(false, true).c_str() << "\n";

        if (save_tree_html)
        {
            std::ofstream tree_html_file;
            string tree_html_filepath = folderpath + "/" + fileprefix + "_" + to_string(i) + "_tree.html";
            tree_html_file.open(tree_html_filepath, std::ios::out);
            tree_html_file << clf.estimators[i].to_html().c_str();
            tree_html_file.close();
        }
    }

    vector<vector<float>> predict_proba_train = clf.predict_proba(X_train);
    vector<int> y_true_train(y_train.begin(), y_train.end());
    result_file << "Train AUC," << ovr_roc_auc_score(predict_proba_train, y_true_train) << "\n";

    vector<vector<float>> predict_proba_val = clf.predict_proba(X_val);
    vector<int> y_true_val(y_val.begin(), y_val.end());
    result_file << "Val AUC," << ovr_roc_auc_score(predict_proba_val, y_true_val) << "\n";

    result_file.close();

    clf.free_intermediate_resources();

    printf("Start graph extraction trial=%s\n", fileprefix.c_str());
    start = chrono::system_clock::now();
    SparseMatrixDOK<float> adj_matrix = extract_adjacency_matrix_from_forest(&clf, attack_start_depth, 1, skip_round);
    printf("Graph construction.... trial==%s\n", fileprefix.c_str());
    Graph g = Graph(adj_matrix);
    end = chrono::system_clock::now();
    elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("Graph extraction is complete %f [ms] trial=%s\n", elapsed, fileprefix.c_str());

    if (save_adj_mat)
    {
        adj_matrix.save(folderpath + "/" + fileprefix + "_adj_mat.txt");
    }

    printf("Start community detection (trial=%s)\n", fileprefix.c_str());
    Louvain louvain = Louvain();
    vector<float> epsilon_random_unfolding_candidates = {epsilon_random_unfolding, 0.5, 0.1};
    float temp_epsilon_random_unfolding = epsilon_random_unfolding;
    future<void> future = async(launch::async, [&louvain, &g, &temp_epsilon_random_unfolding]()
                                { louvain.reset_epsilon(temp_epsilon_random_unfolding); louvain.fit(g); });
    future_status status;
    int count_timeout = 0;
    do
    {
        count_timeout++;
        start = chrono::system_clock::now();
        status = future.wait_for(chrono::seconds(seconds_wait4timeout));
        end = chrono::system_clock::now();

        temp_epsilon_random_unfolding = epsilon_random_unfolding_candidates[(count_timeout - 1 / 5)];
        printf("Set epsilon to %f (trial=%s)\n", louvain.epsilon, fileprefix.c_str());

        switch (status)
        {
        case future_status::deferred:
            printf("deferred\n");
            break;
        case future_status::timeout:
            printf("\033[33mTimeout of community detection -> retry trial=%s\033[0m\n",
                   fileprefix.c_str());
            if (count_timeout == max_timeout_num_patience)
            {
                throw runtime_error("Maximum number of attempts at timeout reached");
            }
            louvain.reseed(louvain.seed + 1);
            break;
        case future_status::ready:
            elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
            printf("Community detection is complete %f [ms] trial=%s\n", elapsed, fileprefix.c_str());
            break;
        }
    } while (count_timeout < max_timeout_num_patience && status != future_status::ready);

    printf("Saving extracted communities trial=%s\n", fileprefix.c_str());
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
