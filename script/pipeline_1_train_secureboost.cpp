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
#include "llatvfl/paillier/keygenerator.h"
#include "llatvfl/lpmst/lpmst.h"
#include "llatvfl/louvain/louvain.h"
#include "llatvfl/utils/metric.h"
using namespace std;

const int min_leaf = 1;
const int max_bin = 32;
const float lam = 1.0;
const float const_gamma = 0.0;
const float eps = 1.0;
const float min_child_weight = -1 * numeric_limits<float>::infinity();
const float subsample_cols = 0.8;
const int max_timeout_num_patience = 5;
const int M_LPMST = 1;

string folderpath;
string fileprefix;
int boosting_rounds = 20;
int completely_secure_round = 0;
int depth = 3;
int n_job = 1;
float learning_rate = 0.3;
float weight_entropy = 0.0;
float eta = 0.3;
float epsilon_random_unfolding = 0.0;
float epsilon_ldp = -1;
int seconds_wait4timeout = 300;
bool use_missing_value = false;
bool is_weighted_graph = false;
bool save_adj_mat = false;

void parse_args(int argc, char *argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "f:p:r:c:a:e:h:j:l:o:z:y:mwg")) != -1)
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
            completely_secure_round = stoi(string(optarg));
            break;
        case 'a':
            learning_rate = stof(string(optarg));
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
        case 'l':
            epsilon_random_unfolding = stof(string(optarg));
            break;
        case 'o':
            epsilon_ldp = stof(string(optarg));
            break;
        case 'z':
            seconds_wait4timeout = stoi(string(optarg));
            break;
        case 'y':
            weight_entropy = stof(string(optarg));
            break;
        case 'm':
            use_missing_value = true;
            break;
        case 'w':
            is_weighted_graph = true;
            break;
        case 'g':
            save_adj_mat = true;
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
            cerr << e.what() << "\n";
        }
    }
    vector<vector<float>> X_train(num_row_train, vector<float>(num_col));
    vector<float> y_train(num_row_train);
    vector<SecureBoostParty> parties(num_party);

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
                if (use_missing_value && x[k][j] == -1)
                {
                    x[k][j] = nan("");
                    num_nan_cell += 1;
                }
                X_train[k][temp_count_feature] = x[k][j];
            }
            temp_count_feature += 1;
        }
        SecureBoostParty party(x, 2, feature_idxs, i, min_leaf, subsample_cols, max_bin, use_missing_value);
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
                cerr << e.what() << "\n";
            }
        }
    }

    PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
    pair<PaillierPublicKey, PaillierSecretKey> keypair = keygenerator.generate_keypair();
    PaillierPublicKey pk = keypair.first;
    PaillierSecretKey sk = keypair.second;

    for (int i = 0; i < num_party; i++)
    {
        parties[i].set_publickey(pk);
    }
    parties[0].set_secretkey(sk);

    std::ofstream result_file;
    string result_filepath = folderpath + "/" + fileprefix + "_result.ans";
    result_file.open(result_filepath, std::ios::out);
    result_file << "train size," << num_row_train << "\n";
    result_file << "val size," << num_row_val << "\n";
    result_file << "column size," << num_col << "\n";
    result_file << "party size," << num_party << "\n";
    result_file << "num of nan," << num_nan_cell << "\n";

    // --- Check Initialization --- //
    SecureBoostClassifier clf = SecureBoostClassifier(2, subsample_cols,
                                                      min_child_weight,
                                                      depth, min_leaf,
                                                      learning_rate,
                                                      boosting_rounds,
                                                      lam, const_gamma, eps,
                                                      0, completely_secure_round,
                                                      0.5, n_job, true);

    printf("Start training trial=%s\n", fileprefix.c_str());
    chrono::system_clock::time_point start, end;
    start = chrono::system_clock::now();
    if (epsilon_ldp > 0)
    {
        LPMST lp_1st(M_LPMST, epsilon_ldp, 0);
        lp_1st.fit(clf, parties, y_train);
    }
    else
    {
        clf.fit(parties, y_train);
    }
    end = chrono::system_clock::now();
    float elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    result_file << "training time," << elapsed << "\n";
    printf("Training is complete %f [ms] trial=%s\n", elapsed, fileprefix.c_str());

    for (int i = 0; i < clf.logging_loss.size(); i++)
    {
        result_file << "round " << i + 1 << ": " << clf.logging_loss[i] << "\n";
    }

    for (int i = 0; i < clf.estimators.size(); i++)
    {
        result_file << "Tree-" << i + 1 << ": " << clf.estimators[i].get_leaf_purity() << "\n";
        // result_file << clf.estimators[i].print(true, true).c_str() << "\n";
    }

    vector<vector<float>> predict_proba_train = clf.predict_proba(X_train);
    vector<float> predict_proba_train_pos(predict_proba_train.size());
    for (int i = 0; i < predict_proba_train.size(); i++)
    {
        predict_proba_train_pos[i] = predict_proba_train[i][1];
    }
    vector<int> y_true_train(y_train.begin(), y_train.end());
    result_file << "Train AUC," << roc_auc_score(predict_proba_train_pos, y_true_train) << "\n";

    vector<vector<float>> predict_proba_val = clf.predict_proba(X_val);
    vector<float> predict_proba_val_pos(predict_proba_val.size());
    for (int i = 0; i < predict_proba_val.size(); i++)
    {
        predict_proba_val_pos[i] = predict_proba_val[i][1];
    }
    vector<int> y_true_val(y_val.begin(), y_val.end());
    result_file << "Val AUC," << roc_auc_score(predict_proba_val_pos, y_true_val) << "\n";

    result_file.close();

    printf("Start graph extraction trial=%s\n", fileprefix.c_str());
    start = chrono::system_clock::now();
    SparseMatrixDOK<float> adj_matrix = extract_adjacency_matrix_from_forest(&clf, 1, is_weighted_graph, completely_secure_round, eta);
    Graph g = Graph(adj_matrix);
    end = chrono::system_clock::now();
    elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("Graph extraction is complete %f [ms] trial=%s\n", elapsed, fileprefix.c_str());

    if (save_adj_mat)
    {
        adj_matrix.save(folderpath + "/" + fileprefix + "_adj_mat.txt");
    }

    printf("Start community detection (epsilon=%f) trial=%s\n",
           epsilon_random_unfolding, fileprefix.c_str());
    Louvain louvain = Louvain(epsilon_random_unfolding);
    future<void> future = async(launch::async, [&louvain, &g]()
                                { louvain.fit(g); });
    future_status status;
    int count_timeout = 0;
    do
    {
        count_timeout++;
        start = chrono::system_clock::now();
        status = future.wait_for(chrono::seconds(seconds_wait4timeout));
        end = chrono::system_clock::now();

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
