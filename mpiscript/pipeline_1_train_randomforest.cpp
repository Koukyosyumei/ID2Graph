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
#include "llatvfl/randomforest/mpirandomforest.h"
#include "llatvfl/utils/metric.h"
using namespace std;

const int min_leaf = 1;
const float subsample_cols = 0.8;
const float max_samples_ratio = 0.8;
const int max_timeout_num_patience = 5;
const int M_LPMST = 1;

string folderpath;
string fileprefix;
int num_trees = 20;
int depth = 3;
int n_job = 1;
int skip_round = 0;
float eta = 0.3;
float epsilon_random_unfolding = 0.0;
float epsilon_ldp = -1;
int seconds_wait4timeout = 300;
bool is_weighted_graph = false;
bool save_adj_mat = false;

void parse_args(int argc, char *argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "f:p:r:h:j:c:e:l:o:z:wg")) != -1)
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
            n_job = stoi(string(optarg));
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
        case 'w':
            is_weighted_graph = true;
            break;
        case 'g':
            save_adj_mat = true;
            break;
        default:
            printf("unknown parameter %s is specified", optarg);
            printf("Usage: %s [-f] [-p] [-r] [-h] [-j] [-c] [-e] [-w] ...\n", argv[0]);
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    boost::mpi::environment env(true);
    boost::mpi::communicator world;

    int my_rank = world.rank();

    parse_args(argc, argv);

    string input_filepath = folderpath + "/" + fileprefix + "_" +
                            to_string(my_rank) + "_data.in";
    std::ifstream input_file(input_filepath);

    // --- Load Data --- //
    int num_row_train, num_row_val, num_col, num_party;
    int num_nan_cell = 0;
    input_file >> num_row_train >> num_col >> num_party;

    vector<vector<float>> X_train(num_row_train, vector<float>(num_col));
    vector<float> y_train(num_row_train);
    vector<vector<float>> X_val;
    vector<float> y_val;

    MPIRandomForestParty my_party;

    int temp_count_feature = 0;
    for (int i = 0; i < num_party; i++)
    {
        int num_col = 0;
        input_file >> num_col;

        vector<int> feature_idxs(num_col);
        vector<vector<float>> x(num_row_train, vector<float>(num_col));
        for (int j = 0; j < num_col; j++)
        {
            feature_idxs[j] = temp_count_feature;
            for (int k = 0; k < num_row_train; k++)
            {
                input_file >> x[k][j];
                X_train[k][temp_count_feature] = x[k][j];
            }
            temp_count_feature += 1;
        }

        if (i == my_rank)
        {
            my_party = MPIRandomForestParty(world, x, feature_idxs, my_rank, depth,
                                            num_trees, min_leaf, subsample_cols);
        }
    }

    MPIRandomForestClassifier clf = MPIRandomForestClassifier(subsample_cols, depth, min_leaf,
                                                              max_samples_ratio, num_trees, 0);

    for (int j = 0; j < num_row_train; j++)
    {
        input_file >> y_train[j];
    }

    if (my_rank == 0)
    {
        my_party.y = y_train;
    }

    input_file >> num_row_val;

    X_val.resize(num_row_val, vector<float>(num_col));
    y_val.resize(num_row_val);
    for (int i = 0; i < num_col; i++)
    {
        for (int j = 0; j < num_row_val; j++)
        {
            input_file >> X_val[j][i];
        }
    }

    for (int j = 0; j < num_row_val; j++)
    {
        input_file >> y_val[j];
    }
    input_file.close();

    world.barrier();

    if (my_rank == 0)
    {
        PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
        pair<PaillierPublicKey, PaillierSecretKey> keypair = keygenerator.generate_keypair();
        PaillierPublicKey pk = keypair.first;
        PaillierSecretKey sk = keypair.second;

        my_party.set_publickey(pk);
        my_party.set_secretkey(sk);

        for (int j = 1; j < num_party; j++)
        {
            world.send(j, TAG_PUBLICKEY, pk);
        }
    }
    else
    {
        input_file.close();
        PaillierPublicKey pk;
        world.recv(0, TAG_PUBLICKEY, pk);
        my_party.set_publickey(pk);
        my_party.pk.init_distribution();
    }

    world.barrier();

    std::ofstream result_file;
    chrono::system_clock::time_point start, end;

    if (my_rank == 0)
    {
        string result_filepath = folderpath + "/" + fileprefix + "_result.ans";
        result_file.open(result_filepath, std::ios::out);
        result_file << "train size," << num_row_train << "\n";
        result_file << "val size," << num_row_val << "\n";
        result_file << "column size," << num_col << "\n";
        result_file << "party size," << num_party << "\n";
        result_file << "num of nan," << num_nan_cell << "\n";

        printf("Start training trial=%s\n", fileprefix.c_str());
        start = chrono::system_clock::now();
    }

    world.barrier();
    clf.fit(my_party, num_party);

    if (my_rank == 0)
    {
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
    }

    world.barrier();
    vector<float> predict_proba_train = clf.predict_proba(X_train);
    world.barrier();
    vector<float> predict_proba_val = clf.predict_proba(X_val);
    world.barrier();

    if (my_rank == 0)
    {
        vector<int> y_true_train(y_train.begin(), y_train.end());
        result_file << "Train AUC," << roc_auc_score(predict_proba_train, y_true_train) << "\n";
        vector<int> y_true_val(y_val.begin(), y_val.end());
        result_file << "Val AUC," << roc_auc_score(predict_proba_val, y_true_val) << "\n";
        result_file.close();
    }
}