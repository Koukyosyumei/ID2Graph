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
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include "llatvfl/secureboost/mpisecureboost.h"
#include "llatvfl/paillier/keygenerator.h"
#include "llatvfl/utils/metric.h"
using namespace std;

const int min_leaf = 1;
const int max_bin = 32;
const float lam = 0.0;
const float const_gamma = 0.0;
const float eps = 1.0;
const float min_child_weight = -1 * numeric_limits<float>::infinity();
const float subsample_cols = 0.8;

string folderpath;
string fileprefix;
int boosting_rounds = 5;
int completely_secure_round = 0;
int depth = 3;
float learning_rate = 0.3;
float eta = 0.3;
bool use_missing_value = false;

void parse_args(int argc, char *argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "f:p:r:c:a:e:h:m")) != -1)
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
        case 'm':
            use_missing_value = true;
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
    boost::mpi::environment env(true);
    boost::mpi::communicator world;

    int my_rank = world.rank();
    cout << "Start rank=" << my_rank << endl;

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

    MPISecureBoostParty my_party;

    int temp_count_feature = 0;
    cout << "Load training data rank=" << my_rank << endl;
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
                if (use_missing_value && x[k][j] == -1)
                {
                    x[k][j] = nan("");
                    num_nan_cell += 1;
                }
                X_train[k][temp_count_feature] = x[k][j];
            }
            temp_count_feature += 1;
        }

        if (i == my_rank)
        {
            cout << "Setup party rank=" << my_rank << endl;
            my_party = MPISecureBoostParty(world, x, feature_idxs, my_rank, depth,
                                           boosting_rounds, min_leaf, subsample_cols,
                                           const_gamma, lam, max_bin, use_missing_value);
            break;
        }
    }

    world.barrier();

    if (my_rank == 0)
    {
        cout << "Load training labels rank=" << my_rank << endl;

        for (int j = 0; j < num_row_train; j++)
        {
            input_file >> y_train[j];
        }

        input_file >> num_row_val;

        cout << "Load validation data rank=" << my_rank << endl;
        vector<vector<float>> X_val(num_row_val, vector<float>(num_col));
        vector<float> y_val(num_row_val);
        for (int i = 0; i < num_col; i++)
        {
            for (int j = 0; j < num_row_val; j++)
            {
                input_file >> X_val[j][i];
                if (use_missing_value && X_val[j][i] == -1)
                {
                    X_val[j][i] = nan("");
                }
            }
        }

        cout << "Load validation label rank=" << my_rank << endl;
        for (int j = 0; j < num_row_val; j++)
        {
            input_file >> y_val[j];
        }

        input_file.close();

        cout << "Generate keypair rank=" << my_rank << endl;

        PaillierKeyGenerator keygenerator = PaillierKeyGenerator(512);
        pair<PaillierPublicKey, PaillierSecretKey> keypair = keygenerator.generate_keypair();
        PaillierPublicKey pk = keypair.first;
        PaillierSecretKey sk = keypair.second;

        my_party.set_publickey(pk);
        my_party.set_secretkey(sk);

        PaillierCipherText demo_ct = my_party.pk.encrypt(13);
        cout << my_party.sk.decrypt<int>(demo_ct) << endl;

        for (int j = 1; j < num_party; j++)
        {
            cout << "Send public key to rank=" << j << endl;
            world.send(j, TAG_PUBLICKEY, pk);
        }

        my_party.y = y_train;

        std::ofstream result_file;
        string result_filepath = folderpath + "/" + fileprefix + "_result.ans";
        result_file.open(result_filepath, std::ios::out);
        result_file << "train size," << num_row_train << "\n";
        result_file << "val size," << num_row_val << "\n";
        result_file << "column size," << num_col << "\n";
        result_file << "party size," << num_party << "\n";
        result_file << "num of nan," << num_nan_cell << "\n";

        MPISecureBoostClassifier clf = MPISecureBoostClassifier(subsample_cols,
                                                                min_child_weight,
                                                                depth, min_leaf,
                                                                learning_rate, boosting_rounds,
                                                                lam, const_gamma, eps,
                                                                0, completely_secure_round,
                                                                0.5, true);

        printf("Start training trial=%s\n", fileprefix.c_str());
        chrono::system_clock::time_point start, end;
        start = chrono::system_clock::now();
        clf.fit(my_party, num_party, y_train);
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

        vector<float> predict_proba_train = clf.predict_proba(X_train);
        vector<int> y_true_train(y_train.begin(), y_train.end());
        result_file << "Train AUC," << roc_auc_score(predict_proba_train, y_true_train) << "\n";
        vector<float> predict_proba_val = clf.predict_proba(X_val);
        vector<int> y_true_val(y_val.begin(), y_val.end());
        result_file << "Val AUC," << roc_auc_score(predict_proba_val, y_true_val) << "\n";
        result_file.close();
    }
    else
    {
        input_file.close();
        PaillierPublicKey pk;
        world.recv(0, TAG_PUBLICKEY, pk);
        cout << "Recv public key rank=" << my_rank << endl;
        my_party.set_publickey(pk);
        my_party.pk.init_distribution();
        cout << my_party.pk.encrypt(123).c << endl;

        cout << "Run rank=" << my_rank << endl;
        my_party.run_as_passive();
    }
}
