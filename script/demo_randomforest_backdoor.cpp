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
#include "llatvfl/attack/pipe.h"
#include "llatvfl/lpmst/lpmst.h"
#include "llatvfl/louvain/louvain.h"
#include "llatvfl/utils/metric.h"
using namespace std;

const int min_leaf = 1;
const float subsample_cols = 0.8;
const float max_samples_ratio = 0.8;
const int max_timeout_num_patience = 5;

string folderpath;
string fileprefix;
int num_trees = 20;
int depth = 3;
int n_job = 1;
int skip_round = 0;
float eta = 0.3;
float mi_bound = numeric_limits<float>::infinity();
float epsilon_random_unfolding = 0.0;
float epsilon_ldp = -1;
int seconds_wait4timeout = 300;
int attack_start_depth = -1;
bool save_adj_mat = false;
int m_lpmst = 2;

void parse_args(int argc, char *argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "f:p:r:h:j:c:e:l:o:z:b:w:x:g")) != -1)
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
        default:
            printf("unknown parameter %s is specified", optarg);
            printf("Usage: %s [-f] [-p] [-r] [-h] [-j] [-c] [-e] [-w] ...\n", argv[0]);
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
    vector<RandomForestBackDoorParty> parties(num_party);

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
        RandomForestBackDoorParty party(x, num_classes, feature_idxs, i, min_leaf, subsample_cols);
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
    RandomForestBackDoorClassifier clf = RandomForestBackDoorClassifier(num_classes, subsample_cols, depth, min_leaf,
                                                                        max_samples_ratio, num_trees,
                                                                        mi_bound, 0, n_job, 0, 3,
                                                                        attack_start_depth, 1, skip_round,
                                                                        epsilon_random_unfolding,
                                                                        seconds_wait4timeout,
                                                                        max_timeout_num_patience);
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
        result_file << clf.estimators[i].print(true, true).c_str() << "\n";
    }

    vector<vector<float>> predict_proba_train = clf.predict_proba(X_train);
    vector<int> y_true_train(y_train.begin(), y_train.end());
    result_file << "Train AUC," << ovr_roc_auc_score(predict_proba_train, y_true_train) << "\n";

    vector<vector<float>> predict_proba_val = clf.predict_proba(X_val);
    vector<int> y_true_val(y_val.begin(), y_val.end());
    result_file << "Val AUC," << ovr_roc_auc_score(predict_proba_val, y_true_val) << "\n";

    result_file.close();

    std::ofstream cl_file;
    string cl_filepath = folderpath + "/" + fileprefix + "_clusters_and_labels.out";
    cl_file.open(cl_filepath, std::ios::out);

    for (int i = 0; i < num_row_train; i++)
    {
        cl_file << clf.estimated_clusters[i] - 1 << " ";
    }
    cl_file << "\n";

    for (int i = 0; i < num_row_train; i++)
    {
        cl_file << y_train[i] << " ";
    }
    cl_file << "\n";

    int temp_size = clf.matched_target_labels_idxs.size();
    for (int i = 0; i < temp_size; i++)
    {
        cl_file << clf.matched_target_labels_idxs[i] << " ";
    }
    cl_file << "\n";

    cl_file.close();

    printf("All done!");
}
