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
const double subsample_cols = 0.8;
const double max_samples_ratio = 0.8;

string folderpath;
string fileprefix;
int num_trees = 20;
int depth = 3;
int n_job = 1;
bool is_weighted_graph = false;
int skip_round = 0;
float eta = 0.3;

void parse_args(int argc, char *argv[])
{
    int opt;
    while ((opt = getopt(argc, argv, "f:p:r:h:j:c:e:w")) != -1)
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
        case 'w':
            is_weighted_graph = true;
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
    int num_row_train, num_row_val, num_col, num_party;
    scanf("%d %d %d", &num_row_train, &num_col, &num_party);
    vector<vector<double>> X_train(num_row_train, vector<double>(num_col));
    vector<double> y_train(num_row_train);
    vector<RandomForestParty> parties(num_party);

    int temp_count_feature = 0;
    for (int i = 0; i < num_party; i++)
    {
        int num_col = 0;
        scanf("%d", &num_col);
        vector<int> feature_idxs(num_col);
        vector<vector<double>> x(num_row_train, vector<double>(num_col));
        for (int j = 0; j < num_col; j++)
        {
            feature_idxs[j] = temp_count_feature;
            for (int k = 0; k < num_row_train; k++)
            {
                scanf("%lf", &x[k][j]);
                X_train[k][temp_count_feature] = x[k][j];
            }
            temp_count_feature += 1;
        }
        RandomForestParty party(x, feature_idxs, i, min_leaf, subsample_cols);
        parties[i] = party;
    }
    for (int j = 0; j < num_row_train; j++)
        scanf("%lf", &y_train[j]);

    scanf("%d", &num_row_val);
    vector<vector<double>> X_val(num_row_val, vector<double>(num_col));
    vector<double> y_val(num_row_val);
    for (int i = 0; i < num_col; i++)
    {
        for (int j = 0; j < num_row_val; j++)
        {
            scanf("%lf", &X_val[j][i]);
        }
    }
    for (int j = 0; j < num_row_val; j++)
        scanf("%lf", &y_val[j]);

    std::ofstream result_file;
    string result_filepath = folderpath + "/" + fileprefix + "_result.ans";
    result_file.open(result_filepath, std::ios::out);
    result_file << "train size," << num_row_train << "\n";
    result_file << "val size," << num_row_val << "\n";
    result_file << "column size," << num_col << "\n";
    result_file << "party size," << num_party << "\n";

    // --- Check Initialization --- //
    RandomForestClassifier clf = RandomForestClassifier(subsample_cols, depth, min_leaf,
                                                        max_samples_ratio, num_trees,
                                                        0, n_job, 0);

    printf("Start training seed=%s\n", fileprefix.c_str());
    chrono::system_clock::time_point start, end;
    start = chrono::system_clock::now();
    clf.fit(parties, y_train);
    end = chrono::system_clock::now();
    double elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("Training is complete %f [ms] seed=%s\n", elapsed, fileprefix.c_str());

    for (int i = 0; i < clf.estimators.size(); i++)
    {
        result_file << "round " << i + 1 << ": " << 0 << "\n";
    }

    for (int i = 0; i < clf.estimators.size(); i++)
    {
        result_file << "Tree-" << i + 1 << ": " << clf.estimators[i].get_leaf_purity() << "\n";
        result_file << clf.estimators[i].print(true, true).c_str() << "\n";
    }

    vector<double> predict_proba_train = clf.predict_proba(X_train);
    vector<int> y_true_train(y_train.begin(), y_train.end());
    result_file << "Train AUC," << roc_auc_score(predict_proba_train, y_true_train) << "\n";
    vector<double> predict_proba_val = clf.predict_proba(X_val);
    vector<int> y_true_val(y_val.begin(), y_val.end());
    result_file << "Val AUC," << roc_auc_score(predict_proba_val, y_true_val) << "\n";
    result_file.close();

    printf("Start graph extraction seed=%s\n", fileprefix.c_str());
    start = chrono::system_clock::now();
    SparseMatrixDOK<float> adj_matrix = extract_adjacency_matrix_from_forest(&clf, 1, is_weighted_graph, skip_round);
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
