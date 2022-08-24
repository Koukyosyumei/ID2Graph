#pragma once
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
#include "attack.h"
#include "llatvfl/louvain/louvain.h"
#include "../utils/dok.h"
#include "../randomforest/randomforest.h"
#include "../xgboost/xgboost.h"
#include "../secureboost/secureboost.h"
#include "../kmeans/kmeans.cpp"
using namespace std;

struct QuickAttackPipeline
{
    int cluster_size;
    int attack_start_depth;
    int target_party_id;
    int skip_round;
    float epsilon_random_unfolding;
    int seconds_wait4timeout;
    int max_timeout_num_patience;

    SparseMatrixDOK<float> adj_matrix;
    Graph g;
    Louvain louvain;
    KMeans kmeans;
    vector<int> cluster_ids;

    QuickAttackPipeline(int cluster_size_, int attack_start_depth_,
                        int target_party_id_, int skip_round_,
                        float epsilon_random_unfolding_,
                        int seconds_wait4timeout_, int max_timeout_num_patience_)
    {
        cluster_size = cluster_size_;
        attack_start_depth = attack_start_depth_;
        target_party_id = target_party_id_;
        skip_round = skip_round_;
        epsilon_random_unfolding = epsilon_random_unfolding_;
        seconds_wait4timeout = seconds_wait4timeout_;
        max_timeout_num_patience = max_timeout_num_patience_;
    }

    template <typename T>
    void prepare_graph(T &clf)
    {
        adj_matrix = extract_adjacency_matrix_from_forest(&clf, attack_start_depth, 1, skip_round);
        g = Graph(adj_matrix);
    }

    void run_louvain()
    {
        printf("Start community detection (epsilon=%f)\n",
               epsilon_random_unfolding);

        louvain = Louvain(epsilon_random_unfolding);

        future<void> future = async(launch::async, [=]()
                                    { this->louvain.fit(this->g); });
        future_status status;

        int count_timeout = 0;
        chrono::system_clock::time_point start, end;
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
                printf("\033[33mTimeout of community detection -> retry \033[0m\n");
                if (count_timeout == max_timeout_num_patience)
                {
                    throw runtime_error("Maximum number of attempts at timeout reached");
                }
                louvain.reseed(louvain.seed + 1);
                break;
            case future_status::ready:
                float elapsed = chrono::duration_cast<chrono::milliseconds>(end - start).count();
                printf("Community detection is complete %f [ms]\n", elapsed);
                break;
            }
        } while (count_timeout < max_timeout_num_patience && status != future_status::ready);
    }

    void concatenate_basex_with_one_hot_encoding_of_communities_allocation(vector<vector<float>> &base_X)
    {
        int com_size = louvain.g.nodes.size();
        int row_num = base_X[0].size();
        for (int i = 0; i < com_size; i++)
        {
            base_X.push_back(vector<float>(row_num, 0));
            for (int j = 0; j < louvain.g.nodes[i].size(); j++)
            {
                base_X[row_num + i][louvain.g.nodes[i][j]] = 1;
            }
        }
    }

    void run_kmeans()
    {
        kmeans = KMeans(2, 100);
        kmeans.run(base_X);
        cluster_ids = kmeans.get_cluster_ids();
    }

    template <typename T>
    void attack(T &clf, vector<vector<float>> &base_X)
    {
        prepare_graph<T>(clf);
        run_louvain();
        concatenate_basex_with_one_hot_encoding_of_communities_allocation(base_X);
        run_kmeans();
    }
};