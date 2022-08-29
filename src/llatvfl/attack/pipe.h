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
#include "../utils/utils.h"
#include "../utils/dok.h"
#include "../randomforest/randomforest.h"
#include "../xgboost/xgboost.h"
#include "../secureboost/secureboost.h"
#include "../kmeans/kmeans.h"
using namespace std;

struct QuickAttackPipeline
{
    int num_class;
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

    QuickAttackPipeline(int num_class_, int attack_start_depth_,
                        int target_party_id_, int skip_round_,
                        float epsilon_random_unfolding_,
                        int seconds_wait4timeout_, int max_timeout_num_patience_)
    {
        num_class = num_class_;
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

        future<void> future = async(launch::async, [=]() mutable
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

    void concatenate_basex_with_one_hot_encoding_of_communities_allocation(vector<vector<float>> &base_X_normalized)
    {
        int com_size = louvain.g.nodes.size();
        int row_num = base_X_normalized.size();
        int column_num = base_X_normalized[0].size();
        for (int i = 0; i < com_size; i++)
        {
            for (int j = 0; j < row_num; j++)
            {
                base_X_normalized[j].push_back(0);
            }

            for (int j = 0; j < louvain.g.nodes[i].size(); j++)
            {
                base_X_normalized[louvain.g.nodes[i][j]][column_num + i] = 1;
            }
        }
    }

    vector<int> run_kmeans(vector<vector<float>> &base_X_normalized)
    {
        kmeans = KMeans(num_class);
        kmeans.run(base_X_normalized);
        return kmeans.get_cluster_ids();
    }

    vector<int> match_prior_and_estimatedclusters(vector<float> &priors, vector<int> &estimated_clusters, int target_class)
    {
        cout << "A" << endl;
        vector<size_t> class_idx(priors.size());
        iota(class_idx.begin(), class_idx.end(), 0);
        stable_sort(class_idx.begin(), class_idx.end(),
                    [&priors](size_t i1, size_t i2)
                    { return priors[i1] < priors[i2]; });
        cout << "B" << endl;

        int rank_of_target_class = class_idx[target_class];

        cout << "C" << endl;

        vector<int> cluster_size(num_class);
        for (int c = 0; c < num_class; c++)
        {
            cluster_size[c] = count(estimated_clusters.begin(), estimated_clusters.end(), c + 1);
        }
        cout << "D" << endl;
        vector<size_t> cluster_idx(cluster_size.size());
        iota(cluster_idx.begin(), cluster_idx.end(), 0);
        stable_sort(cluster_idx.begin(), cluster_idx.end(),
                    [&cluster_size](size_t i1, size_t i2)
                    { return cluster_size[i1] < cluster_size[i2]; });
        cout << "E" << endl;
        cout << cluster_ids.size() << " " << rank_of_target_class << endl;
        int matched_cluster_id = cluster_ids[rank_of_target_class];
        cout << "F" << endl;
        vector<int> matched_cluster_points;
        matched_cluster_points.reserve(cluster_size[matched_cluster_id]);
        for (int i = 0; i < estimated_clusters.size(); i++)
        {
            if (estimated_clusters[i] - 1 == matched_cluster_id)
            {
                matched_cluster_points.push_back(i);
            }
        }
        cout << "G" << endl;

        return matched_cluster_points;
    }

    template <typename T>
    vector<int> attack(T &clf, vector<vector<float>> &base_X)
    {
        prepare_graph<T>(clf);
        run_louvain();
        vector<vector<float>> base_X_normalized = minmax_normaliza(base_X);
        concatenate_basex_with_one_hot_encoding_of_communities_allocation(base_X_normalized);
        vector<int> estimated_clusters = run_kmeans(base_X_normalized);
        return estimated_clusters;
    }
};