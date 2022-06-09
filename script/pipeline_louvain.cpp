#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <numeric>
#include <cmath>
#include <cassert>
#include "louvain/louvain.h"
using namespace std;

const float eta = 0.4;

int main()
{
    int round_num, node_num, temp_adj_num, temp_adj_idx;
    float temp_adj_weight;
    scanf("%d %d", &round_num, &node_num);
    vector<vector<float>> adj_matrix(node_num, vector<float>(node_num, 0));
    for (int i = 0; i < round_num; i++)
    {
        if (i >= 0)
        {
            for (int j = 0; j < node_num; j++)
            {
                scanf("%d", &temp_adj_num);
                for (int k = 0; k < temp_adj_num; k++)
                {
                    scanf("%d %f", &temp_adj_idx, &temp_adj_weight);
                    adj_matrix[j][temp_adj_idx] += pow(eta, float(i)) * temp_adj_weight;
                    adj_matrix[temp_adj_idx][j] += pow(eta, float(i)) * temp_adj_weight;
                }
            }
        }
    }

    vector<unsigned long> degrees;
    vector<unsigned int> links;
    vector<float> weights;

    int cum_degree = 0;
    for (int i = 0; i < node_num; i++)
    {
        for (int j = 0; j < node_num; j++)
        {
            if (adj_matrix[i][j] != 0)
            {
                cum_degree += 1;
                links.push_back(j);
                weights.push_back(adj_matrix[i][j]);
            }
        }
        degrees.push_back(cum_degree);
    }

    adj_matrix.clear();
    Graph g = Graph(node_num, degrees, links, weights);

    Louvain louvain = Louvain();
    louvain.fit(g);

    printf("%lu\n", louvain.g.nodes.size());
    printf("%d\n", node_num);
    for (int i = 0; i < louvain.g.nodes.size(); i++)
    {
        for (int j = 0; j < louvain.g.nodes[i].size(); j++)
        {
            printf("%d ", louvain.g.nodes[i][j]);
        }
        printf("\n");
    }
}