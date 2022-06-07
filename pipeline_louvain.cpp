#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <numeric>
#include <cassert>
#include "louvain/louvain.h"
using namespace std;

int main()
{
    int round_num, node_num, adj_num, temp_adj_idx;
    cin >> round_num >> node_num;
    vector<vector<float>> adj_matrix(node_num, vector<float>(node_num, 0));
    for (int i = 0; i < round_num; i++)
    {
        if (i < 1)
        {
            for (int j = 0; j < node_num; j++)
            {
                cin >> adj_num;
                for (int k = 0; k < adj_num; k++)
                {
                    cin >> temp_adj_idx;
                    adj_matrix[j][temp_adj_idx] += 1;
                    adj_matrix[temp_adj_idx][j] += 1;
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

    cout << louvain.g.nodes.size() << endl;
    cout << node_num << endl;
    for (int i = 0; i < louvain.g.nodes.size(); i++)
    {
        for (int j = 0; j < louvain.g.nodes[i].size(); j++)
        {
            cout << louvain.g.nodes[i][j] << " ";
        }
        cout << endl;
    }
}