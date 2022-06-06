#include <iostream>
#include <limits>
#include <vector>
using namespace std;

struct Graph
{
    vector<int> actual_nodes;
    unsigned long num_nodes;
    unsigned long num_links;
    double total_weight;

    // cumulative degree for each node, deg(0) = degrees[0]
    // deg(k) = degrees[k] - degrees[k-1]
    vector<unsigned long> degrees;
    vector<vector<int>> nodes;
    vector<unsigned int> links;
    vector<float> weights; // TODO check if `double` works or not

    Graph();
    Graph(vector<vector<int>> &c_nodes);
    Graph(int nb_nodes, int nb_links, double total_weight,
          int *degrees, int *links, float *weights);

    // return pointers to the first neighbor and weight of the node
    pair<vector<unsigned int>::iterator, vector<float>::iterator> get_neighbors(unsigned int node)
    {
        if (node == 0)
        {
            return make_pair(links.begin(), weights.begin());
        }
        else if (weights.size() != 0)
        {
            return make_pair(links.begin() + degrees[node - 1], weights.begin() + degrees[node - 1]);
        }
        else
        {
            return make_pair(links.begin() + degrees[node - 1], weights.begin());
        }
    }

    // return the number of neighbors of the node
    unsigned int get_num_neighbors(unsigned int node)
    {
        if (node == 0)
        {
            return degrees[0];
        }
        else
        {
            return degrees[node] - degrees[node - 1];
        }
    }

    // return the number or the weight of self loops of the node
    double get_num_selfloops(unsigned int node)
    {
        pair<vector<unsigned int>::iterator, vector<float>::iterator> p = get_neighbors(node);
        for (unsigned int i = 0; i < get_num_neighbors(node); i++)
        {
            if (*(p.first + i) == node)
            {
                if (weights.size() != 0)
                {
                    return (double)*(p.second + i);
                }
                else
                {
                    return 1;
                }
            }
        }
        return 0.;
    }

    // retrn the weighed degree of the node
    double weighted_degree(unsigned int node)
    {
        if (weights.size() == 0)
        {
            return (double)get_num_neighbors(node);
        }
        else
        {
            pair<vector<unsigned int>::iterator, vector<float>::iterator> p = get_neighbors(node);
            double res = 0;
            for (unsigned int i = 0; i < get_num_neighbors(node); i++)
            {
                res += (double)*(p.second + i);
            }
            return res;
        }
    }
};