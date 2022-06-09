#include <vector>
#include <cassert>
#include <iostream>
#include "../src/secureboost/utils.h"

int main()
{
    vector<double> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    vector<double> quartiles = Quantile<double>(in, {0.25, 0.5, 0.75});
    vector<double> test_quartiles = {3.25, 6, 8.75};
    for (int i = 0; i < quartiles.size(); i++)
    {
        assert(quartiles[i] == test_quartiles[i]);
    }
    cout << "test_utils: all passed!" << endl;
}