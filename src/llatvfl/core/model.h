#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
using namespace std;

template <typename PartyName>
struct TreeModelBase
{
    TreeModelBase(){};
    virtual void fit(vector<PartyName> &parties, vector<double> &y) = 0;
    virtual vector<double> predict_raw(vector<vector<double>> &X) = 0;
};