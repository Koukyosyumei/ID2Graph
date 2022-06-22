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
    virtual void fit(vector<PartyName> &parties, vector<float> &y) = 0;
    virtual vector<float> predict_raw(vector<vector<float>> &X) = 0;
};