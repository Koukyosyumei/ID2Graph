#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <set>
#include <numeric>
#include <array>
using namespace std;

double trapz(vector<double> x, vector<double> y)
{
    double res = 0;
    for (int i = 1; i < x.size(); i++)
    {
        res += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2;
    }
    return res;
}

vector<double> get_thresholds_idxs(vector<double> y_pred)
{
    vector<double> thresholds_idxs;
    set<double> s{};
    reverse(y_pred.begin(), y_pred.end());
    int y_pred_size = y_pred.size();
    for (int i = 0; i < y_pred_size; i++)
    {
        if (s.insert(y_pred[i]).second)
        {
            thresholds_idxs.push_back(y_pred_size - i);
        }
    }
    return thresholds_idxs;
}

double roc_auc_score(vector<double> y_pred, vector<int> y_true)
{
    vector<int> temp_idxs(y_pred.size());
    iota(temp_idxs.begin(), temp_idxs.end(), 0);
    sort(temp_idxs.begin(), temp_idxs.end(), [&y_pred](size_t i, size_t j)
         { return y_pred[i] < y_pred[j]; });
    vector<int> temp_y_true(y_true.size());
    copy(y_true.begin(), y_true.end(), temp_y_true.begin());
    for (int i = 0; i < y_pred.size(); i++)
    {
        y_true[i] = temp_y_true[temp_idxs[i]];
    }
    sort(y_pred.begin(), y_pred.end());

    vector<double> thresholds_idxs = get_thresholds_idxs(y_pred);

    vector<double> tps = {0};
    for (int i = 1; i < y_true.size(); i++)
    {
        tps.push_back(y_true[i] + tps[i - 1]);
    }
    for (int i = 0; i < tps.size(); i++)
    {
        tps[i] = tps[i] / tps[tps.size() - 1];
    }

    vector<double> fps = {1};
    for (int i = 1; i < y_true.size(); i++)
    {
        fps.push_back(1 - y_true[i] + fps[i - 1]);
    }
    for (int i = 0; i < fps.size(); i++)
    {
        fps[i] = fps[i] / fps[fps.size() - 1];
    }

    return trapz(tps, fps);
}