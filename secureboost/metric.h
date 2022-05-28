#include <cmath>
#include <vector>
#include <string>
#include <iostream>
using namespace std;

double roc_auc_score(vector<double> y_pred, vector<int> y_true)
{
    vector<double> thresholds = {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0};
    vector<pair<double, double>> roc_points;
    for (int i = 0; i < thresholds.size(); i++)
    {
        roc_points.push_back(get_fpr_and_tpr(y_pred, y_true, thresholds[i]));
    }
    vector<double> tpr_vec;
    vector<double> fpr_vec;
    for (int i = 0; i < roc_points.size(); i++)
    {
        tpr_vec.push_back(roc_points[i].first);
        fpr_vec.push_back(roc_points[i].second);
    }
    return trapz(tpr_vec, fpr_vec);
}

double trapz(vector<double> x, vector<double> y)
{
    double res = 0;
    for (int i = 1; i < x.size(); i++)
    {
        res += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2;
    }
    return res;
}

pair<double, double> get_fpr_and_tpr(vector<double> y_pred, vector<int> y_true, double threshold)
{

    try
    {
        if (y_pred.size() != y_true.size())
        {
            string err_msg = "";
            err_msg += "the length of y_pred is ";
            err_msg += to_string(y_pred.size());
            err_msg += ", but the length of y_true is ";
            err_msg += to_string(y_true.size());
            throw invalid_argument(err_msg);
        }
    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }

    double tp, fp, tn, fn = 0;

    for (int i = 0; i < y_pred.size(); i++)
    {
        if (y_pred[i] >= threshold)
        {
            if (y_true[i] == 1)
            {
                tp += 1;
            }
            else
            {
                fp += 1;
            }
        }
        else
        {
            if (y_true[i] == 0)
            {
                tn += 1;
            }
            else
            {
                fn += 1;
            }
        }
    }

    double tpr = tp / (tp + fn);
    double fpr = fp / (tn + fp);
    return make_pair(tpr, fpr);
}