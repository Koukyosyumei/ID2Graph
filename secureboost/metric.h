#include <cmath>
#include <vector>
#include <string>
#include <iostream>
using namespace std;

double roc_auc_score(vector<double> y_pred, vector<int> y_true)
{
}

double trapz(vector<double> x, vector<double> y)
{
    double res = 0;
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