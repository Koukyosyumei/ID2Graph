#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <cassert>
#include "../src/llatvfl/secureboost/metric.h"
using namespace std;

int main()
{
    vector<int> y_true = {0, 0, 0, 0, 1, 1, 1, 1};
    vector<double> y_pred = {0.2, 0.3, 0.6, 0.8, 0.4, 0.5, 0.7, 0.9};

    /*
    pair<double, double> fpr_and_tpr = get_fpr_and_tpr(y_pred, y_true, 0.2);
    assert(fpr_and_tpr.first == 1 && fpr_and_tpr.second == 1);
    fpr_and_tpr = get_fpr_and_tpr(y_pred, y_true, 0.4);
    assert(fpr_and_tpr.first == 0.5 && fpr_and_tpr.second == 1);
    fpr_and_tpr = get_fpr_and_tpr(y_pred, y_true, 0.7);
    assert(fpr_and_tpr.first == 0.25 && fpr_and_tpr.second == 0.5);
    fpr_and_tpr = get_fpr_and_tpr(y_pred, y_true, 0.9);
    assert(fpr_and_tpr.first == 0 && fpr_and_tpr.second == 0.25);
    */
    assert(roc_auc_score(y_pred, y_true) == 0.6875);

    cout << "test_metric: all passed!" << endl;
}
