#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <cassert>
#include "secureboost/metric.h"
using namespace std;

vector<int> y_true = {0, 1, 0, 1, 0};
vector<double> y_pred = {0.592837, 0.624829, 0.073848, 0.544891, 0.015118};

pair<double, double> fpr_and_tpr = get_fpr_and_tpr(y_pred, y_true, 0.5);
// assert(fpr_and_tpr.first == 0.76);
// assert(fpr_and_tpr.second = 0.82);
