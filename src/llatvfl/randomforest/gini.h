#pragma once
#include <cmath>
using namespace std;

float inline calc_giniimp(float tot_cnt, float pos_cnt)
{
    float neg_cnt = tot_cnt - pos_cnt;
    float pos_ratio = pos_cnt / tot_cnt;
    float neg_ratio = neg_cnt / tot_cnt;
    return 1 - (pos_ratio * pos_ratio) - (neg_ratio * neg_ratio);
}