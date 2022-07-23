#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
using namespace std;

inline float sigmoid(float x)
{
    float sigmoid_range = 34.538776394910684;
    if (x <= -1 * sigmoid_range)
        return 1e-15;
    else if (x >= sigmoid_range)
        return 1.0 - 1e-15;
    else
        return 1.0 / (1.0 + exp(-1 * x));
}

template <typename T>
inline vector<T> remove_duplicates(vector<T> &inData)
{
    vector<float> outData;
    set<float> s{};
    for (int i = 0; i < inData.size(); i++)
    {
        if (s.insert(inData[i]).second)
        {
            outData.push_back(inData[i]);
        }
    }
    return outData;
}

template <typename T>
static inline float Lerp(T v0, T v1, T t)
{
    return (1 - t) * v0 + t * v1;
}

template <typename T>
static inline std::vector<T> Quantile(const std::vector<T> &inData, const std::vector<T> &probs)
{
    if (inData.empty())
    {
        return std::vector<T>();
    }

    if (1 == inData.size())
    {
        return std::vector<T>(1, inData[0]);
    }

    std::vector<T> data = inData;
    std::sort(data.begin(), data.end());
    std::vector<T> quantiles;

    for (size_t i = 0; i < probs.size(); ++i)
    {
        T poi = Lerp<T>(-0.5, data.size() - 0.5, probs[i]);

        size_t left = std::max(int64_t(std::floor(poi)), int64_t(0));
        size_t right = std::min(int64_t(std::ceil(poi)), int64_t(data.size() - 1));

        T datLeft = data.at(left);
        T datRight = data.at(right);

        T quantile = Lerp<T>(datLeft, datRight, poi - left);

        quantiles.push_back(quantile);
    }

    return quantiles;
}

inline vector<int> get_num_parties_per_process(int n_job, int num_parties)
{
    vector<int> num_parties_per_thread(n_job, num_parties / n_job);
    for (int i = 0; i < num_parties % n_job; i++)
    {
        num_parties_per_thread[i] += 1;
    }
    return num_parties_per_thread;
}

inline bool is_satisfied_with_mi_bound_cond(vector<float> &prior, float mi_delta,
                                            float temp_left_poscnt, float temp_left_size,
                                            float temp_right_poscnt, float temp_right_size,
                                            float entire_pos_cnt, float entire_datasetsize)
{
    if (mi_delta > 0)
    {
        float left_in_pos_ratio = temp_left_poscnt / temp_left_size;
        float left_out_pos_ratio = (entire_pos_cnt - temp_left_poscnt) / (entire_datasetsize - temp_left_size);
        float left_in_diff = max(abs(left_in_pos_ratio - prior[1]),
                                 abs((1 - left_in_pos_ratio) - prior[0]));
        float left_out_diff = max(abs(left_out_pos_ratio - prior[1]),
                                  abs((1 - left_out_pos_ratio) - prior[0]));

        float right_in_pos_ratio = temp_right_poscnt / temp_right_size;
        float right_out_pos_ratio = (entire_pos_cnt - temp_right_poscnt) / (entire_datasetsize - temp_right_size);
        float right_in_diff = max(abs(right_in_pos_ratio - prior[1]),
                                  abs((1 - right_in_pos_ratio) - prior[0]));
        float right_out_diff = max(abs(right_out_pos_ratio - prior[1]),
                                   abs((1 - right_out_pos_ratio) - prior[0]));

        return ((left_in_diff > mi_delta) | (left_out_diff > mi_delta) |
                (right_in_diff > mi_delta) | (right_out_diff > mi_delta));
    }
    else
    {
        return false;
    }
}