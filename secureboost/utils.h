#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
using namespace std;

double sigmoid(double x)
{
    double sigmoid_range = 34.538776394910684;
    if (x <= -1 * sigmoid_range)
        return 1e-15;
    else if (x >= sigmoid_range)
        return 1.0 - 1e-15;
    else
        return 1.0 / (1.0 + exp(-1 * x));
}

template <typename T>
vector<T> remove_duplicates(vector<T> &inData)
{
    vector<double> outData;
    set<double> s{};
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
static inline double Lerp(T v0, T v1, T t)
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
    data.erase(std::remove_if(std::begin(data),
                              std::end(data),
                              [](const auto &value)
                              { return std::isnan(value); }),
               std::end(data));
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
