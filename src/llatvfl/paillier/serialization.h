#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>
#include "paillier.h"
using namespace std;

namespace boost
{
    namespace serialization
    {
        template <class Archive>
        void serialize(Archive &ar, PaillierPublicKey &pk, unsigned int /* version */)
        {
            ar &make_nvp("n", pk.n);
            ar &make_nvp("n2", pk.n2);
            ar &make_nvp("g", pk.g);
            ar &make_nvp("precision", pk.precision);
        }
    }
}