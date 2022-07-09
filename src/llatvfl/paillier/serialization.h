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
            ar &make_nvp("max_val", pk.max_val);
        }

        template <class Archive>
        void serialize(Archive &ar, PaillierCipherText &ct, unsigned int /* version */)
        {
            ar &make_nvp("pk", ct.pk);
            ar &make_nvp("c", ct.c);
            ar &make_nvp("exponent", ct.exponent);
            ar &make_nvp("precision", ct.precision);
        }
    }
}