g++ -O3 -mtune=native -march=native -o script/build/test_utils.out test/test_utils.cpp
./script/build/test_utils.out
g++ -O3 -mtune=native -march=native -o script/build/test_metric.out test/test_metric.cpp
./script/build/test_metric.out
g++ -O3 -mtune=native -march=native -o script/build/test_dok.out test/test_dok.cpp
./script/build/test_dok.out
g++ -O3 -mtune=native -march=native -pthread -o script/build/test_randomforest.out test/test_randomforest.cpp
./script/build/test_randomforest.out < test/data/test_data.in
g++ -O3 -mtune=native -march=native -pthread -o script/build/test_xgboost.out test/test_xgboost.cpp
./script/build/test_xgboost.out < test/data/test_data.in
g++ -O3 -mtune=native -march=native -o script/build/test_louvain.out test/test_louvain.cpp
./script/build/test_louvain.out

rm script/build/test*