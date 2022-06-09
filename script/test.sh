g++ -o test.out test/test_secureboost.cpp
./test.out < test/data/test_data.in
g++ -o test.out test/test_metric.cpp
./test.out
g++ -o test.out test/test_louvain.cpp
./test.out
g++ -o test.out test/test_utils.cpp
./test.out