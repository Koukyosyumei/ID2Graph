g++ -o test.out test/test_secureboost.cpp
./test.out < data/test_data.in
g++ -o test.out test/test_metric.cpp
./test.out
g++ -o test.out test/test_utils.cpp
./test.out