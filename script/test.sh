g++ -pthread -o script/build/test.out test/test_randomforest.cpp
./script/build/test.out < test/data/test_data.in
g++ -pthread -o script/build/test.out test/test_secureboost.cpp
./script/build/test.out < test/data/test_data.in
g++ -o script/build/test.out test/test_metric.cpp
./script/build/test.out
g++ -o script/build/test.out test/test_louvain.cpp
./script/build/test.out
g++ -o script/build/test.out test/test_utils.cpp
./script/build/test.out