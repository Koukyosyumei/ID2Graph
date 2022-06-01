g++ -o pipeline.out pipeline.cpp
for i in $(seq 1 10)
do 
    echo "random seed is $i"
    python3 ./data/prep.py -d ucicreditcard -p ./data/ucicreditcard/ -s $i
    ./pipeline.out < data/ucicreditcard/ucicreditcard.in
done