g++ -o pipeline.out pipeline.cpp
python3 ./data/prep.py -d givemesomecredit -p ./data/givemesomecredit/ -s 42
./pipeline.out < data/givemesomecredit/givemesomecredit.in