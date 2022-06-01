g++ -o pipeline.out pipeline.cpp
python3 ./data/prep.py -d givemesomecredit -p ./data/givemesomecredit/ -s 42
./pipeline.out < data/givemesomecredit/givemesomecredit.in
python3 ./data/prep.py -d givemesomecredit -p ./data/givemesomecredit/ -s 43
./pipeline.out < data/givemesomecredit/givemesomecredit.in
python3 ./data/prep.py -d givemesomecredit -p ./data/givemesomecredit/ -s 44
./pipeline.out < data/givemesomecredit/givemesomecredit.in
python3 ./data/prep.py -d givemesomecredit -p ./data/givemesomecredit/ -s 45
./pipeline.out < data/givemesomecredit/givemesomecredit.in
python3 ./data/prep.py -d givemesomecredit -p ./data/givemesomecredit/ -s 46
./pipeline.out < data/givemesomecredit/givemesomecredit.in
python3 ./data/prep.py -d givemesomecredit -p ./data/givemesomecredit/ -s 47
./pipeline.out < data/givemesomecredit/givemesomecredit.in
python3 ./data/prep.py -d givemesomecredit -p ./data/givemesomecredit/ -s 48
./pipeline.out < data/givemesomecredit/givemesomecredit.in