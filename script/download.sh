wget -P data/avila http://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip
unzip data/avila/avila.zip -d data/avila
rm data/avila/avila.zip
mv data/avila/avila/avila-tr.txt data/avila/avila-tr.txt
mv data/avila/avila/avila-ts.txt data/avila/avila-ts.txt
rm -rf data/avila/avila

wget -P data/drive http://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt

wget -P data/nursery http://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data

wget -P data/phishing "http://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training Dataset.arff"
mv "data/phishing/Training Dataset.arff" "data/phishing/phishing.data"
sed -i '1,36d' data/phishing/phishing.data

wget -P data/dota2 https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip
unzip data/dota2/dota2Dataset.zip -d data/dota2

wget -P data/bank https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
unzip data/bank/bank.zip -d data/bank

wget -P data/sepsis https://archive.ics.uci.edu/ml/machine-learning-databases/00628/s41598-020-73558-3_sepsis_survival_dataset.zip
unzip data/sepsis/s41598-020-73558-3_sepsis_survival_dataset.zip -d data/sepsis

wget -P data/diabetes https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip
unzip data/diabetes/dataset_diabetes.zip -d data/diabetes
mv data/diabetes/dataset_diabetes/diabetic_data.csv data/diabetes/diabetic_data.csv

wget -P data/indoor https://archive.ics.uci.edu/ml/machine-learning-databases/00377/Ipin2016Dataset.zip
unzip data/indoor/Ipin2016Dataset.zip -d data/indoor

wget -P data/brich1 https://cs.joensuu.fi/sipu/datasets/birch1.txt
wget -P data/brich1 https://cs.joensuu.fi/sipu/datasets/b1-gt.pa
