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

wget -P data/diabetes https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip
unzip data/diabetes/dataset_diabetes.zip -d data/diabetes
mv data/diabetes/dataset_diabetes/diabetic_data.csv data/diabetes/diabetic_data.csv

wget -P data/fars https://www.openml.org/data/download/4965247/fars.arff

wget -P data/obesity "https://archive.ics.uci.edu/ml/machine-learning-databases/00544/ObesityDataSet_raw_and_data_sinthetic (2).zip"
unzip "data/obesity/ObesityDataSet_raw_and_data_sinthetic (2).zip" -d data/obesity
