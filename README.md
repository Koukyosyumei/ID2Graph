# SecureBoost

```
./script/run.sh -d ucicreditcard -m r -r 2 -c 0 -h 3 -i 1 -e 1 -n -1 -f 0.5 -p 1 -z 300
```

```
./mpiscript/run.sh -d givemesomecredit -m s -r 2 -c 0 -h 3 -i 1 -e 1 -n 5000 -f 0.5 -p 1 -z 300
```

## Usage

```
    -d : (str) the name of dataset
    -m : (str) type of the model
    -r : (int) total number of rounds for training
    -c : (int) the number of completely secure rounds
    -a : (float) learning rate of XGBoost
    -h : (int) depth
    -j : (int) the number of jobs
    -n : (int) the number of data records sampled for training
    -f : (float) the ratio of features owned by the active party
    -v : (float) the ratio of features owned by the passive party
    -i : (float) the imlalance of dataset
    -e : (float) coefficient of edge weight (tau in our paper)
    -l : (float) epsilon of epsilon-greedy louvain
    -o : (float) epsilon of LP-MST
    -b : (float) bound of mutual information
    -z : wait for timeout of louvain
    -t : (str) the path to the folder where this script saves all results (default=`result`)
    -k : (str) type of clustering method (`vanila` = K-Means, `reduced` = Reduced K-Means)
    -p : (int) number of parallelly executed experiments
    -w : (int) include from this node
    -x : (int) M of LP-MST
    -g : (optional) draw the extracted graph (default=false)
```


## Overview of datasets

|glass|heartdisease|waveform|
|---|---|---|
|6|5|3|

|label|Give Me Some Credit|UCI Credit Card|ionospere|parkinsons|sonar|spambase|
|---|---   | --- | --- | --- | --- | --- |
|0  |139974|23364| 126 |48   | 111 | 2788|
|1  |10026 |6636 | 225 |147  | 97  | 1813|


## Idea

