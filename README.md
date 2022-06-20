# SecureBoost

```
./script/run.sh -d breastcancer -r 20 -c 1 -i 1 -e 0.3
```

## Usage

```
    -d : (str) the name of dataset
    -m : (str) type of the model
    -r : (int) total number of rounds for training
    -c : (int) the number of completely secure rounds
    -h : (int) depth
    -j : (int) the number of jobs
    -n : (int) the number of data records sampled for training
    -f : (float) the ratio of features owned by the active party
    -i : (float) the imlalance of dataset
    -e : (float) coefficient of edge weight (tau in our paper)
    -t : (str) the path to the folder where this script saves all results (default=`result`)
    -k : (str) type of clustering method (`vanila` = K-Means, `reduced` = Reduced K-Means)
    -w : (optional) includes intermidiate nodes for construction of adj_mat (default=false)
    -g : (optional) draw the extracted graph (default=false)
```


## Overview of datasets

- Give Me Some Credit

|label   |count   |
|---|---|
|0  |139974|
|1  |10026|

- UCI Credit Card

|label   |count   |
|---|---|
|0  |23364|
|1  |6636|


## Idea

