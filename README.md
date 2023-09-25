# Eliminating Label Leakage in Tree-based Vertical Federated Learning

## 1. Dependencies

- C++11 compatible compiler
- Boost 1.65 or later
- Python 3.7 or later

We use the following version for the experiments.

```
- Python 3.10.11
- g++ (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0
- Boost 1.71
```

## 2. Usage

### 2.1. Build from source

```shell
pip install -e .
./script/build.sh
```

### 2.2. Download datasets

```shell
./script/download.sh
```

### 2.3. Run experiments

- Example

```shell
for VAL_D in breastcancer parkinson phishing obesity avila
do
  sudo ./script/run.sh -d ${VAL_D} -m r -r 5 -a 1.0 -c 0 -h 6 -e 1.0 -f 0.5 -n -1 -p 5 -z 5 -u ${OUTPUT_DIR} -t result/temp
  sudo ./script/run.sh -d ${VAL_D} -m x -r 5 -a 0.3 -c 0 -h 6 -e 0.6 -f 0.5 -n -1 -p 5 -z 5 -u ${OUTPUT_DIR} -t result/temp
done

for VAL_D in drive fars pucrio fmnist
do
  sudo ./script/run.sh -d ${VAL_D} -m r -r 5 -a 1.0 -c 0 -h 6 -e 1.0 -f 0.5 -n -1 -p 1 -z 5 -w 1000 -y 100 -u ${OUTPUT_DIR} -t result/temp
  sudo ./script/run.sh -d ${VAL_D} -m x -r 5 -a 0.3 -c 0 -h 6 -e 0.6 -f 0.5 -n -1 -p 1 -z 5 -w 1000 -y 100 -u ${OUTPUT_DIR} -t result/temp
done
```

- Basic Arguments

```
    -u : (str) path to the folder to save the final results.
    -d : (str) name of dataset.
    -m : (str) type of training algorithm. `r`: Random Forest, `x`: XGBoost, `g`: GraftingForest
```

- Advanced Arguments

```
    -t : (str) path to the folder to save the temporary results.
    -z : (int) number of trials.
    -p : (int) number of parallelly executed experiments.

    -n : (int) number of data records sampled for training.
    -f : (float) ratio of features owned by the active party.
    -i : (int) setting of feature importance. -1: normal, 1: unbalance

    -r : (int) total number of rounds for training.
    -j : (int) minimum number of samples within a leaf.
    -h : (int) maximum depth.
    -a : (float) learning rate of XGBoost.

    -e : (float) coefficient of edge weight (tau in our paper).
    -k : (float) weight for community variables.
    -v : (str) clustering type. `kmeans` or `xmeans`
    -l : (int) maximum number of iterations of Louvain
    -x : (optional) baseline union attack
    -w : chunk size for memory efficient adjacency matrix
    -y : intra-chunk edge weight for memory efficient adjacency matrix

    -b : (float) epsilon of ID-LMID.
    -c : (int) number of completely secure rounds.
    -o : (float) epsilon of LP-MST.

    -g : (optional) draw the extracted clusters.
    -q : (optional) draw trees as html files.
```

### 3. Note

The above simulations utilize computation on the plaintext. Experiments using Paiilier Encryption can also be performed with `-m sr` and `-m sx`, but they are time-consuming. The results are identical to those obtained with `-m r` and `-m x`, respectively. We also provide MPI backend as a reference implementation in `mpiscipt` folder, but it does not support some features.
