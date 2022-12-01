# "Do not share instance space": Label Leakage from Vertical Federated Learning

This repository contains the implementation of *"Do not share instance space": Label Leakage from Vertical Federated Learning* and all datasets used in this paper.

## 1. Dependencies

- C++11 compatible compiler
- Boost 1.65 or later
- Python 3.7 or later

## 2. Usage

### 2.1. Build from source

```
pip install -e .
./script/build.sh
```

### 2.2. Run experiments

You can run all experiments conducted in the paper with [`/script/run.sh`](./script/run.sh).

- Example

```
./script/run.sh -d ucicreditcard -m r -r 2 -c 0 -h 3 -i 1 -e 1 -n -1 -f 0.5 -p 1 -z 300
```

- Arguments

```
    -d : (str) the name of dataset.
    -m : (str) type of the model.
    -r : (int) total number of rounds for training.
    -c : (int) the number of completely secure rounds.
    -a : (float) learning rate of XGBoost.
    -h : (int) depth.
    -j : (int) the number of jobs.
    -n : (int) the number of data records sampled for training.
    -f : (float) the ratio of features owned by the active party.
    -v : (float) the ratio of features owned by the passive party. if v=-1, the ratio of local features will be 1 - f.
    -i : (float) the imlalance of dataset.
    -e : (float) coefficient of edge weight (tau in our paper).
    -k : (float) weight for community variables.
    -u : (str) the path to the folder for saving the final results.
    -t : (str) the path to the folder where this script saves temporary results (default=`result`).
    -p : (int) number of parallelly executed experiments.
    -l : (float) epsilon of epsilon-greedy louvain.
    -z : patience for timeout of louvain.
    -o : (float) epsilon of LP-MST.
    -b : (float) bound of mutual information.
    -w : (int) include from this node.
    -x : (int) M of LP-MST.
    -y : (optional) steal the exact label values.
    -g : (optional) draw the extracted graph.
    -q : (optional) draw trees as html files.
```

### Google colab

You can run all of our experiments on Google colab. Please use a runtime with high memory.