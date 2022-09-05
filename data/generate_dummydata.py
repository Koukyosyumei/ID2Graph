import numpy as np
import pandas as pd

if __name__ == "__main__":
    np.random.seed(42)
    y = np.array([0 for _ in range(10000)] + [1 for _ in range(10000)])

    x1 = y + np.random.random(y.shape) * 3
    x2 = y + np.random.random(y.shape) * 5
    x3 = y + np.random.random(y.shape) * 6
    x4 = y + np.random.random(y.shape) * 7
    x5 = y + np.random.random(y.shape) * 8
    x6 = y + np.random.random(y.shape) * 3
    x7 = y + np.random.random(y.shape) * 5
    x8 = y + np.random.random(y.shape) * 6
    x9 = y + np.random.random(y.shape) * 7
    x10 = y + np.random.random(y.shape) * 8

    x11 = y + np.random.random(y.shape) * 9
    x12 = y + np.random.random(y.shape) * 11
    x13 = y + np.random.random(y.shape) * 13
    x14 = y + np.random.random(y.shape) * 17
    x15 = y + np.random.random(y.shape) * 19
    x16 = y + np.random.random(y.shape) * 9
    x17 = y + np.random.random(y.shape) * 11
    x18 = y + np.random.random(y.shape) * 13
    x19 = y + np.random.random(y.shape) * 17
    x20 = y + np.random.random(y.shape) * 19

    df = pd.DataFrame()
    df["y"] = y

    df["x1"] = x1
    df["x2"] = x2
    df["x3"] = x3
    df["x4"] = x4
    df["x5"] = x5
    df["x6"] = x6
    df["x7"] = x7
    df["x8"] = x8
    df["x9"] = x9
    df["x10"] = x10
    df["x11"] = x11
    df["x12"] = x12
    df["x13"] = x13
    df["x14"] = x14
    df["x15"] = x15
    df["x16"] = x16
    df["x17"] = x17
    df["x18"] = x18
    df["x19"] = x19
    df["x20"] = x20

    df = df.sample(df.shape[0])
    df.to_csv("dummy.csv", index=False)
