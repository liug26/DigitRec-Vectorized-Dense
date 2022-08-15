import pandas as pd
import numpy as np


def read():
    df = pd.read_csv("train.csv")
    x = df.to_numpy()
    # 1st index is row, 2nd is col
    y = x[:, 0:1]
    x = x[:, 1:]
    x = x / 255
    y2 = np.zeros((10, x.shape[0]))
    y2[y.T.flatten(), range(x.shape[0])] = 1.
    return x.T, y.T, y2
