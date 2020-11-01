import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_target_count():
    train_targets = pd.read_csv('./kaggle/input/lish-moa/train_targets_scored.csv')
    numpy_targets = train_targets.to_numpy()
    rows_reduces = list(numpy_targets[:, 1:].sum(axis = 1))
    targets = {}
    for i in rows_reduces:
        fetched = targets.get(i, 0)
        if fetched == 0:
            targets[i] = 1
        else:
            targets[i] += 1
    keys, values = zip(*sorted(targets.items()))
    plt.bar(keys, values)
    
#def removeNoneTops(top_k = 5):
    
    

def filter_preds(predictions):
    top_indices = (-predictions).argsort(axis = 1)[:, :5]
    zeros = np.zeros(predictions.shape)
    for i in range(zeros.shape[0]):
        zeros[i, top_indices[i,:]] = predictions[i, top_indices[i,:]]
    return zeros
