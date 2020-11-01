from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np 
from sklearn.model_selection import KFold
import torch.nn as nn
NFOLDS = 5

def preprocess_data(X, y = None):
    GENES = [col for col in X.columns if col.startswith('g-')]
    CELLS = [col for col in X.columns if col.startswith('c-')]
    #One hot encoding      
    X = pd.get_dummies(X,
                      columns=["cp_type", "cp_time", "cp_dose"])   
    
    #Z-score scaling
    for col in GENES:
         X[col] = (X[col]-np.mean(X[col])) / (np.std(X[col]))
    for col in CELLS:
        X[col] = (X[col]-np.mean(X[col])) / (np.std(X[col]))
        
    if y is not None: 
        return X.drop(["sig_id"], axis=1), y.drop(["sig_id"], axis=1)
    else:
        return X.drop(["sig_id"], axis=1)
    
def getKthFold(X, y, number_of_folds, current_fold):
    kf = KFold(n_splits=number_of_folds)
    split_indices = kf.split(range(X.shape[0]))
    train_indices, test_indices = [(list(train), list(test)) for train, test in split_indices][current_fold]
    return X.iloc[train_indices], y.iloc[train_indices], X.iloc[test_indices], y.iloc[test_indices]

train_features = pd.read_csv('./kaggle/input/lish-moa/train_features.csv')
train_targets = pd.read_csv('./kaggle/input/lish-moa/train_targets_scored.csv')
test_features = pd.read_csv('./kaggle/input/lish-moa/test_features.csv')
ss = pd.read_csv('./kaggle/input/lish-moa/sample_submission.csv')
ss.loc[:,:] = 0.0

X, y = preprocess_data(train_features, train_targets)

eval_fn = loss_fn = nn.BCELoss()
#Cross validation
for i in range(NFOLDS):
    FOLD_ID = i
    X_train, y_train, X_valid, y_valid = getKthFold(X, y, NFOLDS, FOLD_ID)
    clf = MultiOutputClassifier(
    RandomForestClassifier(n_estimators = 20), n_jobs=-1).fit(X_train, y_train)
    preds = clf.predict(X_valid)
    print(eval_fn(torch.tensor(preds, dtype = torch.float32),
                  torch.tensor(y_valid.to_numpy(), dtype = torch.float32)))
