import numpy as np
import random
import pandas as pd
import os
import torch
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from models import Model1 as Model
from models import Model1_PARAMS as PARAMS
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

#Check if GPU or CPU
def checkDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def refineResults(X_train, y_train, X_test):
    clf = MultiOutputClassifier(RandomForestClassifier(n_estimators = 5), n_jobs=-1).fit(X, y)
    return clf.predict(X_test)
    


#Dataset class
class TrainDataset:
    def __init__(self, X, y):
        self.X, self.y = X, y
        
    def __len__(self):
        return (self.X.shape[0])
    
    def __getitem__(self, idx):
        return {
                    "x": torch.tensor(self.X.iloc[idx], dtype=torch.float),
                    "y": torch.tensor(self.y.iloc[idx], dtype=torch.float) 
                }
#Dataset class    
class TestDataset:
    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return (self.X.shape[0])
    
    def __getitem__(self, idx):
        return {"x": torch.tensor(self.X.iloc[idx], dtype=torch.float)}
    
    
#Train for one epoch    
def train_epoch(model, optimizer, scheduler, loss_fn, dataloader):
    model.train()
    final_loss = 0
    device = checkDevice()
    
    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
        
    final_loss /= len(dataloader)
    return final_loss

#Compute validation loss
def valid_fn(model, loss_fn, dataloader):
    device = checkDevice()
    model.eval()
    final_loss = 0
    valid_preds = []
    
    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())
        
    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)
    
    return final_loss, valid_preds

#Predict 
def inference_fn(model, dataloader, device):
     model.eval()
     preds = []
     for data in dataloader:
        inputs = data['x'].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        preds.append(outputs.sigmoid().detach().cpu().numpy())
     return np.concatenate(preds)
 

    
    
def train(X_train, y_train,
          batch_size, num_features, num_targets,
          learning_rate, weight_decay, hidden_size,
          epochs, fold_id,  X_valid = None, y_valid = None):
    
    device = checkDevice()
    train_dataset = TrainDataset(X_train, y_train)
    if (( X_valid is not None) and (y_valid is not None)):
        valid_dataset = TrainDataset(X_valid, y_valid)
    
    trainloader = torch.utils.data.DataLoader(train_dataset,
                          batch_size=batch_size, shuffle=True)
    
    if (( X_valid is not None) and (y_valid is not None)):
        validloader = torch.utils.data.DataLoader(valid_dataset,
                              batch_size=batch_size, shuffle=False)
    
    model = Model(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
             lr=learning_rate, weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
                      pct_start=0.1, div_factor=1e3, 
                      max_lr=1e-2, epochs=epochs, 
                      steps_per_epoch=len(trainloader))
    
    loss_fn = model.loss_fn
    
    for epoch in range(epochs):
        start = time.time()
        train_loss = train_epoch(model, optimizer,scheduler, loss_fn, trainloader)
        end = time.time()
        print("Epoch: {}  |  Train Loss: {:.5f}  |  Time Elapsed: {:.2f}  |"
              .format(epoch, train_loss, end - start))
        if (( X_valid is not None) and (y_valid is not None)):
            valid_loss, valid_preds = valid_fn(model, model.valid_fn, validloader)
            print("Epoch: {}  |  Test Score: {:.5f}  |".format(epoch, valid_loss))
                
    if (( X_valid is None) or (y_valid is None)):
         torch.save(model.state_dict(), f"./kaggle/working/AllDataset.pth")
    return model


def predict(model, X_test, batch_size):
    testdataset = TestDataset(X_test)
    testloader = torch.utils.data.DataLoader(testdataset,
                     batch_size=batch_size, shuffle=False)
    
    predictions = inference_fn(model, testloader, checkDevice())
    return predictions


def getKthFold(X, y, number_of_folds, current_fold):
    kf = KFold(n_splits=number_of_folds)
    split_indices = kf.split(range(X.shape[0]))
    train_indices, test_indices = [(list(train), list(test)) for train, test in split_indices][current_fold]
    return X.iloc[train_indices], y.iloc[train_indices], X.iloc[test_indices], y.iloc[test_indices]


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
    
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


train_features = pd.read_csv('./kaggle/input/lish-moa/train_features.csv')
train_targets = pd.read_csv('./kaggle/input/lish-moa/train_targets_scored.csv')
test_features = pd.read_csv('./kaggle/input/lish-moa/test_features.csv')
ss = pd.read_csv('./kaggle/input/lish-moa/sample_submission.csv')
ss.loc[:,:] = 0.0


EPOCHS = PARAMS["EPOCHS"]
BATCH_SIZE = PARAMS["BATCH_SIZE"]
LEARNING_RATE = PARAMS["LEARNING_RATE"]
WEIGHT_DECAY = PARAMS["WEIGHT_DECAY"]
NFOLDS = 5

X, y = preprocess_data(train_features, train_targets)

#Cross validation
for i in range(NFOLDS):
    FOLD_ID = i
    X_train, y_train, X_valid, y_valid = getKthFold(X, y, NFOLDS, FOLD_ID)
    
    num_features = X_train.shape[1]
    num_targets = y_train.shape[1]
    hidden_size=1024
    
    seed_everything(42)
    print("************ FOLD {} ************".format(FOLD_ID))
    model = train(X_train, y_train, 
                 BATCH_SIZE, num_features, 
                 num_targets, LEARNING_RATE, 
                 WEIGHT_DECAY, hidden_size, 
                 EPOCHS, FOLD_ID, X_valid, y_valid)

#Train on train+valid and predict test --> write to submission.csv   
model = train(X, y, 
                 BATCH_SIZE, num_features, 
                 num_targets, LEARNING_RATE, 
                 WEIGHT_DECAY, hidden_size, 
                 EPOCHS, FOLD_ID)
    
X_test = preprocess_data(test_features)
preds = predict(model, X_test, BATCH_SIZE)
preds_refined = refineResults(X, y, X_test)
ss.iloc[:, 0] =  test_features.iloc[:,0]
ss.iloc[:, 1:] = preds_refined
ss.to_csv('./kaggle/working/submission.csv', index=False, float_format='%15f')
