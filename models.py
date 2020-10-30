import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


Model1_PARAMS = {
    "EPOCHS": 20,
    "BATCH_SIZE": 256,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-5
    }

class Model1(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model1, self).__init__()
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.valid_fn = nn.BCEWithLogitsLoss()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.5)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x
    
    
    
    
Model2_PARAMS = {
    "EPOCHS": 20,
    "BATCH_SIZE": 512,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-5
    }

class Model2(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model2, self).__init__()
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.valid_fn = nn.BCEWithLogitsLoss()
        
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.6)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, 1028))
        
        self.batch_norm2 = nn.BatchNorm1d(1028)
        self.dropout2 = nn.Dropout(0.6)
        self.dense2 = nn.utils.weight_norm(nn.Linear(1028, 512))
        
        self.batch_norm3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.6)
        self.dense3 = nn.utils.weight_norm(nn.Linear(512, 256))
        
        self.batch_norm4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.6)
        self.dense4 = nn.utils.weight_norm(nn.Linear(256, 256))
        
        self.batch_norm5 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.6)
        self.dense5 = nn.utils.weight_norm(nn.Linear(256, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.relu(self.dense3(x))
        
        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.relu(self.dense4(x))
        
        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = self.dense5(x)
        
        return x    