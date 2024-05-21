import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EarlyStopper:
    def __init__(self, patience, tol):
        self.patience = patience
        self.tol = tol
        self.counter = 0
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.tol):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class SuzukiDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X)  
        self.y = torch.Tensor(y)  
        self.len=len(self.X)                 

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

class FFNN_Model(torch.nn.Module):
    def __init__(self, in_features, hidden_layer_sizes, out_features):
        super().__init__()
        #model
        self.model = torch.nn.Sequential(torch.nn.Linear(in_features, hidden_layer_sizes[0]),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2]),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(hidden_layer_sizes[2], out_features))
    def forward(self, x):
        x = self.model(x)
        return x  

class FFNN():
    def __init__(self, task_type, params, random_seed, X_train, y_train, X_valid, y_valid, X_test, y_test):
        #super().__init__()
        self.task_type = task_type

        #data
        self.in_features = X_train.shape[1]
        if self.task_type == 'bin': self.out_features = 2
        elif self.task_type == 'mul': self.out_features = 4
        elif self.task_type == 'reg': self.out_features = 1
        self.batch_size = 256
        self.train_data = SuzukiDataset(X_train,y_train)
        self.val_data = SuzukiDataset(X_valid,y_valid)
        self.test_data = SuzukiDataset(X_test,y_test)
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True) 
        self.val_dataloader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True) 
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size)

        #hyperparameters
        self.hidden_layer_sizes = params['hidden_layer_sizes']
        self.learning_rate = params['learning_rate_init']
        torch.manual_seed(random_seed)

        self.sm = torch.nn.Softmax(dim=1)

        #peripherals
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       
   
    def train(self, model, optimizer, dataloader):
        epoch_loss = []
        model.train() #set model to training mode 
        
        for batch in dataloader:    
            X, y = batch
            if self.task_type != 'reg': y = y.type(torch.LongTensor)
            X = X.to(self.device)
            y = y.to(self.device)
            
            #train model on each batch 
            y_pred = model(X)
            
            if self.task_type == 'reg': loss = torch.nn.functional.mse_loss(y_pred.ravel(),y.ravel())
            else: loss = torch.nn.functional.cross_entropy(y_pred,y)
            epoch_loss.append(loss.item())
            
            # run backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return np.array(epoch_loss).mean()

    def validate(self, model, dataloader):
        val_loss = []
        model.eval() #set model to evaluation mode 
        with torch.no_grad():    
            for batch in dataloader:
                X, y = batch
                if self.task_type != 'reg': y = y.type(torch.LongTensor)
                X = X.to(self.device)
                y = y.to(self.device)
                
                #validate model on each batch 
                y_pred = model(X)

                if self.task_type == 'reg': loss = torch.nn.functional.mse_loss(y_pred.ravel(),y.ravel())
                else: loss = torch.nn.functional.cross_entropy(y_pred,y)
                val_loss.append(loss.item())      
        return np.array(val_loss).mean()
    
    def fit(self):
        self.model = FFNN_Model(self.in_features,self.hidden_layer_sizes,self.out_features).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', verbose=True, patience=3, factor=0.5)
        self.early_stopper = EarlyStopper(patience=10, tol=1e-4)
        self.train_losses = []
        self.val_losses = []

        for epoch in range(200):
            train_loss = self.train(self.model, self.optimizer, self.train_dataloader)
            val_loss = self.validate(self.model, self.val_dataloader)
            self.scheduler.step(val_loss)

            #record train and loss performance, check for early stopping
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            print(f"{epoch}     {train_loss}     {val_loss}")
            if self.early_stopper.early_stop(val_loss):
                print(f'Early stopping reached at epoch {epoch}')             
                break
    
    def predict(self):
        preds, probs = [],[]
        y_probs = None
        with torch.no_grad():
            self.model.eval()
            for batch in self.test_dataloader:
                X, y = batch

                X = X.to(self.device)
                y = y.to(self.device)

                # evaluate model
                pred = self.model(X)
                if self.task_type == 'bin': probs.append(self.sm(pred).tolist())
                if self.task_type != 'reg': pred = torch.argmax(pred,dim=1)
                preds.append(pred.tolist())

        #flatten outputs from the various batches
        y_pred = np.array([item for sublist in preds for item in sublist])
        if self.task_type == 'bin': y_probs = np.array([item for sublist in probs for item in sublist])
        return y_pred, y_probs