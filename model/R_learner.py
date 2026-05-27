import torch
import torch.nn as nn
from model.backbone import Backbone
from engine.trainer import Trainer
from torch.utils.data import Dataset
from loss.causal_loss import MSElossWrapper, BCElossWrapper

class ResidualDataset(Dataset):
    def __init__(self, x, residual_t, residual_y):
        self.x = x.float()
        self.residual_t = residual_t.float()
        self.residual_y = residual_y.float()
        
    def __len__(self,):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.residual_t[index], self.residual_y[index]
    

class OutcomeModel(nn.Module):
    def __init__(self, x_dim, hidden_dim):
        super().__init__()
        self.backbone = Backbone(x_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = MSElossWrapper()
        
    def forward(self, x):
        return self.head(self.backbone(x))
    
    def compute_loss(self, batch):
        x, _, y, _, _ = batch
        y_pred = self.forward(x)
        return self.loss_fn(y_pred, y)
        
    @torch.no_grad()
    def predict(self, x):
        return self.head(self.backbone(x))
    
class PropensityModel(nn.Module):
    def __init__(self, x_dim, hidden_dim):
        super().__init__()
        self.backbone = Backbone(x_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = BCElossWrapper()
    
    def forward(self, x):
        return self.head(self.backbone(x))
    
    def compute_loss(self, batch):
        x, t, _, _, _ = batch
        logit = self.forward(x)
        return self.loss_fn(logit, t)
    
    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))
    
class CATERegression(nn.Module):
    def __init__(self,x_dim, hidden_dim):
        super().__init__()
        self.backbone = Backbone(x_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = MSElossWrapper()
        
    def forward(self, x):
        y_pred = self.head(self.backbone(x))
        return y_pred
    
    def compute_loss(self, batch):
        x, residual_t, residual_y = batch
        tau_hat = self.forward(x)
        return self.loss_fn(residual_y, residual_t * tau_hat)
    
    @torch.no_grad()
    def predict(self, x):
        y_pred = self.forward(x)
        return y_pred
    
    
class RLearnerEstimator:    
    def __init__(self, x_dim, hidden_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m = OutcomeModel(x_dim, hidden_dim).to(self.device)
        self.e = PropensityModel(x_dim, hidden_dim).to(self.device)
        self.tau = CATERegression(x_dim, hidden_dim).to(self.device)
        
    def fit_stage1(self, train_loader, valid_loader, epochs, lr = 1e-3):
        optimizer_m =  torch.optim.Adam(self.m.parameters(), lr = lr)
        trainer_m = Trainer(self.m, optimizer_m, self.device)
        trainer_m.run(train_loader, valid_loader, epochs)
        
        optimizer_e = torch.optim.Adam(self.e.parameters(), lr = lr)
        trainer_e = Trainer(self.e, optimizer_e, self.device)
        trainer_e.run(train_loader, valid_loader, epochs)
        
    @torch.no_grad()
    def build_residual_dataloader(self, dataloader, batch_size = 64):
        self.m.eval()
        self.e.eval()
        
        x_all = []
        residual_t_all = []
        residual_y_all = []
        for batch in dataloader:
            x, t, y, _, _ = batch
            x = x.to(self.device)
            t = t.to(self.device)
            y = y.to(self.device)

            m_hat = self.m.predict(x)
            e_hat = self.e.predict_proba(x)
            
            residual_t_all.append(t - e_hat)
            residual_y_all.append(y - m_hat)
            x_all.append(x)
        x_all = torch.cat(x_all, dim=0)
        residual_t_all = torch.cat(residual_t_all, dim=0)
        residual_y_all = torch.cat(residual_y_all, dim=0)
        resdataset = ResidualDataset(x_all, residual_t_all, residual_y_all) 
        return torch.utils.data.DataLoader(resdataset, batch_size=batch_size, shuffle= True)
            
        
    def fit_stage2(self, train_loader, valid_loader, epochs, lr = 1e-3):
        train_residual_loader = self.build_residual_dataloader(train_loader)
        val_residual_loader = self.build_residual_dataloader(valid_loader)
        optimizer_tau = torch.optim.Adam(self.tau.parameters(), lr = lr)
        trainer_tau = Trainer(self.tau, optimizer_tau, self.device)
        trainer_tau.run(train_residual_loader, val_residual_loader, epochs)
        
    def fit(self, train_loader, valid_loader, epochs, lr = 1e-3):
        self.fit_stage1(train_loader, valid_loader, epochs, lr)
        self.fit_stage2(train_loader, valid_loader, epochs, lr)
        
    @torch.no_grad()
    def predict_tau(self, x):
        self.tau.eval()
        return self.tau.predict(x)
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.tau.eval()
        tau_hat_all = []
        tau_true_all = []
        
        for batch in dataloader:
            x, t, y, mu0, mu1 = batch
            x = x.to(self.device)
            tau_hat = self.predict_tau(x).detach().cpu()
            tau_true = (mu1 - mu0)
            tau_hat_all.append(tau_hat)
            tau_true_all.append(tau_true)
        
        tau_hat_all = torch.cat(tau_hat_all, dim = 0).cpu().numpy()
        tau_true_all = torch.cat(tau_true_all, dim = 0).cpu().numpy()
        return tau_hat_all, tau_true_all


        


        