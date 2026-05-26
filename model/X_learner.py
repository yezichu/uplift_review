import torch
import torch.nn as nn
from model.backbone import Backbone
from engine.trainer import Trainer
from loss.causal_loss import MSElossWapper, BCElossWapper

"""
    相比 T-learner，X-learner 在 treatment/control 样本严重不均衡时通常效果更好，
    因为它能 cross-impute treatment effect。最后用 e(x)加权，如果 e(x) 越高，那么 mu0
    估的越好，也就是 tau0 估计得更准。
"""

class OutcomeModel(nn.Module):
    def __init__(self,x_dim, hidden_dim, type):
        super().__init__()
        self.backbone = Backbone(x_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = MSElossWapper()
        self.type = type
        
    def forward(self, x):
        y_pred = self.head(self.backbone(x))
        return y_pred
    
    def compute_loss(self, batch):
        x, t, y, _, _ = batch
        target_t = 1 if self.type == "treatment" else 0
        mask = (t == target_t).squeeze()
        if not mask.any():
            return torch.tensor(0.0, device=x.device)
        x_sub = x[mask]
        y_sub = y[mask]
        y_pred = self.forward(x_sub)
        return self.loss_fn(y_pred, y_sub)
    
    @torch.no_grad()
    def predict(self, x):
        y_pred = self.forward(x)
        return y_pred
    
class CATERegression(nn.Module):
    def __init__(self,x_dim, hidden_dim):
        super().__init__()
        self.backbone = Backbone(x_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = MSElossWapper()
        
    def forward(self, x):
        y_pred = self.head(self.backbone(x))
        return y_pred
    
    def compute_loss(self, batch):
        x, t, y, _, _ = batch
        y_pred = self.forward(x)
        return self.loss_fn(y_pred, y)
    
    @torch.no_grad()
    def predict(self, x):
        y_pred = self.forward(x)
        return y_pred
    
class PropensityModel(nn.Module):
    def __init__(self, x_dim, hidden_dim):
        super().__init__()
        self.backbone = Backbone(x_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = BCElossWapper()
        
    def forward(self, x):
        logit = self.head(self.backbone(x))
        return logit
    
    def compute_loss(self, batch):
        x, t, _, _, _ = batch
        logit = self.forward(x)
        return self.loss_fn(logit, t)
    
    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))
    
class XLearnerEstimator:    
    def __init__(self, x_dim, hidden_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mu1 = OutcomeModel(x_dim = x_dim, hidden_dim = hidden_dim, type = 'treatment').to(self.device)
        self.mu0 = OutcomeModel(x_dim = x_dim, hidden_dim = hidden_dim, type = 'control').to(self.device)
    
        self.tau1 = CATERegression(x_dim = x_dim, hidden_dim = hidden_dim).to(self.device)
        self.tau0 = CATERegression(x_dim = x_dim, hidden_dim = hidden_dim).to(self.device)
        
        self.propensity_model = PropensityModel(x_dim, hidden_dim).to(self.device)
        
    def fit_stage1(self,train_loader, valid_loader, epochs, lr = 1e-3):
        optimizer_mu1 = torch.optim.Adam(self.mu1.parameters(), lr=lr)
        trainer_mu1 = Trainer(self.mu1, optimizer_mu1, self.device)
        trainer_mu1.run(train_loader, valid_loader, epochs=epochs)
        
        optimizer_mu0 = torch.optim.Adam(self.mu0.parameters(), lr=lr)
        trainer_mu0 = Trainer(self.mu0, optimizer_mu0, self.device)
        trainer_mu0.run(train_loader, valid_loader, epochs=epochs)
        
    def fit_propensity(self, train_loader, valid_loader, epochs, lr = 1e-3):
        optimizer_e = torch.optim.Adam(self.propensity_model.parameters(), lr=lr)
        trainer_e = Trainer(self.propensity_model, optimizer_e, self.device)
        trainer_e.run(train_loader, valid_loader, epochs=epochs)
        
        
    @torch.no_grad()
    def generate_pseudo_labels(self, dataloader):
        self.mu1.eval()
        self.mu0.eval()
        x_1_all, D_1_all = [], []
        x_0_all, D_0_all = [], []        
        for batch in dataloader:
            x, t, y, _, _ = batch
            x = x.to(self.device)
            t = t.to(self.device)
            y = y.to(self.device)
                
            mask_1 = (t == 1).squeeze()
            mask_0 = (t == 0).squeeze()
            
            x_1 = x[mask_1]
            D_1 = y[mask_1] - self.mu0.predict(x_1)
            x_1_all.append(x_1)
            D_1_all.append(D_1)
            
            x_0 = x[mask_0]
            D_0 = self.mu1.predict(x_0) - y[mask_0] 
            x_0_all.append(x_0)
            D_0_all.append(D_0)
            
        pseudo_1 = (torch.cat(x_1_all, dim=0), torch.cat(D_1_all, dim=0))
        pseudo_0 = (torch.cat(x_0_all, dim=0), torch.cat(D_0_all, dim=0))
        return pseudo_1, pseudo_0
    
    def generate_pseudo_dataset(self, dataloader, batch_size = 64):
        pseudo_1, pseudo_0 = self.generate_pseudo_labels(dataloader)
        x_1, D_1 = pseudo_1
        x_0, D_0 = pseudo_0
        
        dataset_1 = torch.utils.data.TensorDataset(
            x_1,
            torch.zeros_like(D_1),
            D_1,
            torch.zeros_like(D_1),
            torch.zeros_like(D_1),   
        )
        
        dataset_0 = torch.utils.data.TensorDataset(
            x_0,
            torch.zeros_like(D_0),
            D_0,
            torch.zeros_like(D_0),
            torch.zeros_like(D_0),   
        )
        
        loader_1 = torch.utils.data.DataLoader(
            dataset_1,
            batch_size=batch_size,
            shuffle=True,
        )

        loader_0 = torch.utils.data.DataLoader(
            dataset_0,
            batch_size=batch_size,
            shuffle=True,
        )
        return loader_1, loader_0
        
    def fit_stage2(self, train_loader, valid_loader, epochs, lr = 1e-3):
        train_pseudo_loader1, train_pseudo_loader0 = self.generate_pseudo_dataset(train_loader)
        val_pseudo_loader1, val_pseudo_loader0 = self.generate_pseudo_dataset(valid_loader)
        
        optimizer_tau1 = torch.optim.Adam(self.tau1.parameters(), lr=lr)
        trainer_tau1 = Trainer(self.tau1, optimizer_tau1, self.device)
        trainer_tau1.run(train_pseudo_loader1, val_pseudo_loader1, epochs=epochs)
        
        optimizer_tau0 = torch.optim.Adam(self.tau0.parameters(), lr=lr)
        trainer_tau0 = Trainer(self.tau0, optimizer_tau0, self.device)
        trainer_tau0.run(train_pseudo_loader0, val_pseudo_loader0, epochs=epochs)
        
         
    def fit(self, train_loader, valid_loader, epochs, lr = 1e-3):
        self.fit_stage1(train_loader, valid_loader, epochs, lr)
        self.fit_propensity(train_loader, valid_loader, epochs, lr)
        self.fit_stage2(train_loader, valid_loader, epochs, lr)

    
    @torch.no_grad()
    def predict_tau(self, x):
        self.tau1.eval()
        self.tau0.eval()
        tau_from_treated_model = self.tau1.predict(x)
        tau_from_control_model = self.tau0.predict(x)
        return tau_from_treated_model, tau_from_control_model
    
    @torch.no_grad()
    def predict(self, x):
        self.propensity_model.eval()
        tau_from_treated_model, tau_from_control_model = self.predict_tau(x)
        e_hat = self.propensity_model.predict_proba(x)
        return e_hat * tau_from_control_model + (1 - e_hat) * tau_from_treated_model
    
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.mu1.eval()
        self.mu0.eval()
        self.tau1.eval()
        self.tau0.eval()
        self.propensity_model.eval()
        
        tau_hat_all = []
        tau_true_all = []
        
        for batch in dataloader:
            x, t, y, mu0, mu1 = batch
            x = x.to(self.device)
            tau_hat = self.predict(x).detach().cpu()
            tau_true = mu1 - mu0
            tau_hat_all.append(tau_hat)
            tau_true_all.append(tau_true)
        
        tau_hat_all = torch.cat(tau_hat_all, dim = 0).cpu().numpy()
        tau_true_all = torch.cat(tau_true_all, dim = 0).cpu().numpy()
        
        return tau_hat_all, tau_true_all
        

    