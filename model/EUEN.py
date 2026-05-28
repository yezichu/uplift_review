import torch
import torch.nn as nn
from model.backbone import Backbone
from engine.trainer import Trainer
from loss.causal_loss import MSElossWrapper

class OutcomeModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.head(x)
    
    @torch.no_grad()
    def predict(self, x):
        return self.forward(x)
    
class EUEN(nn.Module):
    def __init__(self, x_dim, hidden_dim, alpha = 1):
        super().__init__()
        self.backbone = Backbone(x_dim, hidden_dim)
        self.tau0 = OutcomeModel(hidden_dim)
        self.tau = OutcomeModel(hidden_dim)
        self.loss_fn = MSElossWrapper(reduction = 'none')
        self.alpha = alpha
        
    def forward(self, x):
        phi = self.backbone(x)
        y_0 = self.tau0(phi)
        y_pred = self.tau(phi)
        return y_0, y_pred
    
    def compute_loss(self, batch):
        x, t, y, _, _ = batch 
        y_0, y_pred = self.forward(x)
        loss0 = self.loss_fn(y_0, y)
        loss1 = self.loss_fn(y_0 + y_pred, y)
        L = ((1 - t) * loss0).sum() / (1 - t).sum().clamp(min=1)
        J = (t * loss1).sum() / t.sum().clamp(min=1)
        return self.alpha * L + J
    
    @torch.no_grad()
    def predict(self, x):
        _, y_pred = self.forward(x)
        return y_pred
        
     
class EUENEstimator:
    def __init__(self, x_dim, hidden_dim, lr = 1e-3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EUEN(x_dim = x_dim, hidden_dim = hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = Trainer(self.model, self.optimizer, self.device)
        
    def fit(self, train_loader, valid_loader, epochs):
        self.trainer.run(train_loader, valid_loader, epochs)
        
    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        tau_hat = self.model.predict(x)
        return tau_hat
        
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        tau_hat_all = []
        tau_true_all = []
        for batch in dataloader:
            x, t, y, mu0, mu1 = batch
            x = x.to(self.device)
            tau_hat = self.model.predict(x)
            # true treatment effect
            tau_true = (mu1 - mu0)
            tau_hat_all.append(tau_hat)
            tau_true_all.append(tau_true)
        tau_hat_all = torch.cat(tau_hat_all, dim=0).cpu().numpy()
        tau_true_all = torch.cat(tau_true_all, dim=0).cpu().numpy()
        return tau_hat_all, tau_true_all