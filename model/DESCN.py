import torch
import torch.nn as nn
from model.backbone import Backbone
from engine.trainer import Trainer
from loss.causal_loss import DESCNlossWrapper

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
    
    
class PropensityModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        logit = self.head(x)
        return logit
    
    @torch.no_grad()
    def predict_prob(self, x):
        return torch.sigmoid(self.forward(x))
    

class DESCN(nn.Module):
    def __init__(self,x_dim, hidden_dim, alpha = 1, beta1 = 0.5, beta0 = 0.5, gamma1 = 0.5, gamma0 = 0.5):
        super().__init__()
        self.backbone = Backbone(x_dim, hidden_dim)
        self.mu1 = OutcomeModel(hidden_dim)
        self.mu0 = OutcomeModel(hidden_dim)
        self.tau_pseudo = OutcomeModel(hidden_dim)
        self.pi = PropensityModel(hidden_dim)
        self.loss_fn = DESCNlossWrapper()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta0 = beta0
        self.gamma1 = gamma1
        self.gamma0 = gamma0
        
    def forward(self, x):
        phi = self.backbone(x)
        y0 = self.mu0(phi)
        y1 = self.mu1(phi)
        tau_pseudo = self.tau_pseudo(phi)
        logit = self.pi(phi)
        return y0, y1, tau_pseudo, logit
    
    def compute_loss(self, batch):
        x, t, y, _, _ = batch 
        y0, y1, tau_pseudo, logit = self.forward(x)
        loss = self.loss_fn(y0, y1, y, tau_pseudo, logit, t, self.alpha, self.beta1, self.beta0, self.gamma1, self.gamma0)
        return loss
    
    @torch.no_grad()
    def predict(self, x):
        y0, y1, tau_pseudo, logit = self.forward(x)
        return y1 - y0
    
    
class DESCNEstimator:
    def __init__(self, x_dim, hidden_dim, lr = 1e-3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DESCN(x_dim = x_dim, hidden_dim = hidden_dim).to(self.device)
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
        
    
