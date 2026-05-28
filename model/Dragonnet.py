import torch
import torch.nn as nn
from model.backbone import Backbone
from engine.trainer import Trainer
from loss.causal_loss import DragonnetlossWrapper




class OutcomeModel(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        y_pred = self.head(x)
        return y_pred

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
        
class Dragonnet(nn.Module):
    def __init__(self,x_dim, hidden_dim, alpha = 0.1, beta = 0.1):
        super().__init__()
        self.h1 = OutcomeModel(hidden_dim)
        self.h0 = OutcomeModel(hidden_dim)
        self.backbone = Backbone(x_dim, hidden_dim)
        self.g = PropensityModel(hidden_dim)
        self.loss_fn = DragonnetlossWrapper()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        phi = self.backbone(x)
        y_t = self.h1(phi)
        y_c = self.h0(phi)
        g_nn = self.g(phi)
        return y_t, y_c, g_nn
    
    def compute_loss(self, batch):
        x, t, y, _, _ = batch 
        y_t, y_c, g_nn = self.forward(x)
        y_pred = t * y_t + (1 - t) * y_c
        return self.loss_fn(y_pred, y, g_nn, t, self.epsilon, self.alpha, self.beta)
    
    @torch.no_grad()
    def predict(self, x):
        mu1_hat, mu0_hat, _ = self.forward(x)
        tau_hat = mu1_hat - mu0_hat
        return tau_hat


class DragonnetEstimator:    
    def __init__(self, x_dim, hidden_dim, lr = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Dragonnet(x_dim = x_dim, hidden_dim = hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = Trainer(self.model, self.optimizer, self.device)
            
    def fit(self, train_loader, valid_loader, epochs):
        self.trainer.run(train_loader, valid_loader, epochs=epochs)
       
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
    