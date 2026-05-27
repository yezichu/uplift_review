import torch
import torch.nn as nn
from model.backbone import Backbone
from engine.trainer import Trainer
from loss.causal_loss import TARNetlossWrapper


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
        
class TARNet(nn.Module):
    def __init__(self,x_dim, hidden_dim,):
        super().__init__()
        self.h1 = OutcomeModel(hidden_dim)
        self.h0 = OutcomeModel(hidden_dim)
        self.backbone = Backbone(x_dim, hidden_dim)
        self.u = None
        self.loss_fn = TARNetlossWrapper()
        
    def forward(self, x):
        phi = self.backbone(x)
        y_t = self.h1(phi)
        y_c = self.h0(phi)
        return y_t, y_c
    
    def compute_loss(self, batch):
        x, t, y, _, _ = batch 
        y_t, y_c = self.forward(x)
        y_pred = t * y_t + (1 - t) * y_c
        w = t/(2*self.u) + (1 - t)/(2*(1-self.u))
        return self.loss_fn(y_pred, y, w)
    
    @torch.no_grad()
    def predict(self, x):
        mu1_hat, mu0_hat = self.forward(x)
        tau_hat = mu1_hat - mu0_hat
        return mu1_hat, mu0_hat, tau_hat


class TARNetEstimator:    
    def __init__(self, x_dim, hidden_dim, lr = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TARNet(x_dim = x_dim, hidden_dim = hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = Trainer(self.model, self.optimizer, self.device)
            
    def estimate_treatment_prob(self, dataloader):
        total = 0
        treated = 0
        for batch in dataloader:
            _, t, _, _, _ = batch
            treated += t.sum().item()
            total += t.numel()
        return treated / total
    
    def fit(self, train_loader, valid_loader, epochs):
        treatment_prob = self.estimate_treatment_prob(train_loader)
        self.model.u = treatment_prob
        self.trainer.run(train_loader, valid_loader, epochs=epochs)
       

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        mu1_hat, mu0_hat, tau_hat = self.model.predict(x)
        return mu1_hat, mu0_hat, tau_hat
        
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        tau_hat_all = []
        tau_true_all = []
        for batch in dataloader:
            x, t, y, mu0, mu1 = batch
            x = x.to(self.device)
            mu_hat_1, mu_hat_0, tau_hat = self.model.predict(x)
            # true treatment effect
            tau_true = (mu1 - mu0)
            tau_hat_all.append(tau_hat)
            tau_true_all.append(tau_true)
        tau_hat_all = torch.cat(tau_hat_all, dim=0).cpu().numpy()
        tau_true_all = torch.cat(tau_true_all, dim=0).cpu().numpy()
        return tau_hat_all, tau_true_all
    