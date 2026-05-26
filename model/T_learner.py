import torch
import torch.nn as nn
from model.backbone import Backbone
from loss.causal_loss import MSElossWapper


class TLearner(nn.Module):
    def __init__(self, x_dim, hidden_dim, type = 'treatment'):
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
        x_sub = x[mask]
        y_sub = y[mask]
        y_pred = self.forward(x_sub)
        return self.loss_fn(y_pred, y_sub)
    
    @torch.no_grad()
    def predict(self, x):
        y_pred = self.forward(x)
        return y_pred
    
    

    
        