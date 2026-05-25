import torch
import torch.nn as nn
from model.backbone import Backbone

class SLearner(nn.Module):
    def __init__(self, x_dim, hidden_dim):
        super().__init__()
        self.backbone = Backbone(x_dim + 1, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ) 
    def forward(self, x, t):
        xt =  torch.cat([x, t], dim=-1)
        y_pred = self.head(self.backbone(xt))
        return y_pred
    
    @torch.no_grad()
    def predict(self, x):
        t1 =  torch.ones_like(x[:, 0:1])
        t0 =  torch.zeros_like(x[:,0:1])
        y1 = self.forward(x, t1)
        y0 = self.forward(x, t0)
        return y1 - y0