import torch
import torch.nn as nn
from model.backbone import Backbone
from loss.causal_loss import MSElossWapper

class SLearner(nn.Module):
    """
    优点
        1. 结构极简，单个模型统一建模，代码易实现、训练成本低
        2. 样本利用率高，无需拆分样本
    缺点
        1. 高维特征中干预信号 T 易被协变量稀释，拟合偏弱
        2. 无倾向得分校正，难以抵消混杂变量带来的预估偏差
    """
    def __init__(self, x_dim, hidden_dim):
        super().__init__()
        self.backbone = Backbone(x_dim + 1, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ) 
        self.loss_fn = MSElossWapper()
        
    def forward(self, x, t):
        xt =  torch.cat([x, t], dim=-1)
        y_pred = self.head(self.backbone(xt))
        return y_pred
    
    def compute_loss(self, batch):
        x, t, y, _, _ = batch
        y_pred = self.forward(x, t)
        return self.loss_fn(y_pred, y)

    @torch.no_grad()
    def predict(self, x):
        t1 =  torch.ones_like(x[:, 0:1])
        t0 =  torch.zeros_like(x[:,0:1])
        y1 = self.forward(x, t1)
        y0 = self.forward(x, t0)
        return y1 - y0