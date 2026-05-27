import torch
import torch.nn as nn
from model.backbone import Backbone
from loss.causal_loss import MSElossWrapper
from engine.trainer import Trainer


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
        self.loss_fn = MSElossWrapper()
        
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
    
    
class SLearnerEstimator:
    def __init__(self, x_dim, hidden_dim, lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SLearner(x_dim = x_dim, hidden_dim = hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.trainer = Trainer(self.model, self.optimizer, self.device)
        
    def fit(self, train_loader, valid_loader, epochs):
        self.trainer.run(train_loader, valid_loader, epochs=epochs)
    
    @torch.no_grad()  
    def predict(self, x):
        self.model.eval()
        return self.model.predict(x)
    
    @torch.no_grad()  
    def evaluate(self, dataloader):
        self.model.eval()
        tau_hat_all = []
        tau_true_all = []
        for batch in dataloader:
            x, t, y, mu0, mu1 = batch
            x = x.to(self.device)
            tau_hat = self.predict(x).cpu()
            tau_true = (mu1 - mu0)
            tau_hat_all.append(tau_hat)
            tau_true_all.append(tau_true)
        tau_hat_all = torch.cat(tau_hat_all, dim=0).cpu().numpy()
        tau_true_all = torch.cat(tau_true_all, dim=0).cpu().numpy()
        return tau_hat_all, tau_true_all

