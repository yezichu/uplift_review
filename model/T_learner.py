import torch
import torch.nn as nn
from model.backbone import Backbone
from engine.trainer import Trainer
from loss.causal_loss import MSElossWrapper


class TLearner(nn.Module):
    """
    优点
        1. 结构极简，代码易实现、训练成本低
        2. 结果可解释性较好，两个模型独立训练
    缺点
        1. 数据利用率低
        2. 无倾向得分校正，难以抵消混杂变量带来的预估偏差
        3. 在 treatment imbalance 下效果不好。
    """
    def __init__(self, x_dim, hidden_dim, type = 'treatment'):
        super().__init__()
        self.backbone = Backbone(x_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = MSElossWrapper()
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
    

class TLearnerEstimator:    
    def __init__(self, x_dim, hidden_dim, lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_t = TLearner(x_dim = x_dim, hidden_dim = hidden_dim, type = 'treatment').to(self.device)
        self.optimizer_t = torch.optim.Adam(self.model_t.parameters(), lr=lr)
        self.trainer_t = Trainer(self.model_t, self.optimizer_t, self.device)
        
        self.model_c = TLearner(x_dim = x_dim, hidden_dim = hidden_dim, type = 'control').to(self.device)
        self.optimizer_c = torch.optim.Adam(self.model_c.parameters(), lr=lr)
        self.trainer_c = Trainer(self.model_c, self.optimizer_c, self.device)
        
    def fit(self, train_loader, valid_loader, epochs):
        self.trainer_t.run(train_loader, valid_loader, epochs=epochs)
        self.trainer_c.run(train_loader, valid_loader, epochs=epochs)

    @torch.no_grad()
    def predict(self, x):
        self.model_t.eval()
        self.model_c.eval()
        return self.model_t.predict(x) - self.model_c.predict(x)
        
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model_t.eval()
        self.model_c.eval()
        tau_hat_all = []
        tau_true_all = []
        
        for batch in dataloader:
            x, t, y, mu0, mu1 = batch
            x = x.to(self.device)
            mu_hat_1 = self.model_t.predict(x).cpu()
            mu_hat_0 = self.model_c.predict(x).cpu()
            tau_hat = mu_hat_1 - mu_hat_0
            # true treatment effect
            tau_true = (mu1 - mu0)
            tau_hat_all.append(tau_hat)
            tau_true_all.append(tau_true)
        tau_hat_all = torch.cat(tau_hat_all, dim=0).cpu().numpy()
        tau_true_all = torch.cat(tau_true_all, dim=0).cpu().numpy()
        return tau_hat_all, tau_true_all