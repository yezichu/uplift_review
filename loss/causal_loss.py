import torch
import torch.nn as nn
from geomloss import SamplesLoss



class MSElossWrapper(nn.Module):
    def __init__(self, reduction = "mean"):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction = reduction)
    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)


class BCElossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)
    
    
class TARNetlossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction="none")
    def forward(self, y_pred, y_true, w):
        loss = self.loss_fn(y_pred, y_true)
        loss = w * loss
        return loss.mean()
    

class CFRNetlossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction="none")
        self.sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    def forward(self, y_pred, y_true, phi_t, phi_c, w, alpha,  distance = 'wasserstein'):
        loss = self.loss_fn(y_pred, y_true)
        loss = (w * loss).mean()
        if phi_t.shape[0] < 2:
            return loss
        if phi_c.shape[0] < 2:
            return loss
        if distance == 'wasserstein':
            ipm = self.sinkhorn(phi_t, phi_c)
        elif distance == 'mmd':
            ipm = mmd_rbf(phi_t, phi_c)
        else:
            raise ValueError(f"Unknown distance: {distance}")
        return loss + alpha * ipm
    
class DragonnetlossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_h = nn.MSELoss()
        self.loss_g = nn.BCEWithLogitsLoss()
        
    def forward(self, y_pred, y, g_nn, t, eps, alpha, beta):
        g_prob = torch.sigmoid(g_nn)
        loss_h = self.loss_h(y_pred, y)
        loss_gnn = self.loss_g(g_nn, t)
        # clever covariate
        h = t / g_prob - (1 - t) / (1 - g_prob)
        y_tmle = y_pred + eps * h
        loss_tmle = self.loss_h(y_tmle, y)
        return loss_h + alpha * loss_gnn + beta * loss_tmle
        
        
class DESCNlossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_h = nn.MSELoss(reduction="none")
        self.loss_g = nn.BCEWithLogitsLoss()
        
    def forward(self, y0, y1, y, tau_pseudo, logit, t, alpha, beta1, beta0, gamma1, gamma0):
        L_pi = self.loss_g(logit, t)
        pi = torch.sigmoid(logit)
        L_estr = self.loss_h(y*t, y1 * pi).mean()
        L_escr = self.loss_h(y*(1-t), y0 * (1-pi)).mean()
        
        n_T = t.sum().clamp(min=1)
        n_C = (1-t).sum().clamp(min=1)
            
        y1_pseudo = y0 + tau_pseudo
        y0_pseudo = y1 - tau_pseudo
        
        L_crosstr = self.loss_h(y*t, y1_pseudo).sum() / n_T
        L_crosscr = self.loss_h(y*(1-t), y0_pseudo).sum() / n_C
        total_loss = alpha * L_pi + beta1 * L_estr + beta0 * L_escr + gamma1 * L_crosstr + gamma0 * L_crosscr
    
        return total_loss
        
    
    
    
def rbf_kernel(x1, x2, sigma=1.0):
    dist = torch.cdist(x1, x2)
    return torch.exp(-dist**2/(2*sigma**2))

def mmd_rbf(phi_t, phi_c, sigma=1.0):
    m = phi_t.shape[0]
    n = phi_c.shape[0]

    k_tt = rbf_kernel(phi_t, phi_t, sigma)
    k_cc = rbf_kernel(phi_c, phi_c, sigma)
    k_tc = rbf_kernel(phi_t, phi_c, sigma)

    mmd = (
        (k_tt.sum() - torch.diagonal(k_tt).sum()) / (m * (m - 1))
        + (k_cc.sum() - torch.diagonal(k_cc).sum()) / (n * (n - 1))
        - 2.0 * k_tc.mean()
    )
    return mmd

