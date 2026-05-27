import torch
import torch.nn as nn
from geomloss import SamplesLoss



class MSElossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()
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

