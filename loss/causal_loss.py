import torch.nn as nn


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
    