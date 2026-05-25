import torch.nn as nn

class MSElossWapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()
    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)
