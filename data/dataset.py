import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

def load_ihdp_data(data_path, rep=0, scale_y=False):
    data_np = np.load(data_path)
    data = {
        'x': data_np['x'][:, :, rep].astype(np.float32),
        't': data_np['t'][:, rep].astype(np.float32).reshape(-1, 1),
        'y': data_np['yf'][:, rep].astype(np.float32).reshape(-1, 1),
        'mu0': data_np['mu0'][:, rep].astype(np.float32).reshape(-1, 1),
        'mu1': data_np['mu1'][:, rep].astype(np.float32).reshape(-1, 1),
    }
    if scale_y:
        scaler = StandardScaler()
        data['ys'] = scaler.fit_transform(data['y'])
        data['y_scaler'] = scaler
    return data


class IPDHDataset(Dataset):
    def __init__(self, data):
        self.x = torch.FloatTensor(data['x'])
        self.t = torch.FloatTensor(data['t'])
        self.y = torch.FloatTensor(data['y'])
        self.mu0 = torch.FloatTensor(data['mu0'])
        self.mu1 = torch.FloatTensor(data['mu1'])

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.y[idx], self.mu0[idx], self.mu1[idx]


class IPHDDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        
        