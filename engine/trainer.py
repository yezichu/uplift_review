import torch
from engine.early_stop import EarlyStopping


class Trainer:
    def __init__(self, model, optimizer, device, patience = 5):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.early_stopping = EarlyStopping(patience = patience)
        
    def _move_to_device(self, batch):
        return tuple(b.to(self.device) for b in batch)
        
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = self._move_to_device(batch)
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    @torch.no_grad()
    def val_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        for batch in loader:
            batch = self._move_to_device(batch)
            loss = self.model.compute_loss(batch)
            total_loss += loss.item()
        return total_loss / len(loader)
    
    
    def run(self, train_loader, val_loader, epochs = 100):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)  
            print(f"Ep {epoch+1:2d} | train={train_loss:.4f} | val={val_loss:.4f}")  
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print("✅ 早停触发！停止训练")
                break