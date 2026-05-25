from engine.early_stop import EarlyStopping


class Trainer:
    def __init__(self, model, loss_fn, optimizer, device, patience = 5):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.early_stopping = EarlyStopping(patience = patience)
        
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for x, t, y, _, _ in loader:
            x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x, t)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    def val_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        for x, t, y, _, _ in loader:
            x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
            pred = self.model(x, t)
            loss = self.loss_fn(pred, y)
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