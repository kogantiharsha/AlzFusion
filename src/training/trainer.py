import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from .metrics import Metrics


class Trainer:
    
    def __init__(self, model, device, num_classes=9, lr=0.001, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.metrics = Metrics(num_classes=num_classes)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for genetic_features, mri_features, labels in pbar:
            genetic_features = genetic_features.to(self.device)
            mri_features = mri_features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(genetic_features, mri_features)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for genetic_features, mri_features, labels in tqdm(val_loader, desc='Validating'):
                genetic_features = genetic_features.to(self.device)
                mri_features = mri_features.to(self.device)
                labels = labels.to(self.device)
                
                logits, _ = self.model(genetic_features, mri_features)
                loss = self.criterion(logits, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)
    
    def train(self, train_loader, val_loader, num_epochs=50, save_dir='models', 
              save_best=True, log_dir='logs'):
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        best_val_acc = 0.0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 50)
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Val', val_acc, epoch)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f'✓ Saved best model (Val Acc: {val_acc:.2f}%)')
            
            if (epoch + 1) % 10 == 0:
                metrics = self.metrics.compute_metrics(val_labels, val_preds)
                print(f'\nDetailed Metrics (Epoch {epoch+1}):')
                print(f'  F1 Macro: {metrics["f1_macro"]:.4f}')
                print(f'  F1 Weighted: {metrics["f1_weighted"]:.4f}')
        
        writer.close()
        print(f'\nTraining completed! Best Val Acc: {best_val_acc:.2f}%')
        
        return self.history
