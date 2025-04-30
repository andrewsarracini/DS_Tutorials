# src/models/train_lstm.py

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from src.logger_setup import logger

def train_lstm(model: nn.Module, dataloaders: dict, optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module, device: torch.device, n_epochs=10, 
               verbose=True):
    '''
    Trains a PyTorch LSTM model using epoch-based loop 

    Args: 
        model (nn.Module): LSTM model 
        dataloaders (dict): Must contain 'train' and 'val' DataLoaders
        optimizer (torch.optim.Optimizer): ex. Adam
        loss_fn (torch.nn.Module): loss function, ex. CrossEntropyLoss
        devide (torch.device): torch.device('cuda' or 'cpu') 
        n_epochs (int): Number of training epochs
        verbose (bool): Whether to print progress

    Returns: 
        model: Trained Model
    '''

    model.to(device)

    for epoch in range(n_epochs): 
        model.train()
        train_loss = 0.0 

        for batch in dataloaders['train']:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device) 

            optimizer.zero_grad()
            outputs = model(inputs) 

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step() 

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0 
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in dataloaders['val']:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device) 

                outputs = model(inputs)
                loss = loss_fn(outputs, targets) 
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(targets.cpu().numpy())

            acc = accuracy_score(all_targets, all_preds) 
            f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

            if verbose:
                print(f"[Epoch {epoch+1}/{n_epochs}] "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {acc:.4f} | F1: {f1:.4f}")
                
            logger.info(f'[E{epoch+1}] Train={train_loss:.4f} Val={val_loss:.4f} Acc={acc:.4f} F1={f1:.4f}')
        
        print("✅ LSTM training complete.\n")
        return model