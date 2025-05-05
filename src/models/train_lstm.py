# src/models/train_lstm.py

import torch
import torch.nn as nn
import pandas as pd
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
    # moves model to GPU or CPU 
    # required *before* training starts!
    model.to(device)
    print("✅ Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU\n")

    # puts model in training mode 
    # count up the epoch's total training loss
    for epoch in range(n_epochs): 
        model.train()
        train_loss = 0.0 

        # load each batch of sequences and labels 
        # move both to the GPU
        for batch in dataloaders['train']:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device) 

            # clear prev gradients (zero_grad) 
            # forward pass: `outputs` are raw scores (logits)
            optimizer.zero_grad()
            outputs = model(inputs) 

            # compute the loss 
            # `.backward` computes gradients
            # `.step` updates model weights
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step() 

            # add the batch's loss to the epoch total
            train_loss += loss.item()

        # ===Validation ===
        # `eval` disables dropout & "training-only" layers
        # Initialize eval accumulators 
        model.eval()
        val_loss = 0.0 
        all_preds, all_targets = [], []

        # `no_grad` -- avoids gradients (saves memory) 
        # compute predictions (argmax for class index) 
        # store preds and target for metric calculation
        with torch.no_grad():
            for batch in dataloaders['val']:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device) 

                outputs = model(inputs)
                loss = loss_fn(outputs, targets) 
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy()) 

            print("[DEBUG] Pred label counts:", pd.Series(all_preds).value_counts())
            print("[DEBUG] True label counts:", pd.Series(all_targets).value_counts())

            # sklearn metrics
            acc = accuracy_score(all_targets, all_preds) 
            f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

            if verbose:
                print(f"[Epoch {epoch+1}/{n_epochs}] "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {acc:.4f} | F1: {f1:.4f}")
                
            logger.info(f'[E{epoch+1}] Train={train_loss:.4f} Val={val_loss:.4f} Acc={acc:.4f} F1={f1:.4f}')
        
    print("✅ LSTM training complete.\n")
    return model
    