import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from src.models import build_model
from src.metrics import compute_all_metrics
from src.gating_analysis import compute_gating_entropy, analyze_class_gating

def train_single_model(model_name, train_loader, val_loader, test_loader, class_weights, seed=42, epochs=20, lr=1e-3, weight_decay=1e-4, num_classes=4, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    model, n_params = build_model(model_name, in_features=512, num_classes=num_classes)
    model = model.to(dev)
    weights_tensor = class_weights.to(dev) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_bacc = -1.0
    best_model_weights = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_bacc': [], 'val_f1': [], 'epoch_entropy': []}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        gate_weights_epoch = []
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            if hasattr(model, 'gate') and ('proposed' in model_name or 'symmetrical' in model_name):
                logits, w = model(xb, return_gates=True)
                gate_weights_epoch.append(w.detach().cpu().numpy())
            else:
                logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(yb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        model.eval()
        val_loss, all_val_preds, all_val_targets, all_val_probs = 0.0, [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                logits = model(xb)
                v_loss = criterion(logits, yb)
                val_loss += v_loss.item() * len(yb)
                probs = torch.softmax(logits, dim=1)
                all_val_probs.append(probs.cpu())
                all_val_targets.append(yb.cpu())
        val_loss = val_loss / len(val_loader.dataset)
        val_probs = torch.cat(all_val_probs, dim=0)
        val_targets = torch.cat(all_val_targets, dim=0)
        val_metrics, _ = compute_all_metrics(val_targets, val_probs, num_classes=num_classes)
        epoch_h = 1.0
        if len(gate_weights_epoch) > 0:
            all_gw = np.concatenate(gate_weights_epoch, axis=0)
            epoch_h = float(np.mean(compute_gating_entropy(all_gw)))
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_bacc'].append(val_metrics['balanced_accuracy'])
        history['val_f1'].append(val_metrics['f1_macro'])
        history['epoch_entropy'].append(epoch_h)
        if val_metrics['balanced_accuracy'] > best_val_bacc:
            best_val_bacc = val_metrics['balanced_accuracy']
            best_model_weights = copy.deepcopy(model.state_dict())
    total_training_time = time.time() - t0
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    model.eval()
    all_test_probs, all_test_targets, all_test_gates = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            if hasattr(model, 'gate') and ('proposed' in model_name or 'symmetrical' in model_name):
                logits, w = model(xb, return_gates=True)
                all_test_gates.append(w.cpu().numpy())
            else:
                logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            all_test_probs.append(probs.cpu())
            all_test_targets.append(yb.cpu())
    test_probs = torch.cat(all_test_probs, dim=0)
    test_targets = torch.cat(all_test_targets, dim=0)
    test_metrics, test_cm = compute_all_metrics(test_targets, test_probs, num_classes=num_classes)
    gating_summary = None
    if len(all_test_gates) > 0:
        test_gw = np.concatenate(all_test_gates, axis=0)
        gating_summary = analyze_class_gating(test_gw, test_targets.numpy(), num_classes=num_classes)
    return {
        'model_name': model_name,
        'seed': seed,
        'num_params': n_params,
        'epochs': epochs,
        'best_val_bacc': float(best_val_bacc),
        'test_metrics': test_metrics,
        'confusion_matrix': test_cm.tolist(),
        'gating_analysis': gating_summary,
        'history': history,
        'training_time_sec': float(total_training_time)
    }
