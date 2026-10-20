import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms, models
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

OCT_CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CACHE_FILE = os.path.join(DATA_DIR, 'octmnist_resnet18_features.pt')
RAW_NPZ_FILE = os.path.join(DATA_DIR, 'octmnist.npz')

class OCTMNISTRawDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.fromarray(img, mode='L')
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label

def get_feature_extractor(device):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1]).to(device)
    for p in feature_extractor.parameters():
        p.requires_grad = False
    feature_extractor.eval()
    return feature_extractor

def precompute_and_cache_features(raw_npz_path=RAW_NPZ_FILE, cache_output_path=CACHE_FILE, batch_size=256, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print('=' * 65)
    print('OCTMNIST Feature Extraction and Precomputation')
    print('=' * 65)
    print(f'[*] Loading raw data from: {raw_npz_path}')
    if not os.path.exists(raw_npz_path):
        raise FileNotFoundError(f'Raw NPZ dataset not found at: {raw_npz_path}')
    data = np.load(raw_npz_path)
    train_imgs, train_lbls = data['train_images'], data['train_labels'].squeeze()
    val_imgs, val_lbls = data['val_images'], data['val_labels'].squeeze()
    test_imgs, test_lbls = data['test_images'], data['test_labels'].squeeze()
    print('[OK] Raw dataset shapes:')
    print(f'    - Train: {train_imgs.shape} | Labels: {np.bincount(train_lbls)}')
    print(f'    - Val:   {val_imgs.shape}  | Labels: {np.bincount(val_lbls)}')
    print(f'    - Test:  {test_imgs.shape}   | Labels: {np.bincount(test_lbls)}')
    dev = torch.device(device)
    print(f'[*] Initializing frozen ResNet-18 on device: {dev}')
    torch.set_num_threads(os.cpu_count() or 4)
    feature_extractor = get_feature_extractor(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    def extract_split(images, labels, desc):
        images_t = torch.from_numpy(images).unsqueeze(1).float() / 255.0
        labels_t = torch.from_numpy(labels).long()
        ds = TensorDataset(images_t, labels_t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        feats = []
        with torch.inference_mode():
            for xb, _ in tqdm(loader, desc=f'Extracting {desc}', leave=True):
                xb = xb.to(dev).repeat(1, 3, 1, 1)
                xb = torch.nn.functional.interpolate(xb, size=(224, 224), mode='bilinear', align_corners=False)
                xb = (xb - mean) / std
                f = feature_extractor(xb).flatten(1).cpu()
                feats.append(f)
        return torch.cat(feats, dim=0), labels_t
    X_train_f, y_train_t = extract_split(train_imgs, train_lbls, 'Train Set')
    X_val_f, y_val_t = extract_split(val_imgs, val_lbls, 'Val Set')
    X_test_f, y_test_t = extract_split(test_imgs, test_lbls, 'Test Set')
    cache_data = {
        'train_features': X_train_f,
        'train_labels': y_train_t,
        'val_features': X_val_f,
        'val_labels': y_val_t,
        'test_features': X_test_f,
        'test_labels': y_test_t,
        'test_raw_images': torch.tensor(test_imgs, dtype=torch.uint8),
        'classes': OCT_CLASSES,
        'feature_dim': 512,
        'backbone': 'ResNet18_IMAGENET1K_V1'
    }
    os.makedirs(os.path.dirname(cache_output_path), exist_ok=True)
    torch.save(cache_data, cache_output_path)
    file_size_mb = os.path.getsize(cache_output_path) / (1024 * 1024)
    print(f'[OK] Successfully cached features to: {cache_output_path} ({file_size_mb:.2f} MB)')
    print('=' * 65)

def get_octmnist_loaders(batch_size=128, regime='low_data', seed=42, num_classes=4, scale_features=True, cache_path=CACHE_FILE):
    if not os.path.exists(cache_path):
        print(f'[!] Cache file not found at {cache_path}. Building cache now...')
        precompute_and_cache_features(cache_output_path=cache_path)
    data = torch.load(cache_path, map_location='cpu')
    X_tr = data['train_features']
    y_tr = data['train_labels']
    X_va = data['val_features']
    y_va = data['val_labels']
    X_te = data['test_features']
    y_te = data['test_labels']
    test_raw_images = data['test_raw_images']
    if num_classes == 2:
        y_tr = (y_tr != 3).long()
        y_va = (y_va != 3).long()
        y_te = (y_te != 3).long()
    elif num_classes == 4:
        y_tr = y_tr.long()
        y_va = y_va.long()
        y_te = y_te.long()
    else:
        raise ValueError(f'num_classes must be 2 or 4, got {num_classes}')
    n_train_total = len(X_tr)
    if regime == 'low_data':
        subset_ratio = 0.10
        strat_labels = y_tr.numpy()
        _, idx_sub = train_test_split(np.arange(n_train_total), test_size=subset_ratio, random_state=seed, stratify=strat_labels)
        X_tr = X_tr[idx_sub]
        y_tr = y_tr[idx_sub]
    elif regime == 'full':
        n_samples = min(50000, n_train_total)
        strat_labels = y_tr.numpy()
        _, idx_sub = train_test_split(np.arange(n_train_total), test_size=n_samples, random_state=seed, stratify=strat_labels)
        X_tr = X_tr[idx_sub]
        y_tr = y_tr[idx_sub]
    elif regime == 'all':
        pass
    else:
        raise ValueError(f'Unknown regime: {regime}. Choose low_data, full, or all.')
    if scale_features:
        scaler = RobustScaler()
        X_tr = torch.tensor(scaler.fit_transform(X_tr.numpy()), dtype=torch.float32)
        X_va = torch.tensor(scaler.transform(X_va.numpy()), dtype=torch.float32)
        X_te = torch.tensor(scaler.transform(X_te.numpy()), dtype=torch.float32)
    else:
        X_tr = X_tr.float()
        X_va = X_va.float()
        X_te = X_te.float()
    class_counts = torch.bincount(y_tr, minlength=num_classes).float()
    total_samples = len(y_tr)
    class_weights = total_samples / (num_classes * torch.clamp(class_counts, min=1.0))
    class_weights = class_weights / class_weights.sum() * num_classes
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader, test_loader, class_weights, test_raw_images, y_te

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute OCTMNIST ResNet-18 features')
    parser.add_argument('--cache', action='store_true', help='Run feature extraction and caching')
    parser.add_argument('--batch-size', type=int, default=256, help='Extraction batch size')
    args = parser.parse_args()
    if args.cache:
        precompute_and_cache_features(batch_size=args.batch_size)
    else:
        print('Testing get_octmnist_loaders with Low-Data regime (seed 42, 4 classes)...')
        tr_l, va_l, te_l, weights, raw_imgs, y_test = get_octmnist_loaders(batch_size=128, regime='low_data', seed=42, num_classes=4)
        print(f'[OK] Train batches: {len(tr_l)} (Total: {len(tr_l.dataset)})')
        print(f'[OK] Val batches:   {len(va_l)} (Total: {len(va_l.dataset)})')
        print(f'[OK] Test batches:  {len(te_l)} (Total: {len(te_l.dataset)})')
        print(f'[OK] Balanced Class Weights: {weights.tolist()}')
        print(f'[OK] Raw test images shape: {raw_imgs.shape}')
