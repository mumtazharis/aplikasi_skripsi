"""
LSTM Training Script untuk Micro-Expression Recognition
menggunakan Flow Series (T, 18) dari generator.py

Data: output_flow/ — variable-length time-series, 9 ROI × 2 (du, dv)
Evaluasi: LOSO (Leave-One-Subject-Out) Cross-Validation
Framework: PyTorch
"""

import torch
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, balanced_accuracy_score
)

# ==========================================
# 1. KONFIGURASI
# ==========================================
DATA_DIR = "output_flow_enriched"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# LSTM Architecture
INPUT_DIM = 63          # 9 ROI × 7 features
HIDDEN_DIM = 128        # LSTM hidden size
NUM_LAYERS = 4          # Number of LSTM layers
BIDIRECTIONAL = True    # Bidirectional LSTM
DROPOUT_LSTM = 0.3      # Dropout between LSTM layers
DROPOUT_FC = 0.5        # Dropout before classifier

# Sequence handling
MAX_SEQ_LEN = 60       # Truncate sequences longer than this (outlier protection)

# Early Stopping
PATIENCE = 10

# Label Mapping (sama dengan notebook sebelumnya — binary: Positive vs Negative)
LABEL_MAPPING = {
    'happy': 'Positive',
    'disgust': 'Negative',
    'anger': 'Negative',
    'fear': 'Negative',
    'sad': 'Negative',
}

# Output directory untuk menyimpan hasil
OUTPUT_DIR = "results_lstm_flow_kfold5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_flow_data(data_dir, label_mapping, max_seq_len=MAX_SEQ_LEN):
    """
    Load variable-length flow series dari folder terstruktur.
    Setiap file .npy berisi array (T, 18).

    Returns:
        sequences: list of np.array, masing-masing shape (T_i, 18)
        labels: np.array of encoded labels
        groups: np.array of subject IDs
        class_names: np.array of class names
        lengths: np.array of original sequence lengths
    """
    sequences = []
    labels = []
    groups = []
    lengths = []

    if not os.path.exists(data_dir):
        raise ValueError(f"Folder dataset tidak ditemukan: {data_dir}")

    available_folders = [
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ]
    available_folders.sort()

    print("=" * 60)
    print("LOADING FLOW DATA")
    print("=" * 60)

    for folder_name in available_folders:
        if folder_name not in label_mapping:
            continue
        target_label = label_mapping[folder_name]
        folder_path = os.path.join(data_dir, folder_name)
        files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
        print(f"  [LOAD] '{folder_name}' -> '{target_label}' ({len(files)} files)")

        for f in files:
            data = np.load(f)

            # Validasi dimensi
            if data.ndim != 2 or data.shape[1] != INPUT_DIM:
                print(f"    [SKIP] {os.path.basename(f)} shape {data.shape} != (T, {INPUT_DIM})")
                continue

            # Truncate jika terlalu panjang
            orig_len = len(data)
            if len(data) > max_seq_len:
                data = data[:max_seq_len]

            # Extract subject ID dari filename: spNO.XXX_filename_onXXX_offXXX.npy
            filename = os.path.basename(f)
            subject_id = filename.split('_')[0]

            sequences.append(data.astype(np.float32))
            labels.append(target_label)
            groups.append(subject_id)
            lengths.append(min(orig_len, max_seq_len))

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    groups = np.array(groups)
    lengths = np.array(lengths)

    print(f"\nTotal Data Loaded: {len(sequences)}")
    print(f"Total Subjects: {len(np.unique(groups))}")
    print(f"Classes: {le.classes_}")
    print(f"Class distribution: {dict(zip(*np.unique(labels_encoded, return_counts=True)))}")
    print(f"Seq length - min: {lengths.min()}, max: {lengths.max()}, "
          f"mean: {lengths.mean():.1f}, median: {np.median(lengths):.1f}")
    print("=" * 60)

    return sequences, labels_encoded, groups, le.classes_, lengths


# ==========================================
# 3. AUGMENTASI untuk Time-Series
# ==========================================
def augment_time_series(seq, label, aug_type='all'):
    """
    Augmentasi untuk satu sequence (T, 18).
    Returns list of (augmented_seq, label) tuples.
    """
    augmented = []

    # 1. Jitter (Gaussian noise)
    noise = np.random.normal(0, 0.01, seq.shape).astype(np.float32)
    augmented.append((seq + noise, label))

    # 2. Scaling
    scale = np.random.uniform(0.8, 1.2)
    augmented.append((seq * scale, label))

    # 3. Temporal flip (reverse time)
    augmented.append((seq[::-1].copy(), label))

    # 4. Mirror regions (swap kanan-kiri)
    # ROI_ORDER: 0:dahi, 1:alis_kanan, 2:alis_kiri, 3:antara_alis, 4:pipi_kanan, 
    #            5:pipi_kiri, 6:hidung, 7:mulut_kanan, 8:mulut_kiri
    # Setiap ROI ada 7 features: du, dv, magnitude, angle, std_u, std_v, max_strain
    roi_swap_map = [0, 2, 1, 3, 5, 4, 6, 8, 7]
    mirrored = np.zeros_like(seq)
    for orig_idx, swap_idx in enumerate(roi_swap_map):
        du = seq[:, swap_idx * 7 + 0]
        dv = seq[:, swap_idx * 7 + 1]
        mag = seq[:, swap_idx * 7 + 2]
        # Angle dihitung manual ulang saja biar pasti aman rotasinya
        std_u = seq[:, swap_idx * 7 + 4]
        std_v = seq[:, swap_idx * 7 + 5]
        max_s = seq[:, swap_idx * 7 + 6]
        
        # Titik kanan-kiri dibalik -> gerak sadar horizontal juga kebalik arahnya
        new_du = -du
        new_angle = np.arctan2(dv, new_du)
        
        mirrored[:, orig_idx * 7 + 0] = new_du
        mirrored[:, orig_idx * 7 + 1] = dv
        mirrored[:, orig_idx * 7 + 2] = mag
        mirrored[:, orig_idx * 7 + 3] = new_angle
        mirrored[:, orig_idx * 7 + 4] = std_u
        mirrored[:, orig_idx * 7 + 5] = std_v
        mirrored[:, orig_idx * 7 + 6] = max_s

    augmented.append((mirrored, label))

    # 5. Random cropping (subsequence) — hanya jika cukup panjang
    if len(seq) > 10:
        crop_len = max(5, int(len(seq) * np.random.uniform(0.7, 0.9)))
        start = np.random.randint(0, len(seq) - crop_len + 1)
        augmented.append((seq[start:start + crop_len].copy(), label))

    return augmented


def apply_augmentation(sequences, labels, minority_class_idx, majority_class_idx):
    """
    Augmentasi untuk data training — oversample minority class.
    """
    aug_seqs = list(sequences)
    aug_labels = list(labels)

    # Hitung rasio imbalance
    n_majority = np.sum(labels == majority_class_idx)
    n_minority = np.sum(labels == minority_class_idx)
    target_ratio = n_majority / max(n_minority, 1)

    # Augmentasi minority class
    minority_indices = np.where(labels == minority_class_idx)[0]
    for idx in minority_indices:
        aug_pairs = augment_time_series(sequences[idx], labels[idx])
        for aug_seq, aug_label in aug_pairs:
            aug_seqs.append(aug_seq)
            aug_labels.append(aug_label)

    # Sedikit augmentasi pada majority juga (untuk regularisasi)
    majority_indices = np.where(labels == majority_class_idx)[0]
    for idx in np.random.choice(majority_indices, size=min(len(majority_indices), n_minority), replace=False):
        noise = np.random.normal(0, 0.01, sequences[idx].shape).astype(np.float32)
        aug_seqs.append(sequences[idx] + noise)
        aug_labels.append(labels[idx])

    return aug_seqs, np.array(aug_labels)


# ==========================================
# 4. PYTORCH DATASET & COLLATE
# ==========================================
class FlowSequenceDataset(Dataset):
    """Dataset untuk variable-length flow sequences."""

    def __init__(self, sequences, labels):
        """
        Args:
            sequences: list of np.array, each shape (T_i, 18)
            labels: np.array of int labels
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.sequences[idx]).float()
        label = torch.tensor(self.labels[idx]).long()
        length = torch.tensor(len(self.sequences[idx])).long()
        return seq, label, length


def collate_fn(batch):
    """
    Custom collate: pad sequences ke panjang terpanjang dalam batch.
    Returns padded sequences, labels, dan lengths.
    """
    sequences, labels, lengths = zip(*batch)

    # Sort by length (descending) — required for pack_padded_sequence
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)

    sequences = [sequences[i] for i in sorted_indices]
    labels = torch.stack([labels[i] for i in sorted_indices])
    lengths = torch.stack([lengths[i] for i in sorted_indices])

    # Pad sequences
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    return padded, labels, lengths


# ==========================================
# 5. MODEL ARCHITECTURE (LSTM)
# ==========================================
class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM untuk klasifikasi time-series flow.

    Architecture:
    - LayerNorm pada input
    - Multi-layer Bidirectional LSTM
    - Attention pooling (menggantikan simple last-hidden)
    - Fully connected classifier dengan dropout
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, num_classes=2,
                 bidirectional=BIDIRECTIONAL,
                 dropout_lstm=DROPOUT_LSTM, dropout_fc=DROPOUT_FC):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_lstm if num_layers > 1 else 0.0
        )

        lstm_output_dim = hidden_dim * self.num_directions

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1, bias=False)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(dropout_fc),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_fc * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, lengths):
        """
        Args:
            x: (batch, max_seq_len, input_dim) — padded sequences
            lengths: (batch,) — actual lengths

        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)

        # Input normalization
        x = self.input_norm(x)

        # Pack padded sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)

        # LSTM forward
        packed_output, (h_n, c_n) = self.lstm(packed)

        # Unpack
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # lstm_output: (batch, max_seq_len, hidden_dim * num_directions)

        # Attention pooling (mask padded positions)
        attn_weights = self.attention(lstm_output).squeeze(-1)  # (batch, max_seq_len)

        # Create mask for padded positions
        max_len = lstm_output.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, max_seq_len)

        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        # context: (batch, hidden_dim * num_directions)

        # Classify
        logits = self.classifier(context)
        return logits


# ==========================================
# 6. TRAINING & EVALUATION FUNCTIONS
# ==========================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Satu epoch training."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for padded_seqs, labels, lengths in dataloader:
        padded_seqs = padded_seqs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        outputs = model(padded_seqs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping (menghindari exploding gradient pada LSTM)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    if scheduler is not None:
        scheduler.step()

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluasi model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for padded_seqs, labels, lengths in dataloader:
            padded_seqs = padded_seqs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            outputs = model(padded_seqs, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# ==========================================
# 7. MAIN: LOSO CROSS-VALIDATION
# ==========================================
def main():
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Config: HIDDEN={HIDDEN_DIM}, LAYERS={NUM_LAYERS}, "
          f"BIDIR={BIDIRECTIONAL}, LR={LEARNING_RATE}, "
          f"EPOCHS={EPOCHS}, PATIENCE={PATIENCE}")

    # 1. Load Data
    sequences, labels, groups, class_names, lengths = load_flow_data(
        DATA_DIR, LABEL_MAPPING, max_seq_len=MAX_SEQ_LEN
    )
    num_classes = len(class_names)

    # Cari index class
    positive_idx = np.where(class_names == 'Positive')[0][0] if 'Positive' in class_names else None
    negative_idx = np.where(class_names == 'Negative')[0][0] if 'Negative' in class_names else None

    if positive_idx is None or negative_idx is None:
        print("ERROR: Positive/Negative classes not found!")
        return

    print(f"\nPositive class index: {positive_idx}")
    print(f"Negative class index: {negative_idx}")

    # 2. K-FOLD
    gkf = GroupKFold(n_splits=5)
    unique_groups = np.unique(groups)

    y_true_all = []
    y_pred_all = []
    fold_results = []

    print(f"\n{'='*60}")
    print(f"MEMULAI LOSO CROSS-VALIDATION")
    print(f"Total Folds (Subjects): {len(unique_groups)}")
    print(f"{'='*60}\n")

    for fold_no, (train_idx, val_idx) in enumerate(
        gkf.split(np.zeros(len(labels)), labels, groups=groups), 1
    ):
        current_subjects = np.unique(groups[val_idx])
        print(f"--- Fold {fold_no} | Test Subjects: {current_subjects} ---")

        # Split data
        train_seqs = [sequences[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        val_labels = labels[val_idx]

        print(f"  Train: {len(train_seqs)} samples | Val: {len(val_seqs)} samples")

        # Augmentasi pada training data
        train_seqs_aug, train_labels_aug = apply_augmentation(
            train_seqs, train_labels,
            minority_class_idx=positive_idx,
            majority_class_idx=negative_idx
        )
        print(f"  After augmentation: {len(train_seqs_aug)} samples")


        # Shuffle training data
        perm = np.random.permutation(len(train_seqs_aug))
        train_seqs_aug = [train_seqs_aug[i] for i in perm]
        train_labels_aug = train_labels_aug[perm]

        # Create DataLoaders
        train_dataset = FlowSequenceDataset(train_seqs_aug, train_labels_aug)
        val_dataset = FlowSequenceDataset(val_seqs, val_labels)

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE,
            shuffle=True, collate_fn=collate_fn,
            num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE,
            shuffle=False, collate_fn=collate_fn,
            num_workers=0, pin_memory=True
        )

        # Setup model, loss, optimizer
        weight_tensor = torch.zeros(num_classes)
        weight_tensor[negative_idx] = 1.0
        weight_tensor[positive_idx] = 1.8
        weight_tensor = weight_tensor.to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        # criterion = nn.CrossEntropyLoss()

        model = LSTMClassifier(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_classes=num_classes,
            bidirectional=BIDIRECTIONAL,
            dropout_lstm=DROPOUT_LSTM,
            dropout_fc=DROPOUT_FC
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-6
        )

        # Early Stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scheduler
            )

            val_loss, val_acc, _, _ = evaluate(
                model, val_loader, criterion, device
            )

            # Print progress
            if (epoch + 1) == 1 or (epoch + 1) % 20 == 0 or (epoch + 1) == EPOCHS:
                print(f"  Epoch [{epoch+1}/{EPOCHS}] "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best model and predict
        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

        _, fold_acc, val_preds, val_trues = evaluate(
            model, val_loader, criterion, device
        )

        y_true_all.extend(val_trues)
        y_pred_all.extend(val_preds)

        fold_uar = balanced_accuracy_score(val_trues, val_preds) if len(val_trues) > 1 else 0
        fold_results.append({
            'fold': fold_no,
            'subjects': ", ".join(current_subjects), 
            'accuracy': fold_acc,
            'uar': fold_uar,
            'n_val': len(val_trues)
        })

        print(f"  --> Fold {fold_no} Acc: {fold_acc:.4f} | UAR: {fold_uar:.4f}\n")

        # Cleanup
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()

    # ==========================================
    # 8. EVALUASI AKHIR
    # ==========================================
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    acc = accuracy_score(y_true_all, y_pred_all)
    uar = balanced_accuracy_score(y_true_all, y_pred_all)
    uf1 = f1_score(y_true_all, y_pred_all, average='macro')

    print("\n" + "=" * 60)
    print("HASIL AKHIR LOSO CROSS-VALIDATION (LSTM)")
    print("=" * 60)
    print(f"Accuracy (Overall) : {acc:.4f}")
    print(f"UAR (Balanced Acc) : {uar:.4f}")
    print(f"UF1 (Macro F1)     : {uf1:.4f}")
    print("=" * 60)

    print("\nClassification Report:")
    report = classification_report(y_true_all, y_pred_all, target_names=class_names, zero_division=0)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - LOSO LSTM Flow\nAcc: {acc:.3f} | UAR: {uar:.3f} | UF1: {uf1:.3f}')
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_lstm_flow.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved: {cm_path}")

    # Per-fold results
    fold_df = pd.DataFrame(fold_results)
    fold_csv_path = os.path.join(OUTPUT_DIR, 'fold_results_lstm_flow.csv')
    fold_df.to_csv(fold_csv_path, index=False)
    print(f"Per-fold results saved: {fold_csv_path}")

    # Summary
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'BiLSTM + Attention',
        'data': 'flow_series (T, 18)',
        'accuracy': acc,
        'uar': uar,
        'uf1': uf1,
        'n_samples': len(y_true_all),
        'n_subjects': len(unique_groups),
        'config': {
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'bidirectional': BIDIRECTIONAL,
            'lr': LEARNING_RATE,
            'epochs': EPOCHS,
            'patience': PATIENCE,
            'max_seq_len': MAX_SEQ_LEN,
            'batch_size': BATCH_SIZE,
        }
    }

    import json
    summary_path = os.path.join(OUTPUT_DIR, 'summary_lstm_flow.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    # Save classification report
    report_path = os.path.join(OUTPUT_DIR, 'classification_report_lstm_flow.txt')
    with open(report_path, 'w') as f:
        f.write(f"LSTM Flow LOSO Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"UAR: {uar:.4f}\n")
        f.write(f"UF1: {uf1:.4f}\n")
        f.write(f"{'='*60}\n\n")
        f.write(report)
    print(f"Report saved: {report_path}")

    # ==========================================
    # 9. TRAIN FINAL MODEL ON ALL DATA & SAVE
    # ==========================================
    print(f"\n{'='*60}")
    print("TRAINING MODEL FINAL PADA SELURUH DATA")
    print(f"{'='*60}")

    # Augmentasi seluruh data
    all_seqs_aug, all_labels_aug = apply_augmentation(
        sequences, labels,
        minority_class_idx=positive_idx,
        majority_class_idx=negative_idx
    )

    # Shuffle
    perm = np.random.permutation(len(all_seqs_aug))
    all_seqs_aug = [all_seqs_aug[i] for i in perm]
    all_labels_aug = all_labels_aug[perm]

    # DataLoader
    final_dataset = FlowSequenceDataset(all_seqs_aug, all_labels_aug)
    final_loader = DataLoader(
        final_dataset, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )

    # Model
    weight_tensor = torch.zeros(num_classes)
    weight_tensor[negative_idx] = 1.0
    weight_tensor[positive_idx] = 2.5
    weight_tensor = weight_tensor.to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    final_model = LSTMClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        bidirectional=BIDIRECTIONAL,
        dropout_lstm=DROPOUT_LSTM,
        dropout_fc=DROPOUT_FC
    ).to(device)

    optimizer = optim.AdamW(
        final_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    FINAL_EPOCHS = 60

    for epoch in range(FINAL_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            final_model, final_loader, criterion, optimizer, device
        )

        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0 or (epoch + 1) == FINAL_EPOCHS:
            print(f"  Epoch [{epoch+1}/{FINAL_EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")

    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'final_lstm_flow_model.pth')
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'class_names': class_names.tolist(),
        'input_dim': INPUT_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'bidirectional': BIDIRECTIONAL,
        'num_classes': num_classes,
    }, model_path)
    print(f"\nModel saved: {model_path}")
    print("SELESAI!")


if __name__ == "__main__":
    main()
