import os
import random
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =========================================================
# CONFIG
# =========================================================
DATA_DIR = "train_data"
SAMPLE_RATE = 22050
TARGET_DURATION = 5  # seconds
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
MODEL_PATH = "models/cnn_emotion_model.pth"

EMOTIONS = ["angry", "frustrated", "happy", "neutral", "sad", "satisfied"]
LABEL_TO_IDX = {label: i for i, label in enumerate(EMOTIONS)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# DATASET
# =========================================================
class EmotionMelDataset(Dataset):
    def __init__(self, root_dir: str, augment: bool = False):
        self.root_dir = Path(root_dir)
        self.augment = augment
        self.samples = []

        for emotion in EMOTIONS:
            emotion_dir = self.root_dir / emotion
            if not emotion_dir.exists():
                continue

            for file in emotion_dir.iterdir():
                if file.suffix.lower() in AUDIO_EXTENSIONS:
                    self.samples.append((str(file), LABEL_TO_IDX[emotion]))

        if not self.samples:
            raise ValueError(f"No audio files found in {root_dir}")

        print(f"\nLoaded {len(self.samples)} files from {root_dir}")
        counts = {e: 0 for e in EMOTIONS}
        for _, label in self.samples:
            counts[IDX_TO_LABEL[label]] += 1
        print("Class distribution:")
        for emotion, count in counts.items():
            print(f"  {emotion}: {count}")

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, path: str):
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=25)

        # Normalize
        if len(audio) > 0:
            audio = librosa.util.normalize(audio)

        # Fixed duration
        target_len = SAMPLE_RATE * TARGET_DURATION
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, max(0, target_len - len(audio))))

        return audio, sr

    def _augment_audio(self, audio: np.ndarray):
        # light augmentation only
        choice = random.choice(["none", "noise", "pitch", "stretch"])

        if choice == "noise":
            noise = np.random.normal(0, 0.003, len(audio))
            audio = audio + noise

        elif choice == "pitch":
            steps = random.uniform(-1.0, 1.0)
            audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=steps)

        elif choice == "stretch":
            rate = random.uniform(0.9, 1.1)
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            target_len = SAMPLE_RATE * TARGET_DURATION
            if len(stretched) > target_len:
                audio = stretched[:target_len]
            else:
                audio = np.pad(stretched, (0, max(0, target_len - len(stretched))))

        return audio

    def _to_log_mel(self, audio: np.ndarray, sr: int):
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Standardize per sample
        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-8
        log_mel = (log_mel - mean) / std

        return log_mel.astype(np.float32)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        audio, sr = self._load_audio(path)

        if self.augment:
            audio = self._augment_audio(audio)

        log_mel = self._to_log_mel(audio, sr)

        # Add channel dimension -> [1, n_mels, time]
        tensor = torch.tensor(log_mel).unsqueeze(0)

        return tensor, torch.tensor(label, dtype=torch.long)


# =========================================================
# MODEL
# =========================================================
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================================================
# HELPERS
# =========================================================
def make_weighted_sampler(dataset_subset, full_dataset):
    labels = []
    for idx in dataset_subset.indices:
        _, label = full_dataset.samples[idx]
        labels.append(label)

    class_counts = np.bincount(labels, minlength=len(EMOTIONS))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[label] for label in labels]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())

    acc = accuracy_score(all_true, all_preds)
    return total_loss / max(len(loader), 1), acc, all_true, all_preds


# =========================================================
# TRAIN
# =========================================================
def train():
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    full_dataset = EmotionMelDataset(DATA_DIR, augment=False)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    # separate dataset object with augmentation for train
    train_dataset = EmotionMelDataset(DATA_DIR, augment=True)
    train_dataset.samples = [full_dataset.samples[i] for i in train_subset.indices]

    val_dataset = EmotionMelDataset(DATA_DIR, augment=False)
    val_dataset.samples = [full_dataset.samples[i] for i in val_subset.indices]

    train_sampler = make_weighted_sampler(train_subset, full_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = EmotionCNN(num_classes=len(EMOTIONS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, device, criterion)

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "label_to_idx": LABEL_TO_IDX,
                "idx_to_label": IDX_TO_LABEL,
                "sample_rate": SAMPLE_RATE,
                "target_duration": TARGET_DURATION,
                "n_mels": N_MELS,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "best_val_acc": best_val_acc,
            }, MODEL_PATH)
            print(f"✓ Saved best model to {MODEL_PATH}")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Final report using best model weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, final_acc, y_true, y_pred = evaluate(model, val_loader, device, criterion)

    print("\nFinal Validation Accuracy:", f"{final_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    train()