import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
import time

# --- 🔢 Dataset Classes ---

class ContrastiveWiSDMDataset(Dataset):
    def __init__(self, data, seq_len=128):
        self.samples = []
        for i in range(0, len(data) - seq_len, seq_len):
            self.samples.append(data[i:i+seq_len])
        self.samples = torch.tensor(self.samples, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        return self.augment(x), self.augment(x)

    def augment(self, x):
        x = x.clone()
        x += torch.randn_like(x) * 0.01
        x *= torch.rand(1).item() * 0.1 + 0.95
        return x

class LabeledWiSDMDataset(Dataset):
    def __init__(self, df, seq_len=128):
        self.samples = []
        self.labels = []
        self.label_map = {label: i for i, label in enumerate(df['activity'].unique())}

        grouped = df.groupby('user')
        for _, user_df in grouped:
            user_df = user_df.reset_index(drop=True)
            for i in range(0, len(user_df) - seq_len, seq_len):
                chunk = user_df.iloc[i:i+seq_len]
                if {'x', 'y', 'z'}.issubset(chunk.columns):
                    xyz = chunk[['x', 'y', 'z']].values.astype(np.float32)
                    label = self.label_map.get(chunk['activity'].values[0], 0)
                    self.samples.append(xyz)
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx]), self.labels[idx]

# --- 🧠 Model ---

class TimeSeriesEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.projector = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        feat = self.encoder(x).squeeze(-1)
        proj = self.projector(feat)
        return F.normalize(proj, dim=1)

class ActivityClassifier(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)
        return self.classifier(z)

# --- 🔥 Contrastive Loss ---

def contrastive_loss(z1, z2, temp=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
    labels = torch.cat([torch.arange(N) + N, torch.arange(N)]).to(z.device)
    mask = torch.eye(2 * N, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -9e15)
    return F.cross_entropy(sim, labels)

# --- 🚀 Training Functions ---

def train_contrastive(model, loader, optimizer, device, epochs=10):
    losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1 = model(x1)
            z2 = model(x2)
            loss = contrastive_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f"Contrastive Epoch {epoch+1} Loss: {avg_loss:.4f}")
    return losses

def train_supervised(model, loader, optimizer, criterion, device, epochs=10):
    losses, accs = [], []
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for x, y in loader:
            x, y = x.to(device), torch.tensor(y).to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        acc = correct / total
        losses.append(total_loss / len(loader))
        accs.append(acc)
        print(f"Supervised Epoch {epoch+1}, Loss: {losses[-1]:.4f}, Acc: {acc:.4f}")
    return losses, accs

# --- 📈 Plotting ---

def plot_metrics(metric_lists, labels, ylabel, title):
    for metric, label in zip(metric_lists, labels):
        plt.plot(metric, label=label)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# --- 🔧 MAIN PIPELINE ---

csv_path = 'WiSDM_dataset.csv'
df = pd.read_csv(csv_path)

if not {'x', 'y', 'z'}.issubset(df.columns):
    raise ValueError("The dataset must contain 'x', 'y', 'z' columns.")

# --- Pretraining with Contrastive Learning ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = df[['x','y','z']].values.astype(np.float32)
contrastive_dataset = ContrastiveWiSDMDataset(data)
contrastive_loader = DataLoader(contrastive_dataset, batch_size=256, shuffle=True)

encoder = TimeSeriesEncoder().to(device)
opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
contrastive_losses = train_contrastive(encoder, contrastive_loader, opt, device, epochs=10)

# --- Fine-tuning with Supervised Labels ---
labeled_dataset = LabeledWiSDMDataset(df)
train_size = int(0.8 * len(labeled_dataset))
val_size = len(labeled_dataset) - train_size
train_set, val_set = random_split(labeled_dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

classifier = ActivityClassifier(encoder, num_classes=6).to(device)
opt2 = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
sup_losses, sup_accs = train_supervised(classifier, train_loader, opt2, criterion, device, epochs=10)

# --- 📊 Visualization ---
plot_metrics([contrastive_losses], ['Contrastive Loss'], 'Loss', 'Contrastive Learning Loss')
plot_metrics([sup_accs], ['Accuracy'], 'Accuracy', 'Supervised Fine-tuning Accuracy')
plot_metrics([sup_losses], ['Loss'], 'Loss', 'Supervised Fine-tuning Loss')
