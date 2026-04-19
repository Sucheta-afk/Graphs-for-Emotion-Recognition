import os
import glob
import torch
import pickle
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from model import EmotionGNN
import torch.nn as nn

# ─── STEP 1: LOAD ALL PKL FILES ─────────────────────────────────────────────
data_dir = "data/"   # folder where your pkl files are stored

pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))

all_graphs = []

for file in pkl_files:
    with open(file, "rb") as f:
        session_data = pickle.load(f)

        # ─── STEP 2: CONVERT EACH SAMPLE INTO GRAPH ─────────────────────────
        # Assuming session_data is a list of samples
        for sample in session_data:
            x = torch.tensor(sample["x"], dtype=torch.float)              # node features
            edge_index = torch.tensor(sample["edge_index"], dtype=torch.long)
            y = torch.tensor(sample["y"], dtype=torch.long)              # emotion label
            subject_id = torch.tensor(sample["subject"], dtype=torch.long)

            graph = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                subject=subject_id
            )

            all_graphs.append(graph)

print(f"Total graphs loaded: {len(all_graphs)}")

# ─── STEP 3: DATALOADER ─────────────────────────────────────────────────────
loader = DataLoader(all_graphs, batch_size=16, shuffle=True)

# ─── STEP 4: MODEL ──────────────────────────────────────────────────────────
model = EmotionGNN(
    in_channels=5,
    hidden_channels=64,
    num_emotions=4,
    num_subjects=15
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ─── STEP 5: TRAINING LOOP ──────────────────────────────────────────────────
for epoch in range(20):
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        emotion_logits, domain_logits, _ = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        loss = criterion(emotion_logits, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(loader):.4f}")