import os
import glob
import pickle
import torch
from torch_geometric.data import Data

data_dir = "data/"   # folder with all .pkl files

pkl_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))

all_graphs = []

for file in pkl_files:
    print(f"Processing: {file}")

    # Extract subject ID from filename
    # subject_01_session_1.pkl → 0
    subject_id = int(file.split("_")[1]) - 1

    with open(file, "rb") as f:
        session_data = pickle.load(f)

        # 🔥 THIS DEPENDS ON YOUR PKL STRUCTURE
        # assuming it's a list of samples
        for sample in session_data:

            # ─── Extract fields (adjust if needed) ───
            x = torch.tensor(sample["x"], dtype=torch.float)
            edge_index = torch.tensor(sample["edge_index"], dtype=torch.long)
            y = torch.tensor(sample["y"], dtype=torch.long)

            graph = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                subject_id=torch.tensor(subject_id, dtype=torch.long)
            )

            all_graphs.append(graph)

print(f"\nTotal graphs created: {len(all_graphs)}")

# ─── Save ─────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
torch.save(all_graphs, "data/all_graphs.pt")

print("Saved to data/all_graphs.pt")