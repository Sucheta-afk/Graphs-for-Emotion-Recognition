import torch
from torch_geometric.data import Data, Batch
from model import EmotionGNN

# Dummy batch: 4 graphs, each with 62 nodes, 5 features, ~100 edges
graphs = []
for i in range(4):
    x = torch.randn(62, 5)
    edge_index = torch.randint(0, 62, (2, 100))
    y = torch.tensor([i % 4])
    subject_id = torch.tensor([i % 15])
    graphs.append(Data(x=x, edge_index=edge_index, y=y, subject_id=subject_id))

batch = Batch.from_data_list(graphs)

model = EmotionGNN(in_channels=5, hidden_channels=64, num_emotions=4, num_subjects=15)
emotion_out, domain_out, z = model(batch.x, batch.edge_index, batch.batch, alpha=1.0)

print("emotion_logits shape:", emotion_out.shape)   # expect [4, 4]
print("domain_logits shape: ", domain_out.shape)    # expect [4, 15]
print("z shape:             ", z.shape)             # expect [4, 64]
print("✓ Forward pass OK")

# Check gradients flip through GRL
loss = emotion_out.sum() + domain_out.sum()
loss.backward()
print("✓ Backward pass OK — GRL gradients computed")