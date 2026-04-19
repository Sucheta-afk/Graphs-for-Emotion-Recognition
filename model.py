import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


# ─── Gradient Reversal Layer ───────────────────────────────────────────────
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.saved_tensors[0].item()
        return -alpha * grad_output, None  # flip & scale the gradient


def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


# ─── Main Model ────────────────────────────────────────────────────────────
class EmotionGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_emotions, num_subjects, heads=4):
        """
        Args:
            in_channels:     node feature dim (e.g. 5 for 5-band DE features)
            hidden_channels: GAT hidden dim (e.g. 64)
            num_emotions:    number of emotion classes (e.g. 4 for SEED-IV)
            num_subjects:    number of subjects for domain head (e.g. 15 for SEED-IV)
            heads:           number of GAT attention heads
        """
        super().__init__()

        # ── Encoder: 2-layer GAT ──────────────────────────────────────────
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.3)

        # ── Emotion head ─────────────────────────────────────────────────
        self.emotion_mlp = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_emotions)
        )

        # ── Domain head (adversarial) ─────────────────────────────────────
        self.domain_mlp = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_subjects)
        )

    # def forward(self, x, edge_index, batch, alpha=1.0):
    #     # ── Encode ────────────────────────────────────────────────────────
    #     x = F.elu(self.gat1(x, edge_index))
    #     x = F.elu(self.gat2(x, edge_index))

    #     # ── Pool: per-graph embedding z ───────────────────────────────────
    #     z = global_mean_pool(x, batch)           # shape: [batch_size, hidden_channels]

    #     # ── Emotion prediction (normal gradient) ──────────────────────────
    #     emotion_logits = self.emotion_mlp(z)

    #     # ── Domain prediction (REVERSED gradient) ─────────────────────────
    #     z_rev = grad_reverse(z, alpha)
    #     domain_logits = self.domain_mlp(z_rev)

    #     return emotion_logits, domain_logits, z   # z returned for t-SNE later

    def forward(self, x, edge_index, batch, alpha=1.0, use_grl=True):
        # ── Encode ────────────────────────────────────────────────────────
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))

        # ── Pool ─────────────────────────────────────────────────────────
        z = global_mean_pool(x, batch)

        # ── Emotion head ─────────────────────────────────────────────────
        emotion_logits = self.emotion_mlp(z)

        # ── Domain head (optional) ───────────────────────────────────────
        if use_grl:
            z_rev = grad_reverse(z, alpha)
            domain_logits = self.domain_mlp(z_rev)
        else:
            domain_logits = None

        return emotion_logits, domain_logits, z