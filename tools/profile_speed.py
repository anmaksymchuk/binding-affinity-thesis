"""
Speed check: graph sizes, batch loading time (CPU), GPU transfer and
model fwd+bwd time. Run from the repo root:  python tools/profile_speed.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from config import Config
from dataset import Ligand_Protein_Dataset
from model import GNN_Only_Predictor

c = Config()
dev = c.device
ds = Ligand_Protein_Dataset(
    root=c.root, data_dir=c.data_dir, complexes_csv=c.complexes_csv,
    dataset_tag=c.dataset_tag, esm_path=c.esm_path, esm_dim=c.esm_dim,
    shard_size=c.graph_shard_size,
)

g = torch.Generator().manual_seed(0)
idx = torch.randperm(len(ds), generator=g)[: 32 * 40].tolist()
sub = ds[idx]

# 1. Graph sizes
nodes = np.array([ds.get(i).num_nodes for i in idx[:500]])
edges = np.array([ds.get(i).edge_index_2.size(1) for i in idx[:500]])
print(f"nodes per graph: mean {nodes.mean():.0f}, median {np.median(nodes):.0f}, max {nodes.max()}")
print(f"edges per graph: mean {edges.mean():.0f}, median {np.median(edges):.0f}, max {edges.max()}")

# 2. Data loading only (CPU)
batches = None
for nw in (0, 2):
    loader = DataLoader(sub, batch_size=32, shuffle=True, num_workers=nw, pin_memory=(dev == "cuda"))
    t = time.time()
    bs = [b for b in loader]
    dt = (time.time() - t) / len(bs)
    print(f"loading, num_workers={nw}: {dt * 1000:.0f} ms/batch")
    batches = batches or bs

# 3. Transfer to GPU
t = time.time()
gb = [b.to(dev) for b in batches[:20]]
if dev == "cuda":
    torch.cuda.synchronize()
print(f"to({dev}): {(time.time() - t) / 20 * 1000:.0f} ms/batch")

# 4. Model forward + backward + step on batches already on the device
model = GNN_Only_Predictor(
    in_channels=c.in_channels, num_gnn_layers=c.num_gnn_layers,
    linear_out_channels=c.linear_out_channels, hidden_dim=c.hidden_dim,
    dropout_gnn=c.dropout_gnn, dropout_mlp=c.dropout_mlp,
).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()

def step(b):
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.huber_loss(model(b).view(-1), b.y.view(-1) - 6.3)
    loss.backward()
    opt.step()

for b in gb[:3]:
    step(b)
if dev == "cuda":
    torch.cuda.synchronize()
t = time.time()
for b in gb:
    step(b)
if dev == "cuda":
    torch.cuda.synchronize()
print(f"model fwd+bwd+step: {(time.time() - t) / len(gb) * 1000:.0f} ms/batch")

if dev == "cuda":
    print(f"GPU memory peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
try:
    import psutil
    print(f"RAM used by process: {psutil.Process().memory_info().rss / 1e9:.1f} GB")
except ImportError:
    pass
