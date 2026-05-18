import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.utils import softmax
import random

# --- CONFIGURATION ---
HIDDEN_DIM = 64
LR = 0.01
EPOCHS = 100
LAMBDA_ATT = 0.01 
SEED = 42
N_TRIALS = 5
KNN_K = 5

# Refined Granular Mapping based on your Table
CLASS_MAP = {
    "Bridge Down": {
        "AMF (amfx1 bridge-delif)": 0,
        "AUSF (ausfx1 bridge-delif)": 5,
        "UDM (udmx1 bridge-delif)": 11
    },
    "Interface Down": {
        "AMF (amfx1 ens5 interface-down)": 1,
        "AUSF (ausfx1 ens4 interface-down)": 6,
        "UDM (udmx1 ens4 interface-down)": 12
    },
    "Interface Loss": {
        "AMF (amfx1 ens5 interface-loss-start-70)": 2,
        "AUSF (ausfx1 ens4 interface-loss-start-70)": 7,
        "UDM (udmx1 ens4 interface-loss-start-70)": 13
    },
    "Memory Stress": {
        "AMF (amfx1 memory-stress-start)": 3,
        "AUSF (ausfx1 memory-stress-start)": 8,
        "UDM (udmx1 memory-stress-start)": 14
    },
    "CPU Overload": {
        "AMF (amfx1 vcpu-overload-start)": 4,
        "AUSF (ausfx1 vcpu-overload-start)": 9,
        "UDM (udmx1 vcpu-overload-start)": 15
    }
}

# --- UTILITY FUNCTIONS ---
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def select_csv_file():
    try:
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        root.destroy()
        return path
    except: return 'data.csv'

# --- RELATION-AWARE MODEL (Eq. 3-6) ---
class RelationAwareLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations):
        super().__init__()
        self.num_relations = num_relations
        self.out_dim = out_dim
        self.W_r = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)])
        self.a_r = nn.ModuleList([nn.Linear(2 * out_dim, 1, bias=False) for _ in range(num_relations)])
        self.ffn = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_type):
        num_nodes = x.size(0)
        m_v = torch.zeros((num_nodes, self.out_dim), device=x.device)
        total_att_reg = 0
        for r in range(self.num_relations):
            mask = edge_type == r
            if not mask.any(): continue
            rel_edges = edge_index[:, mask]
            h_transformed = self.W_r[r](x)
            e_uv = F.leaky_relu(self.a_r[r](torch.cat([h_transformed[rel_edges[0]], h_transformed[rel_edges[1]]], dim=-1)), 0.2)
            alpha = softmax(e_uv, rel_edges[1], num_nodes=num_nodes)
            m_v.index_add_(0, rel_edges[1], alpha * h_transformed[rel_edges[0]])
            total_att_reg += torch.norm(alpha, 1)
        return self.ln(x + self.ffn(m_v)), total_att_reg

class HeteroFKG(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations):
        super().__init__()
        self.h0 = nn.Linear(input_dim, hidden_dim)
        self.layer1 = RelationAwareLayer(hidden_dim, hidden_dim, num_relations)
        self.layer2 = RelationAwareLayer(hidden_dim, hidden_dim, num_relations)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        h = F.relu(self.h0(data.x))
        h, reg1 = self.layer1(h, data.edge_index, data.edge_type)
        h, reg2 = self.layer2(h, data.edge_index, data.edge_type)
        return self.classifier(h), (reg1 + reg2)

# --- EVALUATION ENGINE ---
def run_trial(trial_idx, x, y, edge_index, edge_type, num_classes):
    set_seed(SEED + trial_idx)
    data = Data(x=x, y=y, edge_index=edge_index, edge_type=edge_type)
    model = HeteroFKG(x.size(1), HIDDEN_DIM, num_classes, num_relations=3)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for _ in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        logits, l_att = model(data)
        loss = F.cross_entropy(logits, y) + (LAMBDA_ATT * l_att)
        loss.backward(); optimizer.step()
    
    model.eval()
    with torch.no_grad():
        logits, _ = model(data)
        probs = F.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        y_true = y.numpy()

    trial_results = {}
    for main_cat, instances in CLASS_MAP.items():
        for inst_name, class_id in instances.items():
            # One-vs-Rest for specific instance
            y_true_bin = (y_true == class_id).astype(int)
            y_pred_bin = (preds == class_id).astype(int)
            inst_probs = probs[:, class_id]
            
            acc = accuracy_score(y_true_bin, y_pred_bin)
            try: auc = roc_auc_score(y_true_bin, inst_probs)
            except: auc = 0.5
            
            trial_results[inst_name] = {'ACC': acc, 'AUC': auc}
    return trial_results

if __name__ == '__main__':
    csv_path = select_csv_file()
    df = pd.read_csv(csv_path).drop(['time', 'source_name'], axis=1, errors='ignore')
    X = torch.tensor(MinMaxScaler().fit_transform(df.drop('y_true(fc)', axis=1).values), dtype=torch.float32)
    y = torch.tensor(LabelEncoder().fit_transform(df['y_true(fc)'].values), dtype=torch.long)
    
    A = kneighbors_graph(X, n_neighbors=KNN_K, mode='connectivity', include_self=False)
    ei = torch.tensor(np.vstack([A.tocoo().row, A.tocoo().col]), dtype=torch.long)
    et = torch.randint(0, 3, (ei.size(1),))

    results_agg = {inst: {'ACC': [], 'AUC': []} for cat in CLASS_MAP.values() for inst in cat}
    
    for i in range(N_TRIALS):
        trial_data = run_trial(i, X, y, ei, et, len(np.unique(y)))
        for inst, metrics in trial_data.items():
            results_agg[inst]['ACC'].append(metrics['ACC'])
            results_agg[inst]['AUC'].append(metrics['AUC'])

    # --- PRINTING IN TABLE FORMAT ---
    print("\n" + "="*100)
    print(f"{'Failure Type':<18} | {'Instance (events)':<40} | {'AUC ↑':<15} | {'ACC ↑':<15}")
    print("-" * 100)
    
    for main_cat, instances in CLASS_MAP.items():
        first = True
        for inst_name in instances.keys():
            m = results_agg[inst_name]
            cat_str = main_cat if first else ""
            print(f"{cat_str:<18} | {inst_name:<40} | {np.mean(m['AUC']):.4f} ± {np.std(m['AUC']):.4f} | {np.mean(m['ACC']):.4f} ± {np.std(m['ACC']):.4f}")
            first = False
        print("-" * 100)
    print("="*100)