# ============================================================
# Unified KG/GNN Failure Localization Models for 5G Core RCA
# Models: KGroot, GCN, KAFD, LAMs, CF-GNN, HAN, HP-GCN, Proposed
# Metrics: Micro F1, Macro F1, Precision, Recall, AUC
# Reports: Mean ± Standard Deviation over multiple runs
# ============================================================

import random
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tkinter import Tk, filedialog

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.neighbors import kneighbors_graph

from torch_geometric.data import Data
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv
)
from torch_geometric.utils import from_scipy_sparse_matrix, to_dense_adj


warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_COL = "y_true(fc)"
DROP_COLS = ["time", "source_name"]

HIDDEN_DIM = 128
DROPOUT = 0.3
LR = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 150
RUNS = 5
K_NEIGHBORS = 10

SEEDS = [42, 52, 62, 72, 82]


# ============================================================
# Utility Functions
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_csv_file():
    root = Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Select 5G Telemetry CSV",
        filetypes=[("CSV files", "*.csv")]
    )

    if not path:
        raise FileNotFoundError("No CSV file selected.")

    return path


def load_and_preprocess_data():
    csv_path = select_csv_file()
    df = pd.read_csv(csv_path)

    df.drop(columns=DROP_COLS, inplace=True, errors="ignore")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    X = df.drop(columns=[TARGET_COL]).values
    y = df[TARGET_COL].values

    y = LabelEncoder().fit_transform(y)
    X = MinMaxScaler().fit_transform(X)

    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    return x, y


def build_graph(x, k=10):
    adj = kneighbors_graph(
        x.cpu().numpy(),
        n_neighbors=k,
        mode="connectivity",
        include_self=True
    )

    edge_index, edge_weight = from_scipy_sparse_matrix(adj)

    return edge_index.long(), edge_weight.float()


def create_masks(num_nodes, seed):
    indices = np.arange(num_nodes)

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=0.6,
        random_state=seed,
        shuffle=True
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        shuffle=True
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def evaluate(model, data, mask):
    model.eval()

    with torch.no_grad():
        out = model(data)
        prob = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)

    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    y_prob = prob[mask].cpu().numpy()

    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    try:
        num_classes = int(data.y.max().item() + 1)

        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_true_bin = label_binarize(
                y_true,
                classes=np.arange(num_classes)
            )

            auc = roc_auc_score(
                y_true_bin,
                y_prob,
                average="macro",
                multi_class="ovr"
            )

    except ValueError:
        auc = 0.0

    return micro_f1, macro_f1, precision, recall, auc


# ============================================================
# 1. GCN Model
# ============================================================

class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=DROPOUT, training=self.training)
        x = self.conv2(x, edge_index)

        return x


# ============================================================
# 2. KGroot-inspired Model
# ============================================================

class KGrootModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.online_gcn = GCNConv(in_dim, hidden_dim)
        self.kg_gcn = GCNConv(in_dim, hidden_dim)

        self.sim_attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        online_h = F.relu(self.online_gcn(x, edge_index))
        kg_h = F.relu(self.kg_gcn(x, edge_index))

        sim_feature = torch.cat([online_h, kg_h], dim=1)
        alpha = torch.sigmoid(self.sim_attention(sim_feature))

        h = alpha * online_h + (1 - alpha) * kg_h
        h = torch.cat([h, torch.abs(online_h - kg_h)], dim=1)

        return self.classifier(h)


# ============================================================
# 3. KAFD Model
# GraphSAGE-based Knowledge Aggregation Fault Diagnosis
# ============================================================

class KAFDModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)

        self.knowledge_gate = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = F.relu(self.sage1(x, edge_index))
        h = F.dropout(h, p=DROPOUT, training=self.training)
        h = F.relu(self.sage2(h, edge_index))

        gate = torch.sigmoid(self.knowledge_gate(h))
        h = gate * h

        return self.classifier(h)


# ============================================================
# 4. LAMs Model
# Layerwise Attention Mechanism
# ============================================================

class LAMsModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.layer_attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        h3 = F.relu(self.conv3(h2, edge_index))

        layer_stack = torch.stack([h1, h2, h3], dim=1)

        score = self.layer_attention(layer_stack)
        alpha = torch.softmax(score, dim=1)

        h = torch.sum(alpha * layer_stack, dim=1)

        return self.classifier(h)


# ============================================================
# 5. CF-GNN Model
# Classical Fourier Graph Neural Network Approximation
# ============================================================

class CFGNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.low_filter = nn.Linear(in_dim, hidden_dim)
        self.high_filter = nn.Linear(in_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim * 2, out_dim)

    def spectral_filter(self, x, edge_index):
        num_nodes = x.size(0)

        adj = to_dense_adj(
            edge_index,
            max_num_nodes=num_nodes
        )[0]

        deg = torch.diag(adj.sum(dim=1))
        lap = deg - adj

        eigvals, eigvecs = torch.linalg.eigh(lap)

        x_hat = eigvecs.T @ x

        median_freq = torch.median(eigvals)

        low_mask = (eigvals <= median_freq).float().unsqueeze(1)
        high_mask = (eigvals > median_freq).float().unsqueeze(1)

        low_signal = eigvecs @ (low_mask * x_hat)
        high_signal = eigvecs @ (high_mask * x_hat)

        return low_signal, high_signal

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        low_signal, high_signal = self.spectral_filter(x, edge_index)

        h_low = F.relu(self.low_filter(low_signal))
        h_high = F.relu(self.high_filter(high_signal))

        h = torch.cat([h_low, h_high], dim=1)

        return self.classifier(h)


# ============================================================
# 6. HAN-inspired Model
# Node-level and Semantic-level Attention
# ============================================================

class HANInspiredModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.node_att1 = GATConv(
            in_dim,
            hidden_dim,
            heads=2,
            concat=False,
            dropout=DROPOUT
        )

        self.node_att2 = GATConv(
            in_dim,
            hidden_dim,
            heads=2,
            concat=False,
            dropout=DROPOUT
        )

        self.semantic_attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h1 = F.elu(self.node_att1(x, edge_index))
        h2 = F.elu(self.node_att2(x, edge_index))

        semantic_stack = torch.stack([h1, h2], dim=1)

        beta = torch.softmax(
            self.semantic_attention(semantic_stack),
            dim=1
        )

        h = torch.sum(beta * semantic_stack, dim=1)

        return self.classifier(h)


# ============================================================
# 7. HP-GCN Model
# Heterogeneous Propagation GCN with Attention
# ============================================================

class HPGCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.gcn = GCNConv(in_dim, hidden_dim)

        self.gat = GATConv(
            hidden_dim,
            hidden_dim,
            heads=2,
            concat=False,
            dropout=DROPOUT
        )

        self.neighbor_attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = F.relu(self.gcn(x, edge_index))
        h = F.dropout(h, p=DROPOUT, training=self.training)

        h = F.elu(self.gat(h, edge_index))

        alpha = torch.sigmoid(self.neighbor_attention(h))
        h = alpha * h

        return self.classifier(h)


# ============================================================
# 8. Proposed Attention-based KG Failure Localization Model
# ============================================================

class ProposedAttentionKGModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        self.feature_encoder = nn.Linear(in_dim, hidden_dim)

        self.relation_attention1 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=4,
            concat=False,
            dropout=DROPOUT
        )

        self.relation_attention2 = GATConv(
            hidden_dim,
            hidden_dim,
            heads=4,
            concat=False,
            dropout=DROPOUT
        )

        self.failure_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.residual = nn.Linear(in_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h0 = F.relu(self.feature_encoder(x))

        h1 = F.elu(self.relation_attention1(h0, edge_index))
        h2 = F.elu(self.relation_attention2(h1, edge_index))

        failure_score = self.failure_attention(h2)
        failure_weight = torch.sigmoid(failure_score)

        h_att = failure_weight * h2
        h_res = self.residual(x)

        h = torch.cat([h_att, h_res], dim=1)

        return self.classifier(h)


# ============================================================
# Training Function
# ============================================================

def train_model(model, data):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_macro = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        out = model(data)

        loss = F.cross_entropy(
            out[data.train_mask],
            data.y[data.train_mask]
        )

        loss.backward()
        optimizer.step()

        val_micro, val_macro, val_precision, val_recall, val_auc = evaluate(
            model,
            data,
            data.val_mask
        )

        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_micro, test_macro, test_precision, test_recall, test_auc = evaluate(
        model,
        data,
        data.test_mask
    )

    return test_micro, test_macro, test_precision, test_recall, test_auc


# ============================================================
# Experiment Runner
# ============================================================

def run_experiments():
    x, y = load_and_preprocess_data()

    edge_index, edge_weight = build_graph(
        x,
        k=K_NEIGHBORS
    )

    num_nodes = x.size(0)
    in_dim = x.size(1)
    out_dim = len(torch.unique(y))

    model_dict = {
        "KGroot": KGrootModel,
        "GCN": GCNModel,
        "KAFD": KAFDModel,
        "LAMs": LAMsModel,
        "CF-GNN": CFGNNModel,
        "HAN": HANInspiredModel,
        "HP-GCN": HPGCNModel,
        "Proposed-Att-KG": ProposedAttentionKGModel
    }

    final_results = {}

    for model_name, ModelClass in model_dict.items():
        print("\n" + "=" * 80)
        print(f"Training Model: {model_name}")
        print("=" * 80)

        micro_scores = []
        macro_scores = []
        precision_scores = []
        recall_scores = []
        auc_scores = []

        for run, seed in enumerate(SEEDS[:RUNS]):
            set_seed(seed)

            train_mask, val_mask, test_mask = create_masks(
                num_nodes,
                seed
            )

            data = Data(
                x=x,
                y=y,
                edge_index=edge_index,
                edge_weight=edge_weight,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            ).to(DEVICE)

            model = ModelClass(
                in_dim=in_dim,
                hidden_dim=HIDDEN_DIM,
                out_dim=out_dim
            ).to(DEVICE)

            micro, macro, precision, recall, auc = train_model(
                model,
                data
            )

            micro_scores.append(micro)
            macro_scores.append(macro)
            precision_scores.append(precision)
            recall_scores.append(recall)
            auc_scores.append(auc)

            print(
                f"Run {run + 1}: "
                f"Micro F1 = {micro:.4f}, "
                f"Macro F1 = {macro:.4f}, "
                f"Precision = {precision:.4f}, "
                f"Recall = {recall:.4f}, "
                f"AUC = {auc:.4f}"
            )

        final_results[model_name] = {
            "micro_mean": np.mean(micro_scores),
            "micro_std": np.std(micro_scores),

            "macro_mean": np.mean(macro_scores),
            "macro_std": np.std(macro_scores),

            "precision_mean": np.mean(precision_scores),
            "precision_std": np.std(precision_scores),

            "recall_mean": np.mean(recall_scores),
            "recall_std": np.std(recall_scores),

            "auc_mean": np.mean(auc_scores),
            "auc_std": np.std(auc_scores)
        }

    print("\n\n" + "=" * 115)
    print("Final Results: Mean ± Standard Deviation")
    print("=" * 115)

    print(
        f"{'Model':<22} "
        f"{'Micro F1':<18} "
        f"{'Macro F1':<18} "
        f"{'Precision':<18} "
        f"{'Recall':<18} "
        f"{'AUC':<18}"
    )

    print("-" * 115)

    for model_name, result in final_results.items():
        print(
            f"{model_name:<22} "
            f"{result['micro_mean']:.4f} ± {result['micro_std']:.4f}   "
            f"{result['macro_mean']:.4f} ± {result['macro_std']:.4f}   "
            f"{result['precision_mean']:.4f} ± {result['precision_std']:.4f}   "
            f"{result['recall_mean']:.4f} ± {result['recall_std']:.4f}   "
            f"{result['auc_mean']:.4f} ± {result['auc_std']:.4f}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_experiments()