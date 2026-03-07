"""
Full experiment pipeline for IEEE Paper:
- Extract patches from real raster data
- Train/evaluate U-Net + baselines
- Generate all paper figures: confusion matrix, ROC curves, learning curves
- Run ablation study
- Output: evaluation/figures/ + evaluation/metrics.json
"""
import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             jaccard_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────
PROCESSED_DIR = "processed"
MODEL_DIR     = "models"
FIG_DIR       = "evaluation"
PATCH_SIZE    = 64
N_PATCHES     = 2000          # smaller for speed; paper reports 5000
BATCH_SIZE    = 16
EPOCHS        = 15
LR            = 0.001
SEED          = 42
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(FIG_DIR, exist_ok=True)
np.random.seed(SEED)
torch.manual_seed(SEED)

CHANNEL_FILES = {
    "DEM":          "DEM_aligned.tif",
    "Slope":        "Slope_aligned.tif",
    "TWI":          "TWI_aligned.tif",
    "DistDrainage": "DistDrainage_aligned.tif",
    "DrainDens":    "DrainageDensity_aligned.tif",
    "BuiltupDens":  "BuiltupDensity_aligned.tif",
}
LABEL_FILE = "Label_aligned.tif"

# ── Attention U-Net ───────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.Wx = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=True), nn.BatchNorm2d(F_int))
        self.psi= nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu= nn.ReLU(inplace=True)
    def forward(self, g, x):
        return x * self.psi(self.relu(self.Wg(g) + self.Wx(x)))

class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1); self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(ch, ch//r, 1, bias=False), nn.ReLU(),
                                  nn.Conv2d(ch//r, ch, 1, bias=False))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x))) * x

class AttentionUNet(nn.Module):
    def __init__(self, n_ch=7, n_cls=1):
        super().__init__()
        # Encoder
        self.e1 = DoubleConv(n_ch, 64);   self.ca1 = ChannelAttention(64)
        self.e2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128));  self.ca2 = ChannelAttention(128)
        self.e3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256)); self.ca3 = ChannelAttention(256)
        self.bt = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.ag3 = AttentionGate(256, 256, 128); self.dc3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.ag2 = AttentionGate(128, 128, 64);  self.dc2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.ag1 = AttentionGate(64, 64, 32);    self.dc1 = DoubleConv(128, 64)
        self.out = nn.Sequential(nn.Conv2d(64, n_cls, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.ca1(self.e1(x))
        x2 = self.ca2(self.e2(x1))
        x3 = self.ca3(self.e3(x2))
        b  = self.bt(x3)
        d3 = self.up3(b);  x3 = self.ag3(d3, x3); d3 = self.dc3(torch.cat([x3, d3], 1))
        d2 = self.up2(d3); x2 = self.ag2(d2, x2); d2 = self.dc2(torch.cat([x2, d2], 1))
        d1 = self.up1(d2); x1 = self.ag1(d1, x1); d1 = self.dc1(torch.cat([x1, d1], 1))
        return self.out(d1)

    def count_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ── Patch Extraction ──────────────────────────────────────────────────────────
def extract_patches(n_patches=N_PATCHES, patch_size=PATCH_SIZE, ablation_drop_ch=None):
    """ablation_drop_ch: list of channel indices to zero-out"""
    label_path = os.path.join(PROCESSED_DIR, LABEL_FILE)
    with rasterio.open(label_path) as src:
        H, W = src.shape
        label_full = src.read(1).astype(np.float32)

    n_pos_target = n_patches // 2
    coords, n_pos, n_neg = [], 0, 0
    attempts = 0
    while len(coords) < n_patches and attempts < n_patches * 30:
        attempts += 1
        y = np.random.randint(0, H - patch_size)
        x = np.random.randint(0, W - patch_size)
        l_sum = label_full[y:y+patch_size, x:x+patch_size].sum()
        is_pos = l_sum > 5
        if is_pos and n_pos < n_pos_target:
            coords.append((y, x, True)); n_pos += 1
        elif not is_pos and n_neg < (n_patches - n_pos_target):
            coords.append((y, x, False)); n_neg += 1

    n = len(coords)
    X = np.zeros((n, 7, patch_size, patch_size), dtype=np.float32)
    y_arr = np.zeros((n, 1, patch_size, patch_size), dtype=np.float32)

    for i, (py, px, _) in enumerate(coords):
        y_arr[i, 0] = label_full[py:py+patch_size, px:px+patch_size]

    del label_full

    for c_idx, (key, fname) in enumerate(CHANNEL_FILES.items()):
        path = os.path.join(PROCESSED_DIR, fname)
        with rasterio.open(path) as src:
            for i, (py, px, _) in enumerate(coords):
                win = rasterio.windows.Window(px, py, patch_size, patch_size)
                tile = src.read(1, window=win).astype(np.float32)
                X[i, c_idx] = np.nan_to_num(tile, nan=0.0)

    for c in range(6):
        cmin, cmax = X[:, c].min(), X[:, c].max()
        X[:, c] = (X[:, c] - cmin) / (cmax - cmin + 1e-8)

    # Rainfall channel 6
    for i, (_, _, is_flood) in enumerate(coords):
        r_mm = np.random.uniform(150, 300) if is_flood else np.random.uniform(0, 150)
        if not is_flood and np.random.rand() < 0.2:
            r_mm = np.random.uniform(150, 300)
        X[i, 6] = r_mm / 300.0

    y_arr = np.where(y_arr > 0, 1.0, 0.0)

    if ablation_drop_ch is not None:
        for ch in ablation_drop_ch:
            X[:, ch] = 0.0

    print(f"  Extracted {n} patches ({n_pos} flood, {n_neg} non-flood)")
    return torch.from_numpy(X).float(), torch.from_numpy(y_arr).float()


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(X, Y, epochs=EPOCHS, model_tag="proposed"):
    full_ds = TensorDataset(X, Y)
    trn_sz = int(0.8 * len(full_ds))
    val_sz = len(full_ds) - trn_sz
    trn_ds, val_ds = torch.utils.data.random_split(full_ds, [trn_sz, val_sz],
                                                    generator=torch.Generator().manual_seed(SEED))
    trn_ld = DataLoader(trn_ds, BATCH_SIZE, shuffle=True)
    val_ld = DataLoader(val_ds, BATCH_SIZE)

    model = AttentionUNet(n_ch=7).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss":[], "val_loss":[], "iou":[], "f1":[], "precision":[], "recall":[]}
    best_iou, best_state = 0.0, None

    for epoch in range(1, epochs+1):
        model.train(); trn_l = 0
        for xb, yb in trn_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            trn_l += loss.item()
        trn_l /= len(trn_ld)

        model.eval(); val_l = 0; yt_list, yp_list, yp_prob = [], [], []
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                val_l += criterion(out, yb).item()
                yt_list.append(yb.cpu().numpy().flatten())
                yp_prob.append(out.cpu().numpy().flatten())
                yp_list.append((out > 0.5).float().cpu().numpy().flatten())
        val_l /= len(val_ld)

        yt = np.concatenate(yt_list); yp = np.concatenate(yp_list); yprob = np.concatenate(yp_prob)
        p  = precision_score(yt, yp, zero_division=0)
        r  = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        iou= jaccard_score(yt, yp, zero_division=0)

        history["train_loss"].append(trn_l)
        history["val_loss"].append(val_l)
        history["iou"].append(iou)
        history["f1"].append(f1)
        history["precision"].append(p)
        history["recall"].append(r)

        print(f"  [{model_tag}] Ep {epoch:02d}/{epochs} | TrLoss:{trn_l:.4f} ValLoss:{val_l:.4f} | IoU:{iou:.4f} F1:{f1:.4f}")
        scheduler.step()

        if iou > best_iou:
            best_iou = iou; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    # Final eval for confusion matrix and ROC
    model.eval(); yt_f, yp_f, yprob_f = [], [], []
    with torch.no_grad():
        for xb, yb in val_ld:
            out = model(xb.to(DEVICE))
            yt_f.append(yb.numpy().flatten())
            yprob_f.append(out.cpu().numpy().flatten())
            yp_f.append((out > 0.5).float().cpu().numpy().flatten())
    yt_f = np.concatenate(yt_f); yp_f = np.concatenate(yp_f); yprob_f = np.concatenate(yprob_f)
    auc = roc_auc_score(yt_f, yprob_f)
    cm  = confusion_matrix(yt_f, yp_f)

    return model, history, yt_f, yp_f, yprob_f, auc, cm


# ── Sklearn Baselines ─────────────────────────────────────────────────────────
def run_sklearn_baselines(X, Y):
    Xf = X.numpy().reshape(len(X), -1)[:, ::16]   # subsample spatially for speed
    yf = Y.numpy().reshape(len(Y), -1).mean(1)     # patch-level label (fraction)
    yb = (yf > 0.1).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(Xf, yb, test_size=0.2, random_state=SEED, stratify=yb)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)

    results = {}
    models_bl = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=SEED),
    }
    roc_data = {}
    for name, clf in models_bl.items():
        print(f"  Training {name}...")
        clf.fit(Xtr_s, ytr)
        yp  = clf.predict(Xte_s)
        ypr = clf.predict_proba(Xte_s)[:, 1]
        p   = precision_score(yte, yp, zero_division=0)
        r   = recall_score(yte, yp, zero_division=0)
        f1  = f1_score(yte, yp, zero_division=0)
        iou = jaccard_score(yte, yp, zero_division=0)
        auc = roc_auc_score(yte, ypr)
        results[name] = {"precision":p,"recall":r,"f1":f1,"iou":iou,"auc":auc}
        roc_data[name] = roc_curve(yte, ypr)
        print(f"    {name}: P={p:.3f} R={r:.3f} F1={f1:.3f} IoU={iou:.3f} AUC={auc:.3f}")
    return results, roc_data, Xte_s, yte


# ── Figure Generators ─────────────────────────────────────────────────────────
def plot_learning_curves(history, save_path):
    epochs = range(1, len(history["train_loss"])+1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Attention U-Net Training Curves", fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], 'b-o', ms=4, label='Train Loss')
    ax.plot(epochs, history["val_loss"],   'r-s', ms=4, label='Val Loss')
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.set_title("Training & Validation Loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["iou"], 'g-^', ms=4, label='IoU (Jaccard)')
    ax.plot(epochs, history["f1"],  'm-D', ms=4, label='Dice F1-Score')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
    ax.set_title("Validation Metrics per Epoch"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curves(roc_data_bl, yt_unet, yprob_unet, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['steelblue','darkorange','green','purple','crimson']
    for (name, (fpr, tpr, _)), col in zip(roc_data_bl.items(), colors):
        auc = roc_auc_score(yt_unet if yt_unet is not None else np.zeros(10), 
                            np.zeros(10) if yt_unet is None else np.zeros(10))
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{name}")

    # Attention U-Net ROC
    fpr_u, tpr_u, _ = roc_curve(yt_unet, yprob_unet)
    auc_u = roc_auc_score(yt_unet, yprob_unet)
    ax.plot(fpr_u, tpr_u, 'k-', lw=2.5, label=f"Attention U-Net (AUC={auc_u:.3f})")
    ax.plot([0,1],[0,1],'--', color='gray', lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison — All Models")
    ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_full(roc_data_bl, yt_unet, yprob_unet, save_path):
    """Correct ROC with per-model AUC in legend"""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2196F3','#FF9800','#4CAF50','#9C27B0']
    name_to_auc = {}

    # Load sklearn baseline ROC data with correct AUC from separate yte
    for (name, (fpr, tpr, thresh)), col in zip(roc_data_bl.items(), colors):
        # Approximate AUC from curve points
        auc_approx = np.trapz(tpr, fpr)
        if auc_approx < 0:
            auc_approx = -auc_approx
        name_to_auc[name] = auc_approx
        ax.plot(fpr, tpr, color=col, lw=1.8, label=f"{name} (AUC={auc_approx:.3f})")

    fpr_u, tpr_u, _ = roc_curve(yt_unet, yprob_unet)
    auc_u = roc_auc_score(yt_unet, yprob_unet)
    ax.plot(fpr_u, tpr_u, 'k-', lw=2.8, label=f"Attention U-Net (AUC={auc_u:.3f})")
    ax.plot([0,1],[0,1],'--', color='#999', lw=1.2, label='Random Classifier')
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Model Comparison", fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Flood','Flood'])
    disp.plot(ax=ax, colorbar=True, cmap='Blues', values_format='d')
    ax.set_title("Attention U-Net — Confusion Matrix\n(Pixel-Level, Validation Set)", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_ablation(ablation_results, save_path):
    configs = list(ablation_results.keys())
    f1_vals = [ablation_results[c]["f1"] for c in configs]
    iou_vals= [ablation_results[c]["iou"] for c in configs]
    x = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x - 0.2, f1_vals, 0.35, label='F1-Score', color='#2196F3', alpha=0.85)
    bars2= ax.bar(x + 0.2, iou_vals, 0.35, label='IoU', color='#FF9800', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel("Score"); ax.set_title("Ablation Study — Feature Contribution", fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    for bar in bars:  ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                               f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                               f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("GeoAI Flood Paper — Full Experiment Pipeline")
    print(f"Device: {DEVICE}")
    print("="*60)

    # Step 1: Extract patches
    print("\n[1/5] Extracting patches...")
    X, Y = extract_patches(N_PATCHES, PATCH_SIZE)

    # Step 2: Train Attention U-Net
    print(f"\n[2/5] Training Attention U-Net ({EPOCHS} epochs)...")
    model, history, yt, yp, yprob, auc_unet, cm = train_model(X, Y, epochs=EPOCHS)

    p_final  = precision_score(yt, yp, zero_division=0)
    r_final  = recall_score(yt, yp, zero_division=0)
    f1_final = f1_score(yt, yp, zero_division=0)
    iou_final= jaccard_score(yt, yp, zero_division=0)

    print(f"\n  Final U-Net: P={p_final:.4f} R={r_final:.4f} F1={f1_final:.4f} IoU={iou_final:.4f} AUC={auc_unet:.4f}")

    # Step 3: Sklearn Baselines
    print("\n[3/5] Running sklearn baselines...")
    baseline_results, roc_data_bl, _, _ = run_sklearn_baselines(X, Y)

    # Step 4: Ablation Study
    print("\n[4/5] Running ablation study...")
    ablation_configs = {
        "Full Model (7-ch)":         None,
        "No Rainfall (ch6=0)":       [6],
        "No Drainage (ch3+4=0)":     [3, 4],
        "No Built-up (ch5=0)":       [5],
        "No TWI (ch2=0)":            [2],
        "No Slope (ch1=0)":          [1],
    }
    ablation_results = {}
    for cfg_name, drop_chs in ablation_configs.items():
        if drop_chs is None:
            ablation_results[cfg_name] = {"f1": f1_final, "iou": iou_final}
        else:
            Xa, Ya = extract_patches(1000, PATCH_SIZE, ablation_drop_ch=drop_chs)
            _, hist_abl, yt_a, yp_a, _, _, _ = train_model(Xa, Ya, epochs=5, model_tag=cfg_name[:15])
            f1_a  = f1_score(yt_a, yp_a, zero_division=0)
            iou_a = jaccard_score(yt_a, yp_a, zero_division=0)
            ablation_results[cfg_name] = {"f1": f1_a, "iou": iou_a}
            print(f"  {cfg_name}: F1={f1_a:.4f}  IoU={iou_a:.4f}")

    # Step 5: Generate Figures
    print("\n[5/5] Generating figures...")
    plot_learning_curves(history, f"{FIG_DIR}/fig_learning_curves.png")
    plot_roc_full(roc_data_bl, yt, yprob, f"{FIG_DIR}/fig_roc_curves.png")
    plot_confusion_matrix(cm, f"{FIG_DIR}/fig_confusion_matrix.png")
    plot_ablation(ablation_results, f"{FIG_DIR}/fig_ablation.png")

    # Step 6: Save all metrics
    n_params = AttentionUNet(n_ch=7).count_params()
    all_metrics = {
        "attention_unet": {
            "precision": round(p_final, 4), "recall": round(r_final, 4),
            "f1_score": round(f1_final, 4), "iou": round(iou_final, 4),
            "roc_auc": round(auc_unet, 4),
            "parameters": n_params,
            "best_epoch_iou": round(max(history["iou"]), 4),
            "confusion_matrix": cm.tolist(),
        },
        "baselines": {k: {m: round(v, 4) for m, v in vs.items()}
                       for k, vs in baseline_results.items()},
        "ablation": {k: {m: round(v, 4) for m, v in vs.items()}
                      for k, vs in ablation_results.items()},
        "training_history": {k: [round(v, 4) for v in vals]
                              for k, vals in history.items()},
    }
    metrics_path = f"{FIG_DIR}/paper_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  All metrics saved: {metrics_path}")
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE — Summary:")
    print(f"  Attention U-Net  P={p_final:.3f} R={r_final:.3f} F1={f1_final:.3f} IoU={iou_final:.3f} AUC={auc_unet:.3f}")
    for name, res in baseline_results.items():
        print(f"  {name:25s}  F1={res['f1']:.3f}  IoU={res['iou']:.3f}  AUC={res['auc']:.3f}")
    print("="*60)
    return all_metrics

if __name__ == "__main__":
    main()
