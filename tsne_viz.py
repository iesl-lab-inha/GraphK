# tsne_viz.py
# -*- coding: utf-8 -*-
import os, datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

IDX2NAME_IEMOCAP = {
    0: "Angry",
    1: "Happy",
    2: "Sad",
    3: "Neutral",
    4: "Excited",
    5: "Frustrated",
}
LEGEND_IEMOCAP = ["Happy", "Sad", "Neutral", "Angry", "Excited", "Frustrated"]

IDX2NAME_MELD = {
    0: "Neutral",
    1: "Surprise",
    2: "Fear",
    3: "Sadness",
    4: "Joy",
    5: "Disgust",
    6: "Anger",
}
LEGEND_MELD = ["Neutral", "Surprise", "Fear", "Sadness", "Joy", "Disgust", "Anger"]

def _ensure_dir(d): os.makedirs(d, exist_ok=True); return d
def _timestamp(): return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def _to_2d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3:  # (U,F,D)
        return t.mean(dim=1)
    return t

def _mask_iemocap(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 3:
        return mask.squeeze(1).squeeze(-1).bool()
    if mask.dim() == 2:
        return mask.squeeze(1).bool()
    return mask.bool()

def _mask_meld(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        return mask.squeeze(1).bool()
    return mask.bool()

def _tsne_2d(X: np.ndarray, seed: int = 42) -> np.ndarray:
    Xs = StandardScaler().fit_transform(X)
    d  = min(50, Xs.shape[1])
    Xp = PCA(n_components=d, random_state=seed).fit_transform(Xs)
    perplexity = min(30, max(5, len(Xp)//100))
    Z = TSNE(
        n_components=2,
        init="pca",
        learning_rate=200,
        perplexity=perplexity,
        n_iter=1000,
        random_state=seed,
    ).fit_transform(Xp)
    return Z

def _plot_single_with_names(
    Z: np.ndarray, y: np.ndarray, idx2name: dict, legend_order: list,
    title: str, save_path: str,
    edge_width: float = 0.8,
    point_size: int = 12,
    legend_fontsize: int = 18,
    title_fontsize: int = 14
):
    colors = ['#e74c3c','#f1c40f','#2ecc71','#1abc9c','#3498db','#9b59b6','#e91e63',
              '#34495e','#16a085','#8e44ad']
    name2idx = {v: k for k, v in idx2name.items()}
    y = np.asarray(y).reshape(-1)

    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    for i, name in enumerate(legend_order):
        if name not in name2idx:
            continue
        k = name2idx[name]
        m = (y == k)
        if not np.any(m):
            continue
        ax.scatter(
            Z[m, 0], Z[m, 1],
            s=point_size,
            alpha=0.9,
            color=colors[i % len(colors)],
            label=name
        )

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=title_fontsize)
    leg = ax.legend(markerscale=2, frameon=False)
    for txt in leg.get_texts():
        txt.set_fontsize(legend_fontsize)
        txt.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 保存: {save_path}")


def _extract_multi_from_dataset(test_dataset, dataset_name: str):
    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=getattr(test_dataset, "collate_fn", None),
    )
    feats, labels = [], []

    is_iemocap = dataset_name in ["IEMOCAP", "IEMOCAP4"]
    for batch in loader:
        t0, t1, t2, t3, vis, aud, spk, mask, emo, senti, vid = batch
        t0=t0.squeeze(1); t1=t1.squeeze(1); t2=t2.squeeze(1); t3=t3.squeeze(1)
        vis=vis.squeeze(1); aud=aud.squeeze(1); spk=spk.squeeze(1)
        mask = _mask_iemocap(mask) if is_iemocap else _mask_meld(mask)
        emo  = emo.squeeze(1)

        t0, t1, t2, t3 = t0[mask], t1[mask], t2[mask], t3[mask]
        vis, aud      = vis[mask], aud[mask]
        y = emo[mask].cpu().numpy().reshape(-1)

        t0 = _to_2d(t0); t1 = _to_2d(t1); t2 = _to_2d(t2); t3 = _to_2d(t3)
        vis = _to_2d(vis); aud = _to_2d(aud)

        if t0.shape[0] == 0:
            continue

        x = torch.cat([t0, t1, t2, t3, aud, vis], dim=-1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        feats.append(x.cpu().numpy())
        labels.append(y)

    if not feats:
        raise RuntimeError("[TSNE] 未从测试集抽到有效输入特征，请检查数据与 mask。")

    X = np.vstack(feats)           # (N,D)
    y = np.concatenate(labels, 0)  # (N,)
    return X, y

def save_final_tsne_plots(
    args_dataset: str,
    test_loader,
    save_root: str = "tsne_plots",
    learned_feats=None,
    learned_labels=None,
    max_samples: int = None,
    seed: int = 42,
    idx2name_override: dict = None,
    legend_override: list = None,
):
    if args_dataset in ["IEMOCAP", "IEMOCAP4"]:
        idx2name = idx2name_override or IDX2NAME_IEMOCAP
        legend   = legend_override or LEGEND_IEMOCAP
    elif args_dataset == "MELD":
        idx2name = idx2name_override or IDX2NAME_MELD
        legend   = legend_override or LEGEND_MELD
    else:
        print(f"[TSNE] 暂未支持数据集 {args_dataset}。"); return

    outdir = _ensure_dir(os.path.join(save_root, args_dataset))
    ts = _timestamp()

    print("[TSNE] 抽取并可视化 输入特征（multi）...")
    X_in, y_in = _extract_multi_from_dataset(test_loader.dataset, args_dataset)
    if (max_samples is not None) and (len(X_in) > max_samples):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_in), size=max_samples, replace=False)
        X_in, y_in = X_in[idx], y_in[idx]
    Z_in = _tsne_2d(X_in, seed=seed)
    _plot_single_with_names(
        Z_in, y_in, idx2name, legend,
        title="",
        save_path=os.path.join(outdir, f"{args_dataset}_input_multi_{ts}.png")
    )

    if (learned_feats is not None) and (learned_labels is not None):
        print("[TSNE] 可视化 模型学到的特征...")
        chunks = []
        for f in learned_feats if isinstance(learned_feats, (list, tuple)) else [learned_feats]:
            if f is None: continue
            if isinstance(f, torch.Tensor): f = f.detach().cpu().numpy()
            f = np.asarray(f)
            if f.ndim == 1: f = f[:, None]
            if f.ndim == 3:  # (B,T,D) → (N,D)
                f = f.reshape(-1, f.shape[-1])
            chunks.append(f)
        if chunks:
            X_feat = np.vstack(chunks)
            y_feat = np.asarray(learned_labels).reshape(-1)
            n = min(len(X_feat), len(y_feat))
            X_feat, y_feat = X_feat[:n], y_feat[:n]
            if (max_samples is not None) and (len(X_feat) > max_samples):
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(X_feat), size=max_samples, replace=False)
                X_feat, y_feat = X_feat[idx], y_feat[idx]
            Z_feat = _tsne_2d(X_feat, seed=seed)
            _plot_single_with_names(
                Z_feat, y_feat, idx2name, legend,
                title="",
                save_path=os.path.join(outdir, f"{args_dataset}_learned_{ts}.png")
            )
        else:
            print("[TSNE] 未提供有效 learned_feats，跳过。")
    else:
        print("[TSNE] 未提供 learned_feats/learned_labels，跳过模型特征图。")
