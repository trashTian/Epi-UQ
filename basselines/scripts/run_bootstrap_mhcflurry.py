#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from mhcflurry import Class1AffinityPredictor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_ORDER = ["ACC", "F1", "Recall", "MCC", "Precision", "Specificity", "AUC"]


def resolve_mhcflurry_pan_models_dir(user_path: Path) -> Path:
    """
    MHCflurry pan download layout is typically:
      <downloads>/models_class1_pan/models.combined/manifest.csv
    Users may pass either the inner dir (with manifest) or the outer models_class1_pan.
    """
    md = user_path.resolve()
    if (md / "manifest.csv").is_file():
        return md
    for sub in ("models.combined", "models", "combined"):
        cand = md / sub
        if (cand / "manifest.csv").is_file():
            return cand
    raise FileNotFoundError(
        f"No manifest.csv under {md} (tried {md}, {md}/models.combined, {md}/models, {md}/combined). "
        "Point --mhcflurry_models_dir at the directory that contains manifest.csv "
        "(often .../models_class1_pan/models.combined)."
    )


def infer_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Column not found. candidates={candidates}, actual={list(df.columns)}")
    return None


def affinity_to_score(affinity_nm: np.ndarray) -> np.ndarray:
    aff = np.clip(affinity_nm.astype(np.float64), 1.0, 50000.0)
    return 1.0 - (np.log(aff) / np.log(50000.0))


def binary_metrics_from_ic50(y_true: np.ndarray, y_affinity_nm: np.ndarray, ic50_threshold: float = 500.0) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_affinity_nm = np.asarray(y_affinity_nm, dtype=np.float64)
    y_score = affinity_to_score(y_affinity_nm)
    y_pred = (y_affinity_nm < ic50_threshold).astype(np.int64)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_score))

    return {
        "ACC": float(acc),
        "F1": float(f1),
        "Recall": float(rec),
        "MCC": float(mcc),
        "Precision": float(pre),
        "Specificity": spec,
        "AUC": auc,
    }


def bootstrap_once(
    y_true: np.ndarray,
    y_affinity_nm: np.ndarray,
    rng: np.random.Generator,
    ic50_threshold: float,
) -> dict:
    n = len(y_true)
    idx = rng.integers(0, n, size=n)
    yt = y_true[idx]
    ya = y_affinity_nm[idx]
    return binary_metrics_from_ic50(yt, ya, ic50_threshold=ic50_threshold)


def format_mean_std(values: np.ndarray) -> tuple[float, float, str]:
    mean_v = float(np.nanmean(values))
    std_v = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_v, std_v, f"{mean_v:.4f}\u00b1{std_v:.4f}"


def load_allele_panel(path: Path) -> list[str]:
    try:
        panel_df = pd.read_csv(path)
        if panel_df.shape[1] == 1:
            vals = panel_df.iloc[:, 0].astype(str).str.strip().tolist()
        else:
            c = infer_col(panel_df, ["Allele", "allele", "HLA", "hla"])
            vals = panel_df[c].astype(str).str.strip().tolist()
    except Exception:
        panel_df = pd.read_csv(path, header=None)
        vals = panel_df.iloc[:, 0].astype(str).str.strip().tolist()
    vals = [v for v in vals if v]
    if not vals:
        raise ValueError(f"No allele loaded from panel file: {path}")
    return list(dict.fromkeys(vals))


def main() -> None:
    p = argparse.ArgumentParser(description="MHCflurry inference + bootstrap metrics")
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_bootstrap", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allele_col", type=str, default=None)
    p.add_argument("--peptide_col", type=str, default=None)
    p.add_argument("--label_col", type=str, default=None)
    p.add_argument("--peptide_only", action="store_true", help="测试集仅含肽段与标签时启用")
    p.add_argument("--default_allele", type=str, default="HLA-A*02:01", help="peptide_only 模式下统一使用的等位基因")
    p.add_argument("--allele_panel_csv", type=Path, default=None, help="提供 HLA 集合文件，逐个 HLA 预测后聚合")
    p.add_argument("--aggregate", type=str, default="max", choices=["max"], help="多 HLA 聚合方式")
    p.add_argument("--max_alleles", type=int, default=0, help=">0 时仅使用前 N 个 allele（调试/提速）")
    p.add_argument("--ic50_threshold", type=float, default=500.0, help="IC50<nM 判定为正类，默认 500")
    p.add_argument(
        "--mhcflurry_models_dir",
        type=Path,
        default=None,
        help="Class1 pan 模型目录（须含 manifest.csv）。可为 .../models_class1_pan 或 "
        "其下的 models.combined 子目录；脚本会自动尝试 models.combined/",
    )
    args = p.parse_args()

    df = pd.read_csv(args.test_csv)

    allele_col = args.allele_col or infer_col(df, ["Allele", "allele", "mhc", "hla", "HLA"], required=False)
    peptide_col = args.peptide_col or infer_col(df, ["Epitope.1", "peptide", "Peptide", "epitope", "sequence"])
    label_col = args.label_col or infer_col(df, ["Label", "label", "binder", "target", "class", "y"])

    use_panel = args.allele_panel_csv is not None
    use_default_allele = (not use_panel) and (args.peptide_only or (allele_col is None))

    if use_panel:
        work = df[[peptide_col, label_col]].copy()
    elif use_default_allele:
        work = df[[peptide_col, label_col]].copy()
        work["__allele__"] = args.default_allele
        allele_col = "__allele__"
    else:
        work = df[[allele_col, peptide_col, label_col]].copy()

    required_cols = [peptide_col, label_col] if use_panel else [allele_col, peptide_col, label_col]
    work = work.dropna(subset=required_cols).reset_index(drop=True)
    work[peptide_col] = work[peptide_col].astype(str).str.strip().str.upper()
    if not use_panel:
        work[allele_col] = work[allele_col].astype(str).str.strip()
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work = work.dropna(subset=[label_col])
    work = work[work[label_col].isin([0, 1])]
    work[label_col] = work[label_col].astype(int)
    work = work[work[peptide_col].str.len().between(8, 15)]

    if len(work) == 0:
        raise ValueError("No valid rows after filtering.")
    if use_panel:
        print(f"[INFO] panel mode enabled. allele_panel={args.allele_panel_csv}")
    elif use_default_allele:
        print(f"[INFO] peptide_only mode enabled. Use default allele: {args.default_allele}")

    if args.mhcflurry_models_dir is not None:
        md = resolve_mhcflurry_pan_models_dir(args.mhcflurry_models_dir)
        print(f"[INFO] loading MHCflurry pan models from {md}", flush=True)
        predictor = Class1AffinityPredictor.load(models_dir=str(md))
    else:
        predictor = Class1AffinityPredictor.load()
    peptides = work[peptide_col].tolist()
    if use_panel:
        panel = load_allele_panel(args.allele_panel_csv)
        if args.max_alleles > 0:
            panel = panel[: args.max_alleles]
        print(f"[INFO] using {len(panel)} alleles for aggregation={args.aggregate}")
        y_affinity_all = np.full(len(peptides), np.inf, dtype=np.float64)
        for i, allele in enumerate(panel, start=1):
            aff = predictor.predict(peptides=peptides, alleles=[allele] * len(peptides))
            aff = np.asarray(aff, dtype=np.float64)
            y_affinity_all = np.minimum(y_affinity_all, aff)
            if i % 20 == 0 or i == len(panel):
                print(f"[INFO] processed alleles: {i}/{len(panel)}")
    else:
        y_affinity_all = predictor.predict(
            peptides=peptides,
            alleles=work[allele_col].tolist(),
        )
        y_affinity_all = np.asarray(y_affinity_all, dtype=np.float64)
    y_true_all = work[label_col].to_numpy(dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for i in range(args.n_bootstrap):
        m = bootstrap_once(y_true_all, y_affinity_all, rng, ic50_threshold=args.ic50_threshold)
        m["bootstrap_id"] = i + 1
        rows.append(m)

    per_run = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    for metric in METRIC_ORDER:
        vals = per_run[metric].astype(float).to_numpy()
        mean_v, std_v, ms = format_mean_std(vals)
        summary_rows.append(
            {
                "Metric": metric,
                "Mean": mean_v,
                "Std": std_v,
                "Mean±Std": ms,
            }
        )
    summary = pd.DataFrame(summary_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = args.out_dir / "MHCflurry_bootstrap_per_run.csv"
    summary_path = args.out_dir / "MHCflurry_bootstrap_summary.csv"
    txt_path = args.out_dir / "MHCflurry_bootstrap_summary.txt"

    per_run.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        for _, r in summary.iterrows():
            f.write(f"{r['Metric']}: {r['Mean±Std']}\n")

    print(f"Loaded rows: {len(work)}")
    print(f"Saved: {per_run_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {txt_path}")
    print("\nMean±Std:")
    for _, r in summary.iterrows():
        print(f"{r['Metric']}: {r['Mean±Std']}")


if __name__ == "__main__":
    main()
