#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
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
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MHCNUGGETS_CLASS = "II"


def infer_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Column not found. candidates={candidates}, actual={list(df.columns)}")
    return None


def to_mhcnuggets_ii_allele(allele: str) -> str | None:
    if allele is None:
        return None
    s = str(allele).strip().upper()
    if not s:
        return None
    s = s.replace("*", "").replace(":", "")
    s = s.replace("/", "-")
    if not s.startswith("HLA-"):
        s = "HLA-" + s
    if len(s) < 8:
        return None
    return s


def affinity_to_score(affinity_nm: np.ndarray) -> np.ndarray:
    aff = np.clip(affinity_nm.astype(np.float64), 1.0, 50000.0)
    return 1.0 - (np.log(aff) / np.log(50000.0))


def score_to_affinity(score: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(score, dtype=np.float64), 0.0, 1.0)
    return np.power(50000.0, 1.0 - s)


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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) >= 2 else float("nan")

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
    rng: np.random.RandomState,
    ic50_threshold: float,
) -> dict:
    n = len(y_true)
    idx = rng.randint(0, n, size=n)
    yt = y_true[idx]
    ya = y_affinity_nm[idx]
    return binary_metrics_from_ic50(yt, ya, ic50_threshold=ic50_threshold)


def format_mean_std(values: np.ndarray) -> tuple[float, float, str]:
    mean_v = float(np.nanmean(values))
    std_v = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_v, std_v, f"{mean_v:.4f}\u00b1{std_v:.4f}"


def load_allele_panel(path: Path) -> list[str]:
    try:
        sniff = pd.read_csv(path, header=None, nrows=5)
        if sniff.shape[1] == 1:
            panel_df = pd.read_csv(path, header=None)
            vals = panel_df.iloc[:, 0].astype(str).str.strip().tolist()
        else:
            panel_df = pd.read_csv(path)
            if panel_df.shape[1] == 1:
                vals = panel_df.iloc[:, 0].astype(str).str.strip().tolist()
            else:
                c = infer_col(panel_df, ["Allele", "allele", "HLA", "hla"], required=False)
                if c is not None:
                    vals = panel_df[c].astype(str).str.strip().tolist()
                else:
                    vals = []
                    for col in panel_df.columns:
                        vals.extend(panel_df[col].dropna().astype(str).str.strip().tolist())
    except Exception:
        panel_df = pd.read_csv(path, header=None)
        vals = panel_df.iloc[:, 0].astype(str).str.strip().tolist()
    vals = [v for v in vals if v and str(v).lower() != "nan"]
    if not vals:
        raise ValueError(f"No allele loaded from panel file: {path}")
    return list(dict.fromkeys(vals))


def run_mhcnuggets_predict(df_in: pd.DataFrame, peptide_col: str, allele_col: str) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="mhcnuggetsII_boot_") as td:
        td_path = Path(td)
        work = df_in[[peptide_col, allele_col]].copy().reset_index(drop=True)
        work["__orig_idx__"] = np.arange(len(work))
        all_affinity = np.full(len(work), np.nan, dtype=np.float64)

        exe = shutil.which("mhcnuggets-predict") or shutil.which("mhcnuggets_predict")

        for allele, grp in work.groupby(allele_col, sort=False):
            pep_txt = td_path / f"peptides_{re.sub(r'[^A-Za-z0-9]+', '_', str(allele))}.txt"
            out_csv = td_path / f"pred_{re.sub(r'[^A-Za-z0-9]+', '_', str(allele))}.csv"
            grp[peptide_col].to_csv(pep_txt, index=False, header=False)

            if exe is not None:
                cmd = [exe, "-c", MHCNUGGETS_CLASS, "-p", str(pep_txt), "-a", str(allele), "-o", str(out_csv)]
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "mhcnuggets.src.predict",
                    "-c",
                    MHCNUGGETS_CLASS,
                    "-p",
                    str(pep_txt),
                    "-a",
                    str(allele),
                    "-o",
                    str(out_csv),
                ]
            subprocess.check_call(cmd)

            pred = pd.read_csv(out_csv)
            score_col = None
            affinity_col = None
            for c in pred.columns:
                cl = c.lower()
                if "score" in cl or "prob" in cl or "prediction" in cl:
                    score_col = c
                    break
            if score_col is None:
                for c in pred.columns:
                    cl = c.lower()
                    if "affinity" in cl or "ic50" in cl:
                        affinity_col = c
                        break

            if score_col is not None:
                grp_aff = score_to_affinity(pred[score_col].astype(float).to_numpy())
            elif affinity_col is not None:
                grp_aff = pred[affinity_col].astype(float).to_numpy()
            else:
                raise RuntimeError(
                    f"Cannot find prediction score column for allele={allele}. "
                    f"Output columns: {list(pred.columns)}"
                )

            if len(grp_aff) != len(grp):
                raise RuntimeError(
                    f"MHCnuggets II output size mismatch for allele={allele}: "
                    f"pred={len(grp_aff)} vs input={len(grp)}"
                )
            all_affinity[grp["__orig_idx__"].to_numpy()] = grp_aff

        if np.isnan(all_affinity).any():
            n_nan = int(np.isnan(all_affinity).sum())
            raise RuntimeError(f"MHCnuggets II returned NaN/empty scores for {n_nan} rows.")
        return all_affinity


def run_mhcnuggets_predict_panel(peptides: list[str], alleles: list[str]) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="mhcnuggetsII_boot_panel_") as td:
        td_path = Path(td)
        pep_txt = td_path / "peptides.txt"
        pd.Series(peptides).to_csv(pep_txt, index=False, header=False)

        exe = shutil.which("mhcnuggets-predict") or shutil.which("mhcnuggets_predict")
        min_affinity = np.full(len(peptides), np.inf, dtype=np.float64)

        for i, allele in enumerate(alleles, start=1):
            out_csv = td_path / f"pred_{re.sub(r'[^A-Za-z0-9]+', '_', str(allele))}.csv"
            if exe is not None:
                cmd = [exe, "-c", MHCNUGGETS_CLASS, "-p", str(pep_txt), "-a", str(allele), "-o", str(out_csv)]
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "mhcnuggets.src.predict",
                    "-c",
                    MHCNUGGETS_CLASS,
                    "-p",
                    str(pep_txt),
                    "-a",
                    str(allele),
                    "-o",
                    str(out_csv),
                ]
            subprocess.check_call(cmd)

            pred = pd.read_csv(out_csv)
            score_col = None
            affinity_col = None
            for c in pred.columns:
                cl = c.lower()
                if "score" in cl or "prob" in cl or "prediction" in cl:
                    score_col = c
                    break
            if score_col is None:
                for c in pred.columns:
                    cl = c.lower()
                    if "affinity" in cl or "ic50" in cl:
                        affinity_col = c
                        break

            if score_col is not None:
                aff = score_to_affinity(pred[score_col].astype(float).to_numpy())
            elif affinity_col is not None:
                aff = pred[affinity_col].astype(float).to_numpy()
            else:
                raise RuntimeError(
                    f"Cannot find prediction score column for allele={allele}. Columns={list(pred.columns)}"
                )

            if len(aff) != len(peptides):
                raise RuntimeError(
                    f"MHCnuggets II output size mismatch for allele={allele}: pred={len(aff)} vs peptides={len(peptides)}"
                )
            min_affinity = np.minimum(min_affinity, aff)
            if i % 20 == 0 or i == len(alleles):
                print(f"[INFO] processed II alleles: {i}/{len(alleles)}")

        return min_affinity


def main() -> None:
    p = argparse.ArgumentParser(description="MHCnuggets class II inference + bootstrap metrics")
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_bootstrap", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allele_col", type=str, default=None)
    p.add_argument("--peptide_col", type=str, default=None)
    p.add_argument("--label_col", type=str, default=None)
    p.add_argument("--peptide_only", action="store_true")
    p.add_argument(
        "--default_allele",
        type=str,
        default="HLA-DRA0101-DRB10101",
        help="peptide_only 且无 panel 时使用的默认 II 等位基因（MHCnuggets 无冒号格式）",
    )
    p.add_argument("--allele_panel_csv", type=Path, default=None)
    p.add_argument("--aggregate", type=str, default="max", choices=["max"])
    p.add_argument("--max_alleles", type=int, default=0)
    p.add_argument("--ic50_threshold", type=float, default=500.0)
    p.add_argument("--min_peptide_len", type=int, default=8)
    p.add_argument("--max_peptide_len", type=int, default=40)
    args = p.parse_args()

    df = pd.read_csv(args.test_csv)

    allele_col = args.allele_col or infer_col(df, ["Allele", "allele", "mhc", "HLA", "hla"], required=False)
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
        print(f"[INFO] peptide_only mode. default_allele={args.default_allele}")
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
    work = work[work[peptide_col].str.len().between(args.min_peptide_len, args.max_peptide_len)]
    work = work[work[peptide_col].apply(lambda s: all(c in STANDARD_AA for c in s))]

    if use_panel:
        panel = load_allele_panel(args.allele_panel_csv)
        panel = [to_mhcnuggets_ii_allele(a) for a in panel]
        panel = [a for a in panel if a is not None]
        panel = list(dict.fromkeys(panel))
        if args.max_alleles > 0:
            panel = panel[: args.max_alleles]
        if not panel:
            raise ValueError("No valid alleles in panel after MHCnuggets II conversion.")
        print(f"[INFO] panel mode. using {len(panel)} alleles for aggregation={args.aggregate}")
    else:
        raw_n = len(work)
        work[allele_col] = work[allele_col].map(to_mhcnuggets_ii_allele)
        work = work.dropna(subset=[allele_col]).reset_index(drop=True)
        dropped = raw_n - len(work)
        if dropped > 0:
            print(f"[INFO] dropped rows with invalid II alleles: {dropped}")

    if len(work) == 0:
        raise ValueError("No valid rows after filtering.")

    if use_panel:
        y_affinity_all = run_mhcnuggets_predict_panel(work[peptide_col].tolist(), panel)
    else:
        y_affinity_all = run_mhcnuggets_predict(work, peptide_col=peptide_col, allele_col=allele_col)
    y_true_all = work[label_col].to_numpy(dtype=np.int64)

    if len(y_affinity_all) != len(y_true_all):
        raise RuntimeError(f"Prediction size mismatch: pred={len(y_affinity_all)}, label={len(y_true_all)}")

    rng = np.random.RandomState(args.seed)
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
        summary_rows.append({"Metric": metric, "Mean": mean_v, "Std": std_v, "Mean±Std": ms})
    summary = pd.DataFrame(summary_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = args.out_dir / "MHCnuggetsII_bootstrap_per_run.csv"
    summary_path = args.out_dir / "MHCnuggetsII_bootstrap_summary.csv"
    txt_path = args.out_dir / "MHCnuggetsII_bootstrap_summary.txt"

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
