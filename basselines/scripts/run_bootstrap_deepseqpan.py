#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def infer_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Column not found. candidates={candidates}, actual={list(df.columns)}")
    return None


def load_allele_panel(path: Path) -> List[str]:
    if path.suffix.lower() in (".txt", ".tsv"):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    if df.shape[1] == 1:
        vals = df.iloc[:, 0].astype(str).str.strip().tolist()
    else:
        c = infer_col(df, ["allele", "Allele", "HLA", "hla"], required=False)
        if c is not None:
            vals = df[c].astype(str).str.strip().tolist()
        else:
            vals = []
            for col in df.columns:
                vals.extend(df[col].dropna().astype(str).str.strip().tolist())
    vals = [v for v in vals if v and str(v).lower() != "nan"]
    if not vals:
        raise ValueError(f"No allele loaded from panel file: {path}")
    return list(dict.fromkeys(vals))


def split_hla_alpha_beta(allele: str) -> Optional[Tuple[str, str]]:
    if allele is None:
        return None
    s = str(allele).strip()
    if not s or s.lower() == "nan":
        return None
    if s.upper().startswith("HLA-"):
        s = s[4:].strip()

    if "/" in s:
        a, b = s.split("/", 1)
        return a.strip(), b.strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return a.strip(), b.strip()
    return None


def affinity_to_score(affinity_nm: np.ndarray) -> np.ndarray:
    aff = np.clip(affinity_nm.astype(np.float64), 1.0, 50000.0)
    return 1.0 - (np.log(aff) / np.log(50000.0))


def binary_metrics_from_ic50(y_true: np.ndarray, y_affinity_nm: np.ndarray, ic50_threshold: float = 500.0) -> Dict[str, float]:
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
) -> Dict[str, float]:
    n = len(y_true)
    idx = rng.randint(0, n, size=n)
    yt = y_true[idx]
    ya = y_affinity_nm[idx]
    return binary_metrics_from_ic50(yt, ya, ic50_threshold=ic50_threshold)


def format_mean_std(values: np.ndarray) -> Tuple[float, float, str]:
    mean_v = float(np.nanmean(values))
    std_v = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_v, std_v, f"{mean_v:.4f}\u00b1{std_v:.4f}"


def run_allele_worker(
    python_exe: str,
    worker_py: Path,
    code_dir: Path,
    model_path: str,
    hla_a: str,
    hla_b: str,
    peptides_txt: Path,
    out_csv: Path,
    progress_every: int,
) -> None:
    cmd = [
        python_exe,
        str(worker_py),
        "--code_dir",
        str(code_dir),
        "--model_path",
        model_path,
        "--hla_a",
        hla_a,
        "--hla_b",
        hla_b,
        "--peptides_txt",
        str(peptides_txt),
        "--out_csv",
        str(out_csv),
        "--progress_every",
        str(progress_every),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-6000:]
        raise RuntimeError(
            f"deepseqpanii_predict_allele failed (code={proc.returncode}) "
            f"hla_a={hla_a!r} hla_b={hla_b!r}\nstderr_tail:\n{tail}\nstdout_tail:\n{(proc.stdout or '')[-2000:]}"
        )


def read_ic50_csv(path: Path, peptides_order: List[str]) -> np.ndarray:
    df = pd.read_csv(path)
    c_pep = infer_col(df, ["peptide", "Peptide", "sequence"], required=True)
    c_ic = infer_col(df, ["ic50_nm", "IC50", "ic50"], required=True)
    m = dict(zip(df[c_pep].astype(str).str.upper().str.strip(), df[c_ic].astype(float)))
    out = np.full(len(peptides_order), np.nan, dtype=np.float64)
    for i, p in enumerate(peptides_order):
        if p in m:
            out[i] = float(m[p])
    return out


def predict_panel(
    python_exe: str,
    worker_py: Path,
    code_dir: Path,
    model_path: str,
    peptides: List[str],
    panel_pairs: List[Tuple[str, str]],
    strict_allele: bool,
    progress_every: int,
) -> np.ndarray:
    min_ic = np.full(len(peptides), np.inf, dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="deepseq_boot_") as td:
        td_path = Path(td)
        pep_txt = td_path / "peptides.txt"
        pep_txt.write_text("\n".join(peptides) + "\n", encoding="utf-8")

        for i, (ha, hb) in enumerate(panel_pairs, start=1):
            out_csv = td_path / f"pred_{i}.csv"
            try:
                run_allele_worker(
                    python_exe,
                    worker_py,
                    code_dir,
                    model_path,
                    ha,
                    hb,
                    pep_txt,
                    out_csv,
                    progress_every,
                )
                ic = read_ic50_csv(out_csv, peptides)
            except Exception as e:
                if strict_allele:
                    raise
                print(f"[WARN] skip allele pair ({ha}, {hb}): {e}", flush=True)
                continue
            min_ic = np.minimum(min_ic, ic)
            if i % 5 == 0 or i == len(panel_pairs):
                print(f"[INFO] DeepSeqPanII panel pairs processed: {i}/{len(panel_pairs)}", flush=True)

    if not np.all(np.isfinite(min_ic)):
        n_bad = int(np.sum(~np.isfinite(min_ic)))
        raise RuntimeError(
            f"{n_bad} peptides lack finite IC50 after panel "
            f"({'strict' if strict_allele else 'non-strict'} mode)."
        )
    return min_ic


def predict_per_row(
    python_exe: str,
    worker_py: Path,
    code_dir: Path,
    model_path: str,
    df: pd.DataFrame,
    peptide_col: str,
    pair_col: str,
    strict_allele: bool,
    progress_every: int,
) -> np.ndarray:
    all_ic = np.full(len(df), np.nan, dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="deepseq_boot_row_") as td:
        td_path = Path(td)
        for j, (key, grp) in enumerate(df.groupby(pair_col, sort=False), start=1):
            pair = split_hla_alpha_beta(str(key))
            if pair is None:
                if strict_allele:
                    raise ValueError(f"Cannot parse allele pair: {key!r}")
                print(f"[WARN] skip rows with unparsable allele={key!r}", flush=True)
                continue
            ha, hb = pair
            idx = grp.index.to_numpy()
            peps = grp[peptide_col].astype(str).str.upper().str.strip().tolist()
            safe = re.sub(r"[^A-Za-z0-9]+", "_", str(key))[:120]
            pep_txt = td_path / f"pep_{j}_{safe}.txt"
            out_csv = td_path / f"pred_{j}_{safe}.csv"
            pep_txt.write_text("\n".join(peps) + "\n", encoding="utf-8")
            try:
                run_allele_worker(
                    python_exe,
                    worker_py,
                    code_dir,
                    model_path,
                    ha,
                    hb,
                    pep_txt,
                    out_csv,
                    progress_every,
                )
                ic_vals = read_ic50_csv(out_csv, peps)
            except Exception as e:
                if strict_allele:
                    raise
                print(f"[WARN] skip allele pair ({ha}, {hb}): {e}", flush=True)
                continue
            for row_i, v in zip(idx, ic_vals):
                all_ic[row_i] = v
    if not np.all(np.isfinite(all_ic)):
        raise RuntimeError("DeepSeqPanII returned non-finite IC50 for some rows in per-row mode.")
    return all_ic


def main() -> None:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(line_buffering=True)
        except Exception:
            pass

    here = Path(__file__).resolve().parent
    default_root = here / "DeepSeqPanII"
    default_worker = here / "deepseqpanii_predict_allele.py"
    default_model = "../Models/benchmark_weekly/model_bd2013.pytorch"

    p = argparse.ArgumentParser(description="DeepSeqPanII inference + bootstrap metrics (IC50 nM)")
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_bootstrap", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allele_col", type=str, default=None, help="含 α/β 的字符串列，如 DRA*01:01-DRB1*01:01 或 HLA-DRA*01:01/DRB1*03:01")
    p.add_argument("--peptide_col", type=str, default=None)
    p.add_argument("--label_col", type=str, default=None)
    p.add_argument("--peptide_only", action="store_true")
    p.add_argument(
        "--default_allele",
        type=str,
        default="DRA*01:01-DRB1*01:01",
        help="peptide_only 且无 panel 时使用的默认 α-β 串",
    )
    p.add_argument("--allele_panel_txt", type=Path, default=None, help="如 DeepSeqPanII_alleles.txt（allele 列）")
    p.add_argument(
        "--allele_panel_csv",
        type=Path,
        default=None,
        help="与 allele_panel_txt 二选一；可为 .csv/.txt（含 allele 列）",
    )
    p.add_argument("--aggregate", type=str, default="max", choices=["max"], help="panel：多 allele 取最小 IC50")
    p.add_argument("--max_alleles", type=int, default=0, help=">0 时 panel 仅用前 N 个 allele 对")
    p.add_argument("--ic50_threshold", type=float, default=500.0)
    p.add_argument("--deepseqpanii_root", type=Path, default=default_root, help="DeepSeqPanII 仓库根目录")
    p.add_argument(
        "--code_dir",
        type=Path,
        default=None,
        help="code_and_dataset 目录；默认 <deepseqpanii_root>/code_and_dataset",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default=default_model,
        help="相对 code_dir 的模型路径（与官方 main.py 示例一致）",
    )
    p.add_argument(
        "--deepseq_python",
        type=str,
        default=None,
        help="运行 worker 的 Python 可执行文件；默认与当前脚本相同（若 DeepSeqPanII 需旧版 PyTorch，请传该环境的 python）",
    )
    p.add_argument("--worker_script", type=Path, default=default_worker)
    p.add_argument("--strict_alleles", action="store_true")
    p.add_argument("--progress_every", type=int, default=2000, help="每个 allele worker 内打印进度间隔（肽数）")
    args = p.parse_args()

    code_dir = (args.code_dir or (args.deepseqpanii_root / "code_and_dataset")).resolve()
    if not code_dir.is_dir():
        raise FileNotFoundError(f"code_dir not found: {code_dir}")
    worker_py = args.worker_script.resolve()
    if not worker_py.is_file():
        raise FileNotFoundError(f"worker_script not found: {worker_py}")

    python_exe = args.deepseq_python or sys.executable
    print(f"[INFO] DeepSeqPanII code_dir={code_dir}", flush=True)
    print(f"[INFO] worker python: {python_exe}", flush=True)
    print(f"[INFO] worker_script={worker_py}", flush=True)

    df = pd.read_csv(args.test_csv)
    allele_col = args.allele_col or infer_col(
        df, ["Allele", "allele", "mhc", "HLA", "hla", "allele_name"], required=False
    )
    peptide_col = args.peptide_col or infer_col(df, ["Epitope.1", "peptide", "Peptide", "epitope", "sequence"])
    label_col = args.label_col or infer_col(df, ["Label", "label", "binder", "target", "class", "y"])

    panel_path = args.allele_panel_txt or args.allele_panel_csv
    use_panel = panel_path is not None
    use_default_allele = (not use_panel) and (args.peptide_only or (allele_col is None))

    if use_panel:
        work = df[[peptide_col, label_col]].copy()
    elif use_default_allele:
        work = df[[peptide_col, label_col]].copy()
        work["__allele_pair__"] = args.default_allele
        allele_col = "__allele_pair__"
        print(f"[INFO] peptide_only mode. default_allele={args.default_allele}", flush=True)
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
    work = work[work[peptide_col].str.len().between(5, 25)]
    work = work[work[peptide_col].apply(lambda s: all(c in STANDARD_AA for c in s))]

    if len(work) == 0:
        raise ValueError("No valid rows after filtering.")

    peptides = work[peptide_col].tolist()

    if use_panel:
        raw_panel = load_allele_panel(panel_path)
        panel_pairs: List[Tuple[str, str]] = []
        for s in raw_panel:
            pr = split_hla_alpha_beta(s)
            if pr is None:
                print(f"[WARN] skip unparsable panel allele line: {s!r}", flush=True)
                continue
            panel_pairs.append(pr)
        panel_pairs = list(dict.fromkeys(panel_pairs))
        if args.max_alleles > 0:
            panel_pairs = panel_pairs[: args.max_alleles]
        if not panel_pairs:
            raise ValueError("No valid allele pairs in panel after parsing.")
        print(f"[INFO] panel mode: {len(panel_pairs)} allele pairs, aggregate={args.aggregate}", flush=True)
        y_ic50 = predict_panel(
            python_exe,
            worker_py,
            code_dir,
            args.model_path,
            peptides,
            panel_pairs,
            strict_allele=args.strict_alleles,
            progress_every=args.progress_every,
        )
    else:
        pred_df = work[[peptide_col, label_col, allele_col]].copy()
        y_ic50 = predict_per_row(
            python_exe,
            worker_py,
            code_dir,
            args.model_path,
            pred_df,
            peptide_col,
            allele_col,
            strict_allele=args.strict_alleles,
            progress_every=args.progress_every,
        )
        if len(y_ic50) != len(work):
            raise RuntimeError("Internal length mismatch after per-row prediction.")

    y_true = work[label_col].to_numpy(dtype=np.int64)
    if len(y_ic50) != len(y_true):
        raise RuntimeError(f"Prediction size mismatch: pred={len(y_ic50)}, labels={len(y_true)}")

    rng = np.random.RandomState(args.seed)
    rows: List[Dict[str, object]] = []
    for i in range(args.n_bootstrap):
        m = bootstrap_once(y_true, y_ic50, rng, ic50_threshold=args.ic50_threshold)
        m["bootstrap_id"] = i + 1
        rows.append(m)

    per_run = pd.DataFrame(rows)
    summary_rows: List[Dict[str, object]] = []
    for metric in METRIC_ORDER:
        vals = per_run[metric].astype(float).to_numpy()
        mean_v, std_v, ms = format_mean_std(vals)
        summary_rows.append({"Metric": metric, "Mean": mean_v, "Std": std_v, "Mean±Std": ms})
    summary = pd.DataFrame(summary_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = args.out_dir / "DeepSeqPanII_bootstrap_per_run.csv"
    summary_path = args.out_dir / "DeepSeqPanII_bootstrap_summary.csv"
    txt_path = args.out_dir / "DeepSeqPanII_bootstrap_summary.txt"
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