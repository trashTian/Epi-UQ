#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from io import StringIO
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


def infer_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Column not found. candidates={candidates}, actual={list(df.columns)}")
    return None


def load_mixmhc2pred_allele_panel(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    header_i = None
    for i, line in enumerate(lines):
        if line.strip().startswith("AlleleName"):
            header_i = i
            break
    if header_i is None:
        raise ValueError(f"Could not find AlleleName header in {path}")
    df = pd.read_csv(StringIO("\n".join(lines[header_i:])), sep="\t")
    if "AlleleName" not in df.columns:
        raise ValueError(f"Expected AlleleName column after header in {path}, got {list(df.columns)}")
    vals = df["AlleleName"].dropna().astype(str).str.strip().tolist()
    vals = [v for v in vals if v and v.lower() != "nan"]
    if not vals:
        raise ValueError(f"No alleles loaded from {path}")
    return list(dict.fromkeys(vals))


def to_mixmhc2pred_token(allele: str) -> str | None:
    if allele is None:
        return None
    s = str(allele).strip().upper()
    if not s or s.lower() == "nan":
        return None
    if "*" not in s and ":" not in s and not s.startswith("HLA"):
        return s

    def norm_chain(x: str) -> str:
        y = x.replace("*", "_").replace(":", "_")
        y = re.sub(r"_+", "_", y).strip("_")
        return y

    if s.startswith("HLA-"):
        s = s[4:]
    parts = s.split("-")
    if len(parts) == 2 and parts[1].startswith("D"):
        return f"{norm_chain(parts[0])}__{norm_chain(parts[1])}"
    return norm_chain(parts[0]) if parts else None


def rank_to_score_for_auc(rank_pct: np.ndarray) -> np.ndarray:
    r = np.clip(rank_pct.astype(np.float64), 0.0, 100.0)
    return 1.0 - (r / 100.0)


def binary_metrics_from_percent_rank(
    y_true: np.ndarray,
    y_rank_pct: np.ndarray,
    rank_threshold: float,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_rank_pct = np.asarray(y_rank_pct, dtype=np.float64)
    y_score = rank_to_score_for_auc(y_rank_pct)
    y_pred = (y_rank_pct < rank_threshold).astype(np.int64)

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
    y_rank_pct: np.ndarray,
    rng: np.random.RandomState,
    rank_threshold: float,
) -> dict:
    n = len(y_true)
    idx = rng.randint(0, n, size=n)
    yt = y_true[idx]
    yr = y_rank_pct[idx]
    return binary_metrics_from_percent_rank(yt, yr, rank_threshold=rank_threshold)


def format_mean_std(values: np.ndarray) -> tuple[float, float, str]:
    mean_v = float(np.nanmean(values))
    std_v = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_v, std_v, f"{mean_v:.4f}\u00b1{std_v:.4f}"


def resolve_mixmhc2pred_exe(mix_dir: Path | None) -> Path:
    if mix_dir is None:
        for name in ("MixMHC2pred_unix", "MixMHC2pred", "MixMHC2pred.exe"):
            w = shutil.which(name)
            if w:
                return Path(w)
        raise FileNotFoundError(
            "MixMHC2pred executable not found in PATH. "
            "Pass --mixmhc2pred_dir pointing to the unzipped MixMHC2pred-2.0 folder."
        )
    root = mix_dir.resolve()
    if sys.platform == "darwin":
        for n in ("MixMHC2pred", "MixMHC2pred_unix"):
            p = root / n
            if p.is_file():
                return p
    else:
        p = root / "MixMHC2pred_unix"
        if p.is_file():
            return p
        p = root / "MixMHC2pred"
        if p.is_file():
            return p
    raise FileNotFoundError(f"No MixMHC2pred_unix / MixMHC2pred under {root}")


def mixmhc2pred_repo_root(exe: Path) -> Path:
    return exe.resolve().parent


def parse_mixmhc2pred_table(path: Path) -> pd.DataFrame:
    lines: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if line.strip():
                lines.append(line.rstrip("\n"))
    if not lines:
        raise RuntimeError(f"Empty MixMHC2pred output: {path}")
    header = lines[0].split("\t")
    rows = [r.split("\t") for r in lines[1:]]
    max_len = max(len(header), max((len(r) for r in rows), default=0))
    header = header + [f"col_{i}" for i in range(len(header), max_len)]
    rows = [r + [""] * (max_len - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=header)


def ranks_from_table(df: pd.DataFrame, allele_token: str | None = None) -> tuple[np.ndarray, list[str]]:
    low = {c.lower(): c for c in df.columns}
    pep_col = low.get("peptide")
    if pep_col is None:
        raise ValueError(f"MixMHC2pred output missing Peptide column. Columns={list(df.columns)}")

    rank_col = None
    if "%rank_best" in low:
        rank_col = low["%rank_best"]
    elif allele_token:
        key = "%rank_" + allele_token.lower()
        if key in low:
            rank_col = low[key]
        else:
            for k, c in low.items():
                if k.startswith("%rank_") and "best" not in k:
                    rank_col = c
                    break
    if rank_col is None:
        cand = [c for c in df.columns if str(c).lower().startswith("%rank_")]
        raise ValueError(f"Cannot find %%Rank column. Columns={list(df.columns)} rank-like={cand}")

    ranks = pd.to_numeric(df[rank_col], errors="coerce").to_numpy(dtype=np.float64)
    peps = df[pep_col].astype(str).str.strip().str.upper().to_numpy()
    return peps, ranks


def run_mixmhc2pred(
    exe: Path,
    peptide_file: Path,
    out_file: Path,
    allele_args: list[str],
    no_context: bool,
) -> pd.DataFrame:
    repo = mixmhc2pred_repo_root(exe)
    env = os.environ.copy()
    if "TMPDIR" not in env:
        env["TMPDIR"] = str(out_file.parent)

    cmd = [str(exe), "-i", str(peptide_file), "-o", str(out_file)]
    if no_context:
        cmd.append("--no_context")
    cmd.append("-a")
    cmd.extend(allele_args)

    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-6000:]
        raise RuntimeError(
            f"MixMHC2pred failed (code={proc.returncode}). cmd={cmd!r}\nstderr_tail:\n{tail}\nstdout_tail:\n{(proc.stdout or '')[-2000:]}"
        )
    if not out_file.is_file():
        raise FileNotFoundError(f"MixMHC2pred did not create: {out_file}")
    return parse_mixmhc2pred_table(out_file)


def predict_ranks_panel_per_allele(
    exe: Path,
    peptides: list[str],
    panel: list[str],
    strict_allele: bool,
) -> np.ndarray:
    min_r = np.full(len(peptides), np.nan, dtype=np.float64)
    pep_index = {p: i for i, p in enumerate(peptides)}
    with tempfile.TemporaryDirectory(prefix="mixmhc2_boot_") as td:
        td_path = Path(td)
        pep_file = td_path / "peptides_in.txt"
        pep_file.write_text("\n".join(peptides) + "\n", encoding="utf-8")
        for i, al in enumerate(panel, start=1):
            out_f = td_path / f"out_{i}.txt"
            try:
                df = run_mixmhc2pred(exe, pep_file, out_f, [al], no_context=True)
                peps_out, ranks = ranks_from_table(df, allele_token=al)
            except Exception as e:
                if strict_allele:
                    raise
                print(f"[WARN] MixMHC2pred failed for allele={al}; skipping. {e}", flush=True)
                continue
            for p, r in zip(peps_out, ranks):
                if p not in pep_index or not np.isfinite(r):
                    continue
                j = pep_index[p]
                min_r[j] = r if not np.isfinite(min_r[j]) else min(min_r[j], r)
            if i % 20 == 0 or i == len(panel):
                print(f"[INFO] MixMHC2pred II panel alleles: {i}/{len(panel)}", flush=True)
    if not np.all(np.isfinite(min_r)):
        n_bad = int(np.sum(~np.isfinite(min_r)))
        raise RuntimeError(
            f"{n_bad} peptides lack finite %Rank after panel "
            f"({'strict' if strict_allele else 'non-strict'} mode)."
        )
    return min_r


def predict_ranks_per_row(
    exe: Path,
    df: pd.DataFrame,
    peptide_col: str,
    allele_col: str,
    strict_allele: bool,
) -> np.ndarray:
    all_r = np.full(len(df), np.nan, dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="mixmhc2_row_") as td:
        td_path = Path(td)
        for j, (allele, grp) in enumerate(df.groupby(allele_col, sort=False), start=1):
            token = str(allele).strip()
            peps = grp[peptide_col].astype(str).str.upper().str.strip().tolist()
            pep_file = td_path / f"pep_{j}.txt"
            out_f = td_path / f"out_{j}.txt"
            pep_file.write_text("\n".join(peps) + "\n", encoding="utf-8")
            try:
                tab = run_mixmhc2pred(exe, pep_file, out_f, [token], no_context=True)
                peps_out, ranks = ranks_from_table(tab, allele_token=token)
            except Exception as e:
                if strict_allele:
                    raise
                print(f"[WARN] MixMHC2pred failed for allele={token}; rows skipped. {e}", flush=True)
                continue
            best: dict[str, float] = {}
            for p, r in zip(peps_out, ranks):
                if not np.isfinite(r):
                    continue
                if p not in best or r < best[p]:
                    best[p] = float(r)
            for idx_row, p in zip(grp.index.to_numpy(), peps):
                all_r[idx_row] = best.get(p, np.nan)
    if not np.all(np.isfinite(all_r)):
        raise RuntimeError("MixMHC2pred returned non-finite %Rank for some rows in per-allele mode.")
    return all_r


def main() -> None:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(line_buffering=True)
        except Exception:
            pass

    here = Path(__file__).resolve().parent
    default_dir = here / "MixMHC2pred-2.0"

    p = argparse.ArgumentParser(description="MixMHC2pred (HLA-II) + bootstrap metrics (%Rank)")
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_bootstrap", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allele_col", type=str, default=None)
    p.add_argument("--peptide_col", type=str, default=None)
    p.add_argument("--label_col", type=str, default=None)
    p.add_argument("--peptide_only", action="store_true")
    p.add_argument("--default_allele", type=str, default="DRB1_01_01", help="peptide_only 且无 panel 时的 MixMHC2pred allele token")
    p.add_argument("--allele_panel_txt", type=Path, default=None, help="如 MixMhc2pred_alleles.txt（含 AlleleName 列）")
    p.add_argument("--aggregate", type=str, default="max", choices=["max"])
    p.add_argument("--max_alleles", type=int, default=0)
    p.add_argument("--rank_threshold", type=float, default=5.0)
    p.add_argument(
        "--ic50_threshold",
        type=float,
        default=500.0,
        help="兼容其它 baseline；本脚本用 %%Rank，请用 --rank_threshold。",
    )
    p.add_argument("--mixmhc2pred_dir", type=Path, default=None, help="解压后的 MixMHC2pred 根目录（含 MixMHC2pred_unix 或 MixMHC2pred）")
    p.add_argument("--strict_alleles", action="store_true")
    p.add_argument("--min_peptide_len", type=int, default=8)
    p.add_argument("--max_peptide_len", type=int, default=40)
    args = p.parse_args()

    if args.ic50_threshold != 500.0:
        print(
            f"[INFO] --ic50_threshold={args.ic50_threshold} ignored; using --rank_threshold={args.rank_threshold}.",
            flush=True,
        )

    mix_dir = args.mixmhc2pred_dir if args.mixmhc2pred_dir is not None else default_dir
    exe = resolve_mixmhc2pred_exe(mix_dir if mix_dir.is_dir() else None)
    print(f"[INFO] MixMHC2pred executable: {exe}", flush=True)

    df = pd.read_csv(args.test_csv)
    allele_col = args.allele_col or infer_col(df, ["Allele", "allele", "mhc", "HLA", "hla"], required=False)
    peptide_col = args.peptide_col or infer_col(df, ["Epitope.1", "peptide", "Peptide", "epitope", "sequence"])
    label_col = args.label_col or infer_col(df, ["Label", "label", "binder", "target", "class", "y"])

    use_panel = args.allele_panel_txt is not None
    use_default_allele = (not use_panel) and (args.peptide_only or (allele_col is None))

    if use_panel:
        work = df[[peptide_col, label_col]].copy()
    elif use_default_allele:
        work = df[[peptide_col, label_col]].copy()
        work["__allele__"] = args.default_allele
        allele_col = "__allele__"
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
    work = work[work[peptide_col].str.len().between(args.min_peptide_len, args.max_peptide_len)]
    work = work[work[peptide_col].apply(lambda s: all(c in STANDARD_AA for c in s))]

    if len(work) == 0:
        raise ValueError("No valid rows after filtering.")

    peptides = work[peptide_col].tolist()

    if use_panel:
        panel = load_mixmhc2pred_allele_panel(args.allele_panel_txt)
        panel = [to_mixmhc2pred_token(a) or a for a in panel]
        panel = [a for a in panel if a]
        panel = list(dict.fromkeys(panel))
        if args.max_alleles > 0:
            panel = panel[: args.max_alleles]
        print(f"[INFO] panel: {len(panel)} alleles, aggregate={args.aggregate}", flush=True)
        y_rank = predict_ranks_panel_per_allele(exe, peptides, panel, strict_allele=args.strict_alleles)
    else:
        pred_df = work[[peptide_col, label_col, allele_col]].copy()
        pred_df["__tok__"] = pred_df[allele_col].map(lambda x: to_mixmhc2pred_token(str(x)) or str(x).strip())
        pred_df = pred_df.dropna(subset=["__tok__"]).reset_index(drop=True)
        work = pred_df.rename(columns={"__tok__": allele_col})
        y_rank = predict_ranks_per_row(exe, work, peptide_col, allele_col, strict_allele=args.strict_alleles)
        if len(y_rank) != len(work):
            raise RuntimeError("Length mismatch in per-row mode.")

    y_true = work[label_col].to_numpy(dtype=np.int64)
    if len(y_rank) != len(y_true):
        raise RuntimeError(f"pred/label length mismatch: {len(y_rank)} vs {len(y_true)}")

    rng = np.random.RandomState(args.seed)
    rows: list[dict] = []
    for i in range(args.n_bootstrap):
        m = bootstrap_once(y_true, y_rank, rng, rank_threshold=args.rank_threshold)
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
    per_run_path = args.out_dir / "MixMHC2pred_bootstrap_per_run.csv"
    summary_path = args.out_dir / "MixMHC2pred_bootstrap_summary.csv"
    txt_path = args.out_dir / "MixMHC2pred_bootstrap_summary.txt"
    per_run.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        for _, r in summary.iterrows():
            f.write(f"{r['Metric']}: {r['Mean±Std']}\n")

    print(f"Loaded rows: {len(work)}")
    print(f"Saved: {per_run_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    main()
