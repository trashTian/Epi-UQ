#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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


def infer_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Column not found. candidates={candidates}, actual={list(df.columns)}")
    return None


def load_allele_panel(path: Path) -> list[str]:
    panel_df = pd.read_csv(path)
    if panel_df.shape[1] == 1:
        vals = panel_df.iloc[:, 0].astype(str).str.strip().tolist()
    else:
        allele_col = infer_col(
            panel_df,
            ["Allele", "allele", "HLA", "hla", "alleles"],
            required=False,
        )
        if allele_col is not None:
            vals = panel_df[allele_col].astype(str).str.strip().tolist()
        else:
            vals = []
            for c in panel_df.columns:
                vals.extend(panel_df[c].dropna().astype(str).str.strip().tolist())
    vals = [v for v in vals if v and str(v).lower() != "nan"]
    if not vals:
        raise ValueError(f"No allele loaded from panel file: {path}")
    return list(dict.fromkeys(vals))


def to_mixmhcpred_allele(allele: str) -> str | None:
    if allele is None:
        return None
    s = str(allele).strip().upper()
    if not s or s.lower() == "nan":
        return None

    m = re.match(r"^([ABC])(\d{2})(\d{2})$", s)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

    m = re.match(r"^(?:HLA-)?([ABC])\*(\d{1,3}):(\d{1,3})(?::\d+)?$", s)
    if m:
        g, a1, a2 = m.group(1), int(m.group(2)), int(m.group(3))
        return f"{g}{a1:02d}{a2:02d}"

    m = re.match(r"^HLA-([ABC])(\d{1,3}):(\d{1,3})(?::\d+)?$", s)
    if m:
        g, a1, a2 = m.group(1), int(m.group(2)), int(m.group(3))
        return f"{g}{a1:02d}{a2:02d}"

    return None


def _stable_dedup_peptides(peptides: list[str]) -> tuple[list[str], np.ndarray]:
    key_to_uidx: dict[str, int] = {}
    unique: list[str] = []
    inv = np.empty(len(peptides), dtype=np.int64)
    for i, p in enumerate(peptides):
        if p not in key_to_uidx:
            key_to_uidx[p] = len(unique)
            unique.append(p)
        inv[i] = key_to_uidx[p]
    return unique, inv


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
    rng: np.random.Generator,
    rank_threshold: float,
) -> dict:
    n = len(y_true)
    idx = rng.integers(0, n, size=n)
    yt = y_true[idx]
    yr = y_rank_pct[idx]
    return binary_metrics_from_percent_rank(yt, yr, rank_threshold=rank_threshold)


def format_mean_std(values: np.ndarray) -> tuple[float, float, str]:
    mean_v = float(np.nanmean(values))
    std_v = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_v, std_v, f"{mean_v:.4f}\u00b1{std_v:.4f}"


def resolve_mixmhcpred_exe(mixmhcpred_dir: Path | None) -> Path:
    if mixmhcpred_dir is not None:
        root = mixmhcpred_dir.resolve()
        cand = root / "MixMHCpred"
        if cand.is_file():
            return cand
        raise FileNotFoundError(f"Missing executable: {cand}")
    w = shutil.which("MixMHCpred")
    if w:
        return Path(w)
    raise FileNotFoundError(
        "MixMHCpred executable not found. Pass --mixmhcpred_dir pointing to the cloned repo root."
    )


def mixmhcpred_repo_root(exe: Path) -> Path:
    return exe.resolve().parent


def parse_mixmhcpred_table(path: Path) -> pd.DataFrame:
    lines: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if line.strip():
                lines.append(line.rstrip("\n"))
    if not lines:
        raise RuntimeError(f"Empty MixMHCpred output: {path}")
    header = lines[0].split("\t")
    rows = [r.split("\t") for r in lines[1:]]
    max_len = max(len(header), max((len(r) for r in rows), default=0))
    header = header + [f"col_{i}" for i in range(len(header), max_len)]
    rows = [r + [""] * (max_len - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=header)


def ranks_from_table(df: pd.DataFrame, peptide_col: str = "Peptide") -> tuple[np.ndarray, list[str]]:
    if peptide_col not in df.columns:
        low = {c.lower(): c for c in df.columns}
        if "peptide" in low:
            peptide_col = low["peptide"]
        else:
            raise ValueError(f"MixMHCpred output missing Peptide column. Got: {list(df.columns)}")

    low = {c.lower(): c for c in df.columns}
    best_key = None
    for k in ("%rank_bestallele", "rank_bestallele"):
        if k in low:
            best_key = low[k]
            break
    if best_key is not None:
        ranks = pd.to_numeric(df[best_key], errors="coerce").to_numpy(dtype=np.float64)
    else:
        rank_cols = [c for c in df.columns if c.lower().startswith("%rank_") and "best" not in c.lower()]
        if not rank_cols:
            raise ValueError(
                f"Cannot find %Rank_bestAllele or per-allele %%Rank_* columns. Columns={list(df.columns)}"
            )
        mat = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64) for c in rank_cols])
        ranks = np.nanmin(mat, axis=1)

    peps = df[peptide_col].astype(str).str.strip().str.upper().to_numpy()
    return peps, ranks


def run_mixmhcpred(
    exe: Path,
    peptides: list[str],
    alleles_csv: str,
    tmpdir: Path,
    out_name: str = "mix_out.txt",
) -> pd.DataFrame:
    pep_file = tmpdir / "peptides_in.txt"
    out_file = tmpdir / out_name
    pd.Series(peptides).to_csv(pep_file, index=False, header=False)

    repo = mixmhcpred_repo_root(exe)
    env = os.environ.copy()
    env["TMPDIR"] = str(tmpdir)

    cmd = [str(exe), "-i", str(pep_file), "-o", str(out_file), "-a", alleles_csv, "-p", "1"]
    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-4000:]
        raise RuntimeError(
            f"MixMHCpred failed (code={proc.returncode}). cmd={cmd!r}\nstderr_tail:\n{tail}\nstdout_tail:\n{(proc.stdout or '')[-2000:]}"
        )
    if not out_file.is_file():
        raise FileNotFoundError(f"MixMHCpred did not create output file: {out_file}")
    return parse_mixmhcpred_table(out_file)


def predict_ranks_panel_per_allele(
    exe: Path,
    peptides: list[str],
    panel: list[str],
    strict_allele: bool,
) -> np.ndarray:
    unique_peps, inv = _stable_dedup_peptides(peptides)
    n_rows, n_uq = len(peptides), len(unique_peps)
    if n_uq < n_rows:
        print(
            f"[INFO] MixMHCpred panel: de-dup for CLI input n_rows={n_rows} n_unique={n_uq}",
            flush=True,
        )
    min_u = np.full(n_uq, np.nan, dtype=np.float64)
    uniq_index = {p: j for j, p in enumerate(unique_peps)}
    with tempfile.TemporaryDirectory(prefix="mixmhc_boot_") as td:
        td_path = Path(td)
        for i, al in enumerate(panel, start=1):
            try:
                df = run_mixmhcpred(exe, unique_peps, al, td_path, out_name=f"out_{i}.txt")
                peps_out, ranks = ranks_from_table(df)
            except Exception as e:
                if strict_allele:
                    raise
                print(f"[WARN] MixMHCpred failed for allele={al}; skipping. {e}", flush=True)
                continue
            for p, r in zip(peps_out, ranks):
                if not np.isfinite(r):
                    continue
                j = uniq_index.get(p)
                if j is None:
                    continue
                min_u[j] = r if not np.isfinite(min_u[j]) else min(min_u[j], r)
            if i % 20 == 0 or i == len(panel):
                print(f"[INFO] MixMHCpred panel alleles processed: {i}/{len(panel)}", flush=True)
    if not np.all(np.isfinite(min_u)):
        bad = ~np.isfinite(min_u)
        n_bad = int(np.sum(bad))
        bad_peps = [unique_peps[j] for j in np.flatnonzero(bad)]
        print(
            f"[WARN] {n_bad} unique peptide(s) had no finite %%Rank across the panel (omitted by MixMHCpred or NaN on "
            f"every allele). Imputing %%Rank=100.0 (weakest; same as no evidence). First few: {bad_peps[:8]}",
            flush=True,
        )
        min_u[bad] = 100.0
    return min_u[inv]


def predict_ranks_panel_one_call(exe: Path, peptides: list[str], panel: list[str]) -> np.ndarray:
    unique_peps, inv = _stable_dedup_peptides(peptides)
    n_rows, n_uq = len(peptides), len(unique_peps)
    if n_uq < n_rows:
        print(
            f"[INFO] MixMHCpred panel_one_call: de-dup for CLI input n_rows={n_rows} n_unique={n_uq}",
            flush=True,
        )
    alleles_csv = ",".join(panel)
    with tempfile.TemporaryDirectory(prefix="mixmhc_boot_one_") as td:
        td_path = Path(td)
        df = run_mixmhcpred(exe, unique_peps, alleles_csv, td_path, out_name="out_panel.txt")
        peps_out, ranks = ranks_from_table(df)
    uniq_index = {p: j for j, p in enumerate(unique_peps)}
    out_u = np.full(n_uq, np.nan, dtype=np.float64)
    for p, r in zip(peps_out, ranks):
        if not np.isfinite(r):
            continue
        j = uniq_index.get(p)
        if j is None:
            continue
        out_u[j] = r if not np.isfinite(out_u[j]) else min(out_u[j], r)
    if not np.all(np.isfinite(out_u)):
        bad = ~np.isfinite(out_u)
        n_bad = int(np.sum(bad))
        bad_peps = [unique_peps[j] for j in np.flatnonzero(bad)]
        print(
            f"[WARN] {n_bad} unique peptide(s) missing or non-finite %%Rank in panel_one_call output. "
            f"Imputing %%Rank=100.0 (weakest). First few: {bad_peps[:8]}",
            flush=True,
        )
        out_u[bad] = 100.0
    return out_u[inv]


def predict_ranks_per_row(
    exe: Path,
    df: pd.DataFrame,
    peptide_col: str,
    allele_col: str,
    strict_allele: bool,
) -> np.ndarray:
    all_r = np.full(len(df), np.nan, dtype=np.float64)
    for allele, grp in df.groupby(allele_col, sort=False):
        idx = grp.index.to_numpy()
        peps = grp[peptide_col].astype(str).str.upper().str.strip().tolist()
        token = to_mixmhcpred_allele(str(allele))
        if token is None:
            if strict_allele:
                raise ValueError(f"Cannot convert allele to MixMHCpred token: {allele!r}")
            print(f"[WARN] skip rows with unmapped allele={allele!r}", flush=True)
            continue
        with tempfile.TemporaryDirectory(prefix="mixmhc_boot_row_") as td:
            td_path = Path(td)
            try:
                tab = run_mixmhcpred(exe, peps, token, td_path, out_name=f"out_{token}.txt")
                peps_out, ranks = ranks_from_table(tab)
            except Exception as e:
                if strict_allele:
                    raise
                print(f"[WARN] MixMHCpred failed for allele={allele}; rows skipped. {e}", flush=True)
                continue
        best: dict[str, float] = {}
        for p, r in zip(peps_out, ranks):
            if not np.isfinite(r):
                continue
            if p not in best or r < best[p]:
                best[p] = float(r)
        for j, p in zip(idx, peps):
            all_r[j] = best.get(p, np.nan)
    if not np.all(np.isfinite(all_r)):
        bad = ~np.isfinite(all_r)
        n_bad = int(np.sum(bad))
        print(
            f"[WARN] {n_bad} row(s) lack finite %%Rank in per-row mode; imputing %%Rank=100.0 (weakest).",
            flush=True,
        )
        all_r[bad] = 100.0
    return all_r


def main() -> None:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(line_buffering=True)
        except Exception:
            pass

    p = argparse.ArgumentParser(description="MixMHCpred inference + bootstrap metrics (%Rank-based)")
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_bootstrap", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allele_col", type=str, default=None)
    p.add_argument("--peptide_col", type=str, default=None)
    p.add_argument("--label_col", type=str, default=None)
    p.add_argument("--peptide_only", action="store_true")
    p.add_argument("--default_allele", type=str, default="HLA-A*02:01", help="peptide_only 模式下统一使用的等位基因")
    p.add_argument("--allele_panel_csv", type=Path, default=None)
    p.add_argument("--aggregate", type=str, default="max", choices=["max"], help="panel 下多 allele：取最小 %%Rank（最强呈递）")
    p.add_argument("--max_alleles", type=int, default=0, help=">0 时 panel 仅用前 N 个 allele")
    p.add_argument(
        "--rank_threshold",
        type=float,
        default=5.0,
        help="二元分类：%%Rank < 该阈值判为阳性（MixMHCpred 无 IC50；越小越强）",
    )
    p.add_argument(
        "--ic50_threshold",
        type=float,
        default=500.0,
        help="其它 baseline 兼容参数；本脚本不使用 IC50，请用 --rank_threshold。若传入非默认值会打印提示。",
    )
    p.add_argument(
        "--mixmhcpred_dir",
        type=Path,
        default=None,
        help="MixMHCpred 克隆仓库根目录（内含可执行文件 MixMHCpred）；不设则依赖 PATH 中的 MixMHCpred",
    )
    p.add_argument(
        "--panel_one_call",
        action="store_true",
        help="panel 模式下一次传入全部 allele（读 %%Rank_bestAllele）；默认逐 allele 更稳但较慢",
    )
    p.add_argument(
        "--strict_alleles",
        action="store_true",
        help="panel 逐 allele 或 per-row 时，某个 allele 失败则立即报错；默认跳过失败 allele",
    )
    args = p.parse_args()

    if args.ic50_threshold != 500.0:
        print(
            f"[INFO] --ic50_threshold={args.ic50_threshold} is ignored by MixMHCpred "
            f"(no IC50). Using --rank_threshold={args.rank_threshold} for binary calls.",
            flush=True,
        )

    exe = resolve_mixmhcpred_exe(args.mixmhcpred_dir)
    print(f"[INFO] MixMHCpred executable: {exe}", flush=True)

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
    work = work[work[peptide_col].str.len().between(8, 15)]
    work = work[work[peptide_col].apply(lambda s: all(c in STANDARD_AA for c in s))]

    if len(work) == 0:
        raise ValueError("No valid rows after filtering.")

    peptides = work[peptide_col].tolist()

    if use_panel:
        panel = load_allele_panel(args.allele_panel_csv)
        panel = [to_mixmhcpred_allele(a) for a in panel]
        panel = [a for a in panel if a is not None]
        panel = list(dict.fromkeys(panel))
        if args.max_alleles > 0:
            panel = panel[: args.max_alleles]
        if not panel:
            raise ValueError("No valid alleles in panel after MixMHCpred conversion.")
        print(f"[INFO] panel mode: {len(panel)} alleles, aggregate={args.aggregate}", flush=True)
        if args.panel_one_call:
            y_rank_all = predict_ranks_panel_one_call(exe, peptides, panel)
        else:
            y_rank_all = predict_ranks_panel_per_allele(exe, peptides, panel, strict_allele=args.strict_alleles)
    else:
        raw_n = len(work)
        work["__mix_token__"] = work[allele_col].map(lambda x: to_mixmhcpred_allele(str(x)))
        work = work.dropna(subset=["__mix_token__"]).reset_index(drop=True)
        dropped = raw_n - len(work)
        if dropped > 0:
            print(f"[INFO] dropped rows with unmappable allele: {dropped}", flush=True)
        pred_df = work[[peptide_col, label_col, "__mix_token__"]].rename(
            columns={"__mix_token__": "__mix_allele__"},
        )
        y_rank_all = predict_ranks_per_row(
            exe,
            pred_df,
            peptide_col,
            "__mix_allele__",
            strict_allele=args.strict_alleles,
        )
        if len(y_rank_all) != len(work):
            raise RuntimeError("Internal length mismatch after per-row prediction.")

    y_true_all = work[label_col].to_numpy(dtype=np.int64)
    if len(y_rank_all) != len(y_true_all):
        raise RuntimeError(f"Prediction size mismatch: pred={len(y_rank_all)}, labels={len(y_true_all)}")

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for i in range(args.n_bootstrap):
        m = bootstrap_once(y_true_all, y_rank_all, rng, rank_threshold=args.rank_threshold)
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
    per_run_path = args.out_dir / "MixMHCpred_bootstrap_per_run.csv"
    summary_path = args.out_dir / "MixMHCpred_bootstrap_summary.csv"
    txt_path = args.out_dir / "MixMHCpred_bootstrap_summary.txt"
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
