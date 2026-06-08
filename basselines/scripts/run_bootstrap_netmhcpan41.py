#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
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
DEFAULT_LENGTHS = "8,9,10,11,12,13,14,15"


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
        allele_col = infer_col(panel_df, ["Allele", "allele", "HLA", "hla"], required=False)
        if allele_col is not None:
            vals = panel_df[allele_col].astype(str).str.strip().tolist()
        else:
            vals = []
            for c in panel_df.columns:
                vals.extend(panel_df[c].dropna().astype(str).str.strip().tolist())
    vals = [v for v in vals if v and v.lower() != "nan"]
    if not vals:
        raise ValueError(f"No allele loaded from panel file: {path}")
    return list(dict.fromkeys(vals))


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


def format_mean_std(values: np.ndarray) -> tuple[float, float, str]:
    mean_v = float(np.nanmean(values))
    std_v = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_v, std_v, f"{mean_v:.4f}\u00b1{std_v:.4f}"


def find_netmhcpan_executable(nmhome: Path) -> Path:
    for sub in sorted(nmhome.iterdir()):
        if not sub.is_dir():
            continue
        if not (sub.name.startswith("Linux_") or sub.name.startswith("Darwin_")):
            continue
        cand = sub / "bin" / "netMHCpan"
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError(
        f"No executable netMHCpan under {nmhome} (expected e.g. Linux_x86_64/bin/netMHCpan). "
        "Use the Linux tarball from DTU and unpack data.tar.gz; see netMHCpan-4.1.readme."
    )


def ensure_netmhcpan_top_bin_helpers(nmhome: Path) -> None:
    """
    NetMHCpan calls <NMHOME>/bin/estimate_PCC. Many installs only ship helpers under
    Linux_x86_64/bin/ next to the main binary — symlink those into NMHOME/bin/ when missing.
    """
    exe = find_netmhcpan_executable(nmhome)
    plat_bin = exe.parent
    top_bin = nmhome / "bin"
    top_bin.mkdir(parents=True, exist_ok=True)

    linked = 0
    for src in sorted(plat_bin.iterdir()):
        if src.name == "netMHCpan":
            continue
        if not src.is_file() and not src.is_symlink():
            continue
        dst = top_bin / src.name
        if dst.exists():
            continue
        rel = os.path.relpath(src, start=top_bin)
        try:
            dst.symlink_to(rel)
            linked += 1
        except OSError as e:
            raise FileNotFoundError(
                f"Cannot create symlink {dst} -> {rel}: {e}. "
                f"Fix permissions or run: mkdir -p {top_bin} && ln -sf {rel} {dst}"
            ) from e

    need = top_bin / "estimate_PCC"
    if not need.exists():
        raise FileNotFoundError(
            f"Missing {need} and no estimate_PCC under {plat_bin}. "
            "Re-unpack the full DTU netMHCpan-4.1 package."
        )
    if linked:
        print(f"[INFO] linked {linked} helper file(s) from {plat_bin} -> {top_bin}", flush=True)


def to_netmhcpan_allele(allele: str) -> str | None:
    if allele is None:
        return None
    s = str(allele).strip()
    if not s or s.lower() == "nan":
        return None
    return s


def _header_peptide_and_aff_indices(header_line: str) -> tuple[int, int]:
    parts = re.split(r"\s+", header_line.strip())
    try:
        pep_i = parts.index("Peptide")
    except ValueError as e:
        raise ValueError(f"NetMHCpan header missing Peptide: {header_line!r}") from e
    aff_i = None
    for i, p in enumerate(parts):
        if p.startswith("Aff") and "Rank" not in p:
            aff_i = i
            break
        if p == "BA" and aff_i is None:
            aff_i = i
    if aff_i is None:
        for i, p in enumerate(parts):
            if "nM" in p and "Rank" not in p:
                aff_i = i
                break
    if aff_i is None:
        raise ValueError(f"NetMHCpan header missing Aff(nM)/BA: {header_line!r}")
    return pep_i, aff_i


def parse_netmhcpan_ba_stdout(stdout: str) -> tuple[list[str], list[float]]:
    lines = stdout.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Pos") and "Peptide" in stripped:
            start = i
            break
    if start is None:
        raise RuntimeError("NetMHCpan output: missing Pos/Peptide header (check -BA and stderr).")

    pep_i, aff_i = _header_peptide_and_aff_indices(lines[start])
    peps: list[str] = []
    affs: list[float] = []
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if line.strip().startswith("Protein"):
            break
        if not line.strip() or line.strip().startswith("---"):
            continue
        parts = re.split(r"\s+", line.strip())
        if len(parts) <= max(pep_i, aff_i):
            continue
        try:
            affs.append(float(parts[aff_i]))
            peps.append(str(parts[pep_i]).upper())
        except (ValueError, IndexError):
            continue
    if not peps:
        raise RuntimeError("NetMHCpan output: no data rows parsed after header.")
    return peps, affs


def _is_unsupported_netmhcpan_allele(blob: str) -> bool:
    t = blob.lower()
    if "cannot be found" not in t:
        return False
    return "pseudo" in t or "synlist" in t or "hla_pseudo" in t


def run_netmhcpan_ba(
    nmhome: Path,
    peptides: list[str],
    allele: str,
    lengths: str = DEFAULT_LENGTHS,
    strict_allele: bool = False,
) -> np.ndarray:
    exe = find_netmhcpan_executable(nmhome)
    allele_s = to_netmhcpan_allele(allele)
    if not allele_s:
        raise ValueError(f"Invalid allele: {allele!r}")

    with tempfile.TemporaryDirectory(prefix="netmhcpan_boot_") as td:
        td_path = Path(td)
        pep_file = td_path / "peptides.txt"
        pd.Series(peptides).to_csv(pep_file, index=False, header=False)

        root = nmhome.resolve()
        env = os.environ.copy()
        env["TMPDIR"] = str(td_path)
        # Binary resolves data/ via $NETMHCpan; without this you get "Unable to open $NETMHCpan/data/version".
        env["NMHOME"] = str(root)
        env["NETMHCpan"] = str(root)

        cmd = [
            str(exe),
            "-rdir",
            str(root),
            "-p",
            str(pep_file),
            "-a",
            allele_s,
            "-l",
            lengths,
            "-BA",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(nmhome),
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            blob = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if not strict_allele and _is_unsupported_netmhcpan_allele(blob):
                print(
                    f"[WARN] NetMHCpan has no pseudo-sequence / model for allele={allele_s}; "
                    "skipping (panel min-IC50 ignores this allele).",
                    flush=True,
                )
                return np.full(len(peptides), np.inf, dtype=np.float64)
            tail = (proc.stderr or "")[-4000:]
            raise RuntimeError(
                f"netMHCpan failed (code={proc.returncode}) allele={allele_s}\n"
                f"stderr_tail:\n{tail}\nstdout_tail:\n{(proc.stdout or '')[-2000:]}"
            )

        peps_out, aff_out = parse_netmhcpan_ba_stdout(proc.stdout)

        best: dict[str, float] = {}
        for p, a in zip(peps_out, aff_out):
            if np.isnan(a):
                continue
            if p not in best or a < best[p]:
                best[p] = float(a)

        out = np.full(len(peptides), np.nan, dtype=np.float64)
        for i, p in enumerate(peptides):
            if p in best:
                out[i] = best[p]
        if np.isnan(out).any():
            n_bad = int(np.isnan(out).sum())
            raise RuntimeError(f"netMHCpan missing BA for {n_bad} peptides (allele={allele_s}).")
        return out


def run_netmhcpan_predict_panel(
    nmhome: Path, peptides: list[str], alleles: list[str], lengths: str, strict_allele: bool
) -> np.ndarray:
    min_aff = np.full(len(peptides), np.inf, dtype=np.float64)
    for i, allele in enumerate(alleles, start=1):
        print(f"[INFO] NetMHCpan allele {i}/{len(alleles)}: {allele}", flush=True)
        aff = run_netmhcpan_ba(nmhome, peptides, allele, lengths=lengths, strict_allele=strict_allele)
        min_aff = np.minimum(min_aff, aff)
    if not np.all(np.isfinite(min_aff)):
        n_bad = int(np.sum(~np.isfinite(min_aff)))
        raise RuntimeError(
            f"{n_bad} peptides have no finite IC50 after the full panel "
            "(every allele was unsupported or failed). Use --strict_alleles to fail fast on the first bad allele, "
            "or switch to a curated allele list supported by NetMHCpan."
        )
    return min_aff


def run_netmhcpan_predict_per_row(
    nmhome: Path, df: pd.DataFrame, peptide_col: str, allele_col: str, lengths: str, strict_allele: bool
) -> np.ndarray:
    all_aff = np.full(len(df), np.nan, dtype=np.float64)
    for allele, grp in df.groupby(allele_col, sort=False):
        idx = grp.index.to_numpy()
        peps = grp[peptide_col].astype(str).str.upper().str.strip().tolist()
        aff = run_netmhcpan_ba(nmhome, peps, str(allele), lengths=lengths, strict_allele=strict_allele)
        all_aff[idx] = aff
    if not np.all(np.isfinite(all_aff)):
        raise RuntimeError(
            "NetMHCpan returned non-finite IC50 for some rows in per-allele mode "
            "(unsupported allele with --strict_alleles off still yields inf for those rows)."
        )
    return all_aff


def main() -> None:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(line_buffering=True)
        except Exception:
            pass

    default_nm = Path(__file__).resolve().parent / "netMHCpan-4.1"
    p = argparse.ArgumentParser(description="NetMHCpan-4.1 (-BA) inference + bootstrap metrics")
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_bootstrap", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allele_col", type=str, default=None)
    p.add_argument("--peptide_col", type=str, default=None)
    p.add_argument("--label_col", type=str, default=None)
    p.add_argument("--peptide_only", action="store_true")
    p.add_argument("--default_allele", type=str, default="HLA-A*02:01")
    p.add_argument("--allele_panel_csv", type=Path, default=None)
    p.add_argument("--aggregate", type=str, default="max", choices=["max"])
    p.add_argument("--max_alleles", type=int, default=0)
    p.add_argument("--ic50_threshold", type=float, default=500.0)
    p.add_argument("--netmhcpan_home", type=Path, default=default_nm, help="Path to netMHCpan-4.1 root (contains data/ and Linux_*/bin/)")
    p.add_argument(
        "--lengths",
        type=str,
        default=DEFAULT_LENGTHS,
        help='NetMHCpan -l argument, e.g. "8,9,10,11,12,13,14,15"',
    )
    p.add_argument(
        "--strict_alleles",
        action="store_true",
        help="若某个 allele 不在 NetMHCpan 的 pseudo 列表中则立即报错；默认跳过该 allele 并继续",
    )
    args = p.parse_args()
    print(f"[INFO] bootstrap script: {Path(__file__).resolve()}", flush=True)

    nmhome = args.netmhcpan_home.resolve()
    if not (nmhome / "data").is_dir():
        raise FileNotFoundError(
            f"Expected {nmhome / 'data'} directory. Download data.tar.gz into netMHCpan-4.1 and unpack (see readme)."
        )
    _ = find_netmhcpan_executable(nmhome)
    ensure_netmhcpan_top_bin_helpers(nmhome)

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

    if use_panel:
        panel = load_allele_panel(args.allele_panel_csv)
        panel = [to_netmhcpan_allele(a) for a in panel]
        panel = [a for a in panel if a is not None]
        panel = list(dict.fromkeys(panel))
        if args.max_alleles > 0:
            panel = panel[: args.max_alleles]
        if not panel:
            raise ValueError("No allele left in panel.")
        if len(panel) > 400:
            print(
                f"[WARN] panel has {len(panel)} alleles (unusually many). "
                "If the CSV is multi-column without an 'Allele'/'HLA' header, every cell is treated as an allele "
                "(often wrong). Prefer one column or a single Allele column.",
                flush=True,
            )
        print(f"[INFO] panel mode enabled. using {len(panel)} alleles for aggregation={args.aggregate}", flush=True)
        y_affinity_all = run_netmhcpan_predict_panel(
            nmhome, work[peptide_col].tolist(), panel, args.lengths, strict_allele=args.strict_alleles
        )
    else:
        raw_n = len(work)
        work[allele_col] = work[allele_col].map(to_netmhcpan_allele)
        work = work.dropna(subset=[allele_col]).reset_index(drop=True)
        dropped = raw_n - len(work)
        if dropped > 0:
            print(f"[INFO] dropped rows with empty allele: {dropped}", flush=True)
        y_affinity_all = run_netmhcpan_predict_per_row(
            nmhome, work, peptide_col, allele_col, args.lengths, strict_allele=args.strict_alleles
        )

    y_true_all = work[label_col].to_numpy(dtype=np.int64)
    if len(y_affinity_all) != len(y_true_all):
        raise RuntimeError(f"Prediction size mismatch: pred={len(y_affinity_all)}, label={len(y_true_all)}")

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for i in range(args.n_bootstrap):
        idx = rng.integers(0, len(y_true_all), size=len(y_true_all))
        yt = y_true_all[idx]
        ya = y_affinity_all[idx]
        m = binary_metrics_from_ic50(yt, ya, ic50_threshold=args.ic50_threshold)
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
    per_run_path = args.out_dir / "NetMHCpan41_bootstrap_per_run.csv"
    summary_path = args.out_dir / "NetMHCpan41_bootstrap_summary.csv"
    txt_path = args.out_dir / "NetMHCpan41_bootstrap_summary.txt"
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
