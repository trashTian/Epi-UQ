from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import time
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


def score_to_affinity(score: np.ndarray) -> np.ndarray:
    s = np.asarray(score, dtype=np.float64)
    if np.nanmax(s) > 1.0:
        s = s / 100.0
    s = np.clip(s, 0.0, 1.0)
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


def format_mean_std(values: np.ndarray) -> tuple[float, float, str]:
    mean_v = float(np.nanmean(values))
    std_v = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    return mean_v, std_v, f"{mean_v:.4f}\u00b1{std_v:.4f}"


_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


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


def _anthem_scores_by_peptide_field(pred_file: Path, peptides: list[str]) -> np.ndarray | None:
    """
    Anthem length_*_prediction_result.txt is tab-separated; footer/summary lines also contain
    digits, so we must align by peptide string (exact tab field), not 'last number per line'.
    """
    lines = pred_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    scores: list[float] = []
    used: set[int] = set()
    for pep in peptides:
        for j, line in enumerate(lines):
            if j in used:
                continue
            fields = [x.strip() for x in line.split("\t")]
            if pep not in fields:
                continue
            nums = _FLOAT_RE.findall(line)
            if not nums:
                continue
            scores.append(float(nums[-1]))
            used.add(j)
            break
        else:
            return None
    return score_to_affinity(np.asarray(scores, dtype=np.float64))


def _anthem_scores_by_aa_token_length(pred_file: Path, n_expected: int, peptide_length: int) -> np.ndarray | None:
    vals: list[float] = []
    for line in pred_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "predicted binders" in line.lower() or "hla allele" in line.lower():
            continue
        parts = [p.strip() for p in line.split("\t")]
        for p in parts:
            if len(p) != peptide_length or any(c not in STANDARD_AA for c in p):
                continue
            nums = _FLOAT_RE.findall(line)
            if nums:
                vals.append(float(nums[-1]))
            break
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) != n_expected:
        return None
    return score_to_affinity(arr)


def _extract_affinity_from_output(pred_file: Path, peptides: list[str], peptide_length: int) -> np.ndarray:
    n_expected = len(peptides)
    try:
        df = pd.read_csv(pred_file)
        cand_score = None
        cand_aff = None
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in ["ic50", "affinity", "nm"]):
                cand_aff = c
                break
        if cand_aff is None:
            for c in df.columns:
                cl = c.lower()
                if any(k in cl for k in ["score", "prob", "prediction", "binder"]):
                    cand_score = c
                    break
        if cand_aff is not None:
            arr = df[cand_aff].astype(float).to_numpy()
        elif cand_score is not None:
            arr = score_to_affinity(df[cand_score].astype(float).to_numpy())
        else:
            raise ValueError("No usable prediction column in CSV.")
        if len(arr) == n_expected:
            return arr
    except Exception:
        pass

    by_pep = _anthem_scores_by_peptide_field(pred_file, peptides)
    if by_pep is not None:
        return by_pep

    by_len = _anthem_scores_by_aa_token_length(pred_file, n_expected, peptide_length)
    if by_len is not None:
        return by_len

    vals: list[float] = []
    for line in pred_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        nums = _FLOAT_RE.findall(line)
        if nums:
            vals.append(float(nums[-1]))
    arr = np.asarray(vals, dtype=np.float64)
    if len(arr) != n_expected:
        raise RuntimeError(
            f"Cannot parse Anthem output rows={len(arr)}, expected={n_expected}, file={pred_file}. "
            "Peptide-field alignment failed; check result file for extra/missing peptide lines."
        )
    return score_to_affinity(arr)


def _new_prediction_files(
    directory: Path,
    before_mtime: dict[Path, float],
    t0: float,
    exclude_resolved: set[Path],
) -> list[Path]:
    patterns = ("*.txt", "*.csv", "*.tsv", "*.out")
    candidates: list[Path] = []
    for pat in patterns:
        for p in directory.glob(pat):
            if not p.is_file():
                continue
            rp = p.resolve()
            if rp in exclude_resolved:
                continue
            mt = p.stat().st_mtime
            if (rp not in before_mtime) or (mt > before_mtime[rp]) or (mt >= t0 - 0.5):
                candidates.append(p)
    return candidates


def _find_anthem_result_in_timestamp_dirs(anthem_root: Path, peptide_length: int, t0: float) -> list[Path]:
    """
    Anthem prediction writes to cwd (anthem when using cwd=anthem_root):
    <anthem_root>/<YYYYMMDDHHMMSS>/length_<L>_prediction_result.txt
    See bin/sware_d_prediction.py (resultfolder) and bin/sware_w_writeresult.py.
    """
    skew = 2.0
    needle = f"length_{peptide_length}_prediction_result.txt"
    found: list[Path] = []
    for sub in anthem_root.iterdir():
        if not sub.is_dir():
            continue
        p = sub / needle
        try:
            if p.is_file() and p.stat().st_mtime >= t0 - skew:
                found.append(p)
        except OSError:
            continue
    return found


def run_anthem_predict_for_allele(anthem_root: Path, peptides: list[str], allele: str, peptide_length: int) -> np.ndarray:
    script = anthem_root / "sware_b_main.py"
    if not script.exists():
        raise FileNotFoundError(f"Anthem entry not found: {script}")

    unique_peps, inv = _stable_dedup_peptides(peptides)
    n_in, n_uq = len(peptides), len(unique_peps)
    if n_uq < n_in:
        print(
            f"[INFO] Anthem input de-dup (Anthem drops duplicate sequences): n={n_in} unique={n_uq} "
            f"(length={peptide_length})",
            flush=True,
        )

    with tempfile.TemporaryDirectory(prefix="anthem_predict_") as td:
        td_path = Path(td)
        pep_file = td_path / "peptides.txt"
        pd.Series(unique_peps).to_csv(pep_file, index=False, header=False)

        before_root: dict[Path, float] = {}
        for p in list(anthem_root.glob("*.txt")) + list(anthem_root.glob("*.csv")):
            before_root[p.resolve()] = p.stat().st_mtime
        before_td = {p.resolve(): p.stat().st_mtime for p in td_path.iterdir() if p.is_file()}
        t0 = time.time()
        exclude_td = {pep_file.resolve()}

        cmd = [
            sys.executable,
            str(script),
            "--mode",
            "prediction",
            "--length",
            str(peptide_length),
            "--HLA",
            allele,
            "--peptide_file",
            str(pep_file),
        ]
        print(
            f"[INFO] Anthem subprocess start: allele={allele} length={peptide_length} "
            f"n_peptides={n_uq} (Java/Weka; first batch can take tens of minutes)",
            flush=True,
        )
        t_sub = time.monotonic()
        subprocess.check_call(cmd, cwd=str(anthem_root))
        print(
            f"[INFO] Anthem subprocess done: allele={allele} length={peptide_length} "
            f"elapsed_s={time.monotonic() - t_sub:.1f}",
            flush=True,
        )

        candidates = _find_anthem_result_in_timestamp_dirs(anthem_root, peptide_length, t0)
        candidates.extend(_new_prediction_files(td_path, before_td, t0, exclude_td))
        candidates.extend(_new_prediction_files(anthem_root, before_root, t0, set()))
        seen: set[Path] = set()
        uniq: list[Path] = []
        for p in candidates:
            r = p.resolve()
            if r not in seen:
                seen.add(r)
                uniq.append(p)
        candidates = uniq
        if not candidates:
            raise RuntimeError(
                "No Anthem output file detected after prediction. "
                "Expected something like anthem_root/<timestamp>/length_<L>_prediction_result.txt "
                "(see sware_w_writeresult.writeresult). If Java/Weka failed, check Weka under source/weka-3-9-3."
            )
        pred_file = sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        aff_u = _extract_affinity_from_output(pred_file, peptides=unique_peps, peptide_length=peptide_length)
        return aff_u[inv]


def main() -> None:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(line_buffering=True)
        except Exception:
            pass

    p = argparse.ArgumentParser(description="Anthem inference + bootstrap metrics")
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_bootstrap", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--peptide_col", type=str, default=None)
    p.add_argument("--label_col", type=str, default=None)
    p.add_argument("--allele_panel_csv", type=Path, required=True)
    p.add_argument("--aggregate", type=str, default="max", choices=["max"])
    p.add_argument("--max_alleles", type=int, default=0)
    p.add_argument("--ic50_threshold", type=float, default=500.0)
    p.add_argument("--anthem_root", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    print(f"[INFO] bootstrap script: {Path(__file__).resolve()}", flush=True)

    df = pd.read_csv(args.test_csv)
    peptide_col = args.peptide_col or infer_col(df, ["Epitope.1", "peptide", "Peptide", "epitope", "sequence"])
    label_col = args.label_col or infer_col(df, ["Label", "label", "binder", "target", "class", "y"])

    work = df[[peptide_col, label_col]].copy()
    work = work.dropna(subset=[peptide_col, label_col]).reset_index(drop=True)
    work[peptide_col] = work[peptide_col].astype(str).str.strip().str.upper()
    work[label_col] = pd.to_numeric(work[label_col], errors="coerce")
    work = work.dropna(subset=[label_col])
    work = work[work[label_col].isin([0, 1])]
    work[label_col] = work[label_col].astype(int)
    work = work[work[peptide_col].str.len().between(8, 15)]
    work = work[work[peptide_col].apply(lambda s: all(c in STANDARD_AA for c in s))]
    if len(work) == 0:
        raise ValueError("No valid rows after filtering.")

    panel = load_allele_panel(args.allele_panel_csv)
    if args.max_alleles > 0:
        panel = panel[: args.max_alleles]
    if not panel:
        raise ValueError("No allele left for prediction.")
    print(f"[INFO] panel mode enabled. using {len(panel)} alleles for aggregation={args.aggregate}", flush=True)
    print(
        "[INFO] Progress: each (allele, length) runs one full Anthem call (many Weka models). "
        "Until the first 'Anthem subprocess done' line appears, the process is usually still working — "
        "use `top`/`htop` and look for `java` if unsure.",
        flush=True,
    )

    peptides = work[peptide_col].tolist()
    y_affinity_all = np.full(len(peptides), np.inf, dtype=np.float64)
    anthem_skipped: list[tuple[str, int]] = []

    for i, allele in enumerate(panel, start=1):
        print(f"[INFO] --- allele {i}/{len(panel)}: {allele} ---", flush=True)
        t_allele = time.monotonic()
        pred_aff = np.full(len(peptides), np.inf, dtype=np.float64)
        for length in sorted(work[peptide_col].str.len().unique().tolist()):
            idx = np.where(work[peptide_col].str.len().to_numpy() == length)[0]
            sub_peps = [peptides[j] for j in idx]
            try:
                sub_aff = run_anthem_predict_for_allele(args.anthem_root, sub_peps, allele, int(length))
            except subprocess.CalledProcessError as e:
                print(
                    f"[WARN] Anthem subprocess failed (exit {e.returncode}): allele={allele} length={length}. "
                    f"Treating as +inf IC50 for this allele only (panel uses min across alleles; other alleles unchanged).",
                    flush=True,
                )
                anthem_skipped.append((allele, int(length)))
                sub_aff = np.full(len(sub_peps), np.inf, dtype=np.float64)
            pred_aff[idx] = sub_aff
        y_affinity_all = np.minimum(y_affinity_all, pred_aff)
        print(
            f"[INFO] allele {i}/{len(panel)} finished in {time.monotonic() - t_allele:.1f}s "
            f"(cumulative min-affinity updated)",
            flush=True,
        )

    if anthem_skipped:
        print(
            f"[INFO] Anthem skipped {len(anthem_skipped)} (allele, length) pair(s) with no model; "
            f"first few: {anthem_skipped[:5]}",
            flush=True,
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
    per_run_path = args.out_dir / "Anthem_bootstrap_per_run.csv"
    summary_path = args.out_dir / "Anthem_bootstrap_summary.csv"
    txt_path = args.out_dir / "Anthem_bootstrap_summary.txt"
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
