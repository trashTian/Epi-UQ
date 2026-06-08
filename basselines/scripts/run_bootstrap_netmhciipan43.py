#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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


_NETMHCIIPAN_BIN_NAMES = ("NetMHCIIpan-4.3", "netMHCIIpan-4.3")


def _ensure_bin_executables(bin_dir: Path) -> None:
    for f in sorted(bin_dir.iterdir()):
        if not f.is_file():
            continue
        try:
            m = f.stat().st_mode
            if m & stat.S_IXUSR:
                continue
            os.chmod(f, m | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            print(f"[INFO] chmod +x {f}", flush=True)
        except OSError as e:
            print(f"[WARN] could not chmod {f}: {e}", flush=True)


def _find_executable_in_platform_bin(plat_dir: Path) -> Path | None:
    bind = plat_dir / "bin"
    if not bind.is_dir():
        return None
    for name in _NETMHCIIPAN_BIN_NAMES:
        p = bind / name
        if not p.is_file():
            continue
        if not os.access(p, os.X_OK):
            _ensure_bin_executables(bind)
        if os.access(p, os.X_OK):
            return p
        raise PermissionError(
            f"NetMHCIIpan 主程序无执行权限且自动 chmod 失败: {p}\n"
            f'请手动执行: chmod +x "{bind}/NetMHCIIpan-4.3" "{bind}"/estimate_PCC "{bind}"/mhcfsa2psseq "{bind}"/nnalign_*'
        )
    return None


def _find_package_root_with_data(platform_dir: Path) -> Path:
    cur: Path = platform_dir.resolve()
    for _ in range(5):
        if (cur / "data").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return platform_dir.parent


def resolve_netmhciipan_install(nm_root: Path) -> tuple[Path, Path, Path]:
    """
    解析安装根目录、平台目录、可执行文件。

    支持：
    - nm_root 为 DTU 根目录（含 data/ 与 Linux_*/Darwin_*）；
    - nm_root 直接为平台目录（如 .../netMHCIIpan-4.3/Linux_x86_64）；
    - 在 nm_root 下浅层搜索 */bin/NetMHCIIpan-4.3（目录结构被改过时）。
    返回 (package_root, platform_dir, exe_path)。
    """
    root = nm_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    exe = _find_executable_in_platform_bin(root)
    if exe is not None:
        pkg = _find_package_root_with_data(root)
        return pkg, root, exe

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        low = sub.name.lower()
        if not (low.startswith("linux_") or low.startswith("darwin_")):
            continue
        exe = _find_executable_in_platform_bin(sub)
        if exe is not None:
            pkg = _find_package_root_with_data(sub)
            return pkg, sub, exe

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        low = sub.name.lower()
        if low.startswith("linux_") or low.startswith("darwin_"):
            continue
        exe = _find_executable_in_platform_bin(sub)
        if exe is not None:
            pkg = _find_package_root_with_data(sub)
            return pkg, sub, exe

    tried = ", ".join(_NETMHCIIPAN_BIN_NAMES)
    raise FileNotFoundError(
        f"No executable [{tried}] under {root}. "
        "请确认已解压对应平台的 bin（例如 Linux_x86_64/bin/NetMHCIIpan-4.3），"
        "且 chmod +x；或把 --netmhciipan_home 设为含 data/ 的安装根目录，或直接设为 Linux_x86_64 目录。"
    )


def _digits_compact(typed: str) -> str:
    return typed.replace(":", "").replace("*", "")


def to_netmhciipan_allele(allele: str) -> str | None:
    if allele is None:
        return None
    s0 = str(allele).strip()
    if not s0 or s0.lower() == "nan":
        return None
    s = s0.replace(" ", "")

    if re.fullmatch(r"HLA-DQA1\d+-DQB1\d+", s, flags=re.I):
        return s.upper()
    if re.fullmatch(r"HLA-DPA1\d+-DPB1\d+", s, flags=re.I):
        return s.upper()

    m = re.match(
        r"^(HLA-)?DQA1\*([\d:]+)[/\-]DQB1\*([\d:]+)$",
        s,
        flags=re.I,
    )
    if m:
        a = _digits_compact(m.group(2))
        b = _digits_compact(m.group(3))
        return f"HLA-DQA1{a}-DQB1{b}".upper()

    m = re.match(
        r"^(HLA-)?DPA1\*([\d:]+)[/\-]DPB1\*([\d:]+)$",
        s,
        flags=re.I,
    )
    if m:
        a = _digits_compact(m.group(2))
        b = _digits_compact(m.group(3))
        return f"HLA-DPA1{a}-DPB1{b}".upper()

    m = re.match(
        r"^HLA-DRA\*[\d:]+/(DRB\d+)\*([\d:]+)$",
        s,
        flags=re.I,
    )
    if m:
        gene = m.group(1).upper()
        rest = _digits_compact(m.group(2))
        return f"{gene}_{rest}"

    m = re.match(r"^(HLA-)?(DRB\d+)\*([\d:]+)$", s, flags=re.I)
    if m:
        return f"{m.group(2).upper()}_{_digits_compact(m.group(3))}"

    m = re.match(r"^(HLA-)?(DRB\d+)_([\d]+)$", s, flags=re.I)
    if m:
        return f"{m.group(2).upper()}_{m.group(3)}"

    if re.match(r"^(BoLA-|H-2-)", s, flags=re.I):
        return s

    return s0


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


def _header_peptide_and_nm_indices(header_line: str) -> tuple[int, int]:
    parts = re.split(r"\s+", header_line.strip())
    try:
        pep_i = parts.index("Peptide")
    except ValueError as e:
        raise ValueError(f"NetMHCIIpan header missing Peptide: {header_line!r}") from e
    nm_i = None
    for i, p in enumerate(parts):
        if p == "nM" and not p.endswith("Rank"):
            nm_i = i
            break
    if nm_i is None:
        for i, p in enumerate(parts):
            if "Affinity" in p and "nM" in p:
                nm_i = i
                break
    if nm_i is None:
        for i, p in enumerate(parts):
            if p == "BA(nM)" or (p.startswith("BA(") and "nM" in p):
                nm_i = i
                break
    if nm_i is None:
        raise ValueError(f"NetMHCIIpan -BA header missing nM / Affinity(nM): {header_line!r}")
    return pep_i, nm_i


def parse_netmhciipan_ba_stdout(stdout: str) -> tuple[list[str], list[float]]:
    lines = stdout.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Pos") and "Peptide" in stripped:
            start = i
            break
    if start is None:
        raise RuntimeError("NetMHCIIpan output: missing Pos/Peptide header (check -BA and stderr).")

    pep_i, nm_i = _header_peptide_and_nm_indices(lines[start])
    peps: list[str] = []
    nms: list[float] = []
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.strip().startswith("---") or line.strip().startswith("Number of"):
            if peps and line.strip().startswith("Number of"):
                break
            continue
        parts = re.split(r"\s+", line.strip())
        if len(parts) <= max(pep_i, nm_i):
            continue
        if not parts[0].isdigit():
            continue
        try:
            nms.append(float(parts[nm_i]))
            peps.append(str(parts[pep_i]).upper())
        except (ValueError, IndexError):
            continue
    if not peps:
        raise RuntimeError("NetMHCIIpan output: no data rows parsed after header.")
    return peps, nms


def _is_unsupported_netmhciipan_allele(stderr: str | None) -> bool:
    """
    仅根据 stderr 判断是否「等位基因不被工具支持」。

    注意：不能把 stdout 拼进来判断——正常输出里也有 “Allele for prediction” 等字样，
    与 stderr 里泛化的 “Error.” 组合会造成大量误判，进而把合法 DQ 组合当成 inf 跳过。
    """
    if not (stderr or "").strip():
        return False
    t = stderr.lower()
    needles = (
        "unknown molecule",
        "unknown mhc",
        "molecule not found",
        "allele not in",
        "not among the alleles",
        "not among valid",
        "invalid molecule",
        "could not find molecule",
        "no valid mhc",
        "is not a valid",
        "not covered",
    )
    return any(n in t for n in needles)


def run_netmhciipan_ba(
    plat_dir: Path,
    exe: Path,
    peptides: list[str],
    allele: str,
    strict_allele: bool = False,
    tmp_root: Path | None = None,
) -> np.ndarray:
    allele_s = to_netmhciipan_allele(allele)
    if not allele_s:
        raise ValueError(f"Invalid allele: {allele!r}")

    base = (tmp_root or Path(tempfile.gettempdir())).resolve()
    base.mkdir(parents=True, exist_ok=True)
    run_id = f"{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex}"
    # 二进制内为 mkdtemp，默认模板 $TMPDIR/netMHCIIpan_XXXXXX；路径必须以六个 X 结尾，否则会 “Cannot make tmpdir”。
    work = base / f"netmhc_wrap_{run_id}"
    work.mkdir(mode=0o700)
    tdir_template = work / "netMHCIIpan_XXXXXX"
    pep_file = work / "peptides.txt"
    try:
        pd.Series(peptides).to_csv(pep_file, index=False, header=False)

        root_plat = plat_dir.resolve()
        env = os.environ.copy()
        env["TMPDIR"] = str(work)
        env["TEMP"] = str(work)
        env["TMP"] = str(work)
        env["NETMHCIIpan"] = str(root_plat)

        cmd = [
            str(exe),
            "-rdir",
            str(root_plat),
            "-tdir",
            str(tdir_template),
            "-inptype",
            "1",
            "-f",
            str(pep_file),
            "-a",
            allele_s,
            "-BA",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(work),
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            err = proc.stderr or ""
            out = proc.stdout or ""
            if not strict_allele and _is_unsupported_netmhciipan_allele(err):
                tail = err.strip()[-500:] if err.strip() else "(stderr empty)"
                print(
                    f"[WARN] NetMHCIIpan: allele not supported (stderr) allele={allele_s} | {tail}",
                    flush=True,
                )
                return np.full(len(peptides), np.inf, dtype=np.float64)
            tail_e = err[-4000:] if err.strip() else "(stderr empty)"
            tail_o = out[-4000:] if out.strip() else "(stdout empty)"
            raise RuntimeError(
                f"NetMHCIIpan failed (code={proc.returncode}) allele={allele_s}\n"
                f"stderr_tail:\n{tail_e}\nstdout_tail:\n{tail_o}"
            )

        peps_out, nm_out = parse_netmhciipan_ba_stdout(proc.stdout)

        best: dict[str, float] = {}
        for p, a in zip(peps_out, nm_out):
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
            raise RuntimeError(f"NetMHCIIpan missing BA nM for {n_bad} peptides (allele={allele_s}).")
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_netmhciipan_predict_panel_parallel(
    plat_dir: Path,
    exe: Path,
    peptides: list[str],
    alleles: list[str],
    strict_allele: bool,
    panel_workers: int,
    tmp_root: Path,
) -> np.ndarray:
    unique_peps, inv = _stable_dedup_peptides(peptides)
    n_rows, n_uq = len(peptides), len(unique_peps)
    if n_uq < n_rows:
        print(f"[INFO] NetMHCIIpan panel: de-dup n_rows={n_rows} n_unique={n_uq}", flush=True)

    min_u = np.full(n_uq, np.inf, dtype=np.float64)
    workers = max(1, min(panel_workers, len(alleles)))

    if workers == 1:
        for i, allele in enumerate(alleles, start=1):
            print(f"[INFO] NetMHCIIpan allele {i}/{len(alleles)}: {allele}", flush=True)
            aff = run_netmhciipan_ba(
                plat_dir, exe, unique_peps, allele, strict_allele=strict_allele, tmp_root=tmp_root
            )
            min_u = np.minimum(min_u, aff)
    else:
        print(f"[INFO] NetMHCIIpan panel parallel workers={workers}", flush=True)
        futures: dict = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, allele in enumerate(alleles, start=1):
                fut = pool.submit(
                    run_netmhciipan_ba,
                    plat_dir,
                    exe,
                    unique_peps,
                    allele,
                    strict_allele,
                    tmp_root,
                )
                futures[fut] = (i, allele)
            done = 0
            for fut in as_completed(futures):
                i, allele = futures[fut]
                aff = fut.result()
                min_u = np.minimum(min_u, aff)
                done += 1
                if done % 20 == 0 or done == len(alleles):
                    print(f"[INFO] NetMHCIIpan panel alleles finished: {done}/{len(alleles)} (last={allele})", flush=True)

    if not np.all(np.isfinite(min_u)):
        n_bad = int(np.sum(~np.isfinite(min_u)))
        raise RuntimeError(
            f"{n_bad} unique peptides have no finite IC50 after the full panel. "
            "Use --strict_alleles to fail fast, or fix the allele list / names for NetMHCIIpan."
        )
    return min_u[inv]


def _per_row_group_job(
    plat_dir: Path,
    exe: Path,
    idx: np.ndarray,
    peps: list[str],
    allele_s: str,
    strict_allele: bool,
    tmp_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    unique_peps, inv = _stable_dedup_peptides(peps)
    aff_u = run_netmhciipan_ba(plat_dir, exe, unique_peps, allele_s, strict_allele=strict_allele, tmp_root=tmp_root)
    aff_full = aff_u[inv]
    return idx, aff_full


def run_netmhciipan_predict_per_row_parallel(
    plat_dir: Path,
    exe: Path,
    df: pd.DataFrame,
    peptide_col: str,
    allele_col: str,
    strict_allele: bool,
    panel_workers: int,
    tmp_root: Path,
) -> np.ndarray:
    all_aff = np.full(len(df), np.nan, dtype=np.float64)
    groups: list[tuple[str, np.ndarray, list[str]]] = []
    for allele, grp in df.groupby(allele_col, sort=False):
        idx = grp.index.to_numpy()
        peps = grp[peptide_col].astype(str).str.upper().str.strip().tolist()
        groups.append((str(allele).strip(), idx, peps))

    workers = max(1, min(panel_workers, len(groups)))

    if workers == 1:
        for allele_s, idx, peps in groups:
            _, aff_full = _per_row_group_job(plat_dir, exe, idx, peps, allele_s, strict_allele, tmp_root)
            all_aff[idx] = aff_full
    else:
        print(f"[INFO] NetMHCIIpan per-row parallel workers={workers} (groups={len(groups)})", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(
                    _per_row_group_job,
                    plat_dir,
                    exe,
                    idx,
                    peps,
                    allele_s,
                    strict_allele,
                    tmp_root,
                ): allele_s
                for allele_s, idx, peps in groups
            }
            for fut in as_completed(futs):
                idx, aff_full = fut.result()
                all_aff[idx] = aff_full

    if not np.all(np.isfinite(all_aff)):
        raise RuntimeError(
            "NetMHCIIpan returned non-finite IC50 for some rows in per-allele mode "
            "(unsupported allele with --strict_alleles off still yields inf for those rows)."
        )
    return all_aff


def main() -> None:
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(line_buffering=True)
        except Exception:
            pass

    default_nm = Path(__file__).resolve().parent / "netMHCIIpan-4.3"
    default_workers = max(1, min(8, (os.cpu_count() or 4)))

    p = argparse.ArgumentParser(description="NetMHCIIpan-4.3 (-BA nM) + bootstrap — panel parallel + peptide de-dup")
    p.add_argument("--test_csv", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_bootstrap", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allele_col", type=str, default=None)
    p.add_argument("--peptide_col", type=str, default=None)
    p.add_argument("--label_col", type=str, default=None)
    p.add_argument("--peptide_only", action="store_true")
    p.add_argument("--default_allele", type=str, default="DRB1_0101")
    p.add_argument("--allele_panel_csv", type=Path, default=None)
    p.add_argument("--aggregate", type=str, default="max", choices=["max"])
    p.add_argument("--max_alleles", type=int, default=0)
    p.add_argument("--ic50_threshold", type=float, default=500.0)
    p.add_argument(
        "--netmhciipan_home",
        type=Path,
        default=default_nm,
        help="NetMHCIIpan 安装根目录（含 Linux_*/Darwin_* 子目录与顶层 data/）",
    )
    p.add_argument("--min_peptide_len", type=int, default=9)
    p.add_argument("--max_peptide_len", type=int, default=40)
    p.add_argument(
        "--strict_alleles",
        action="store_true",
        help="某一 allele 调用失败时立即退出；默认跳过该 allele（panel 下视为 inf）",
    )
    p.add_argument(
        "--panel_workers",
        type=int,
        default=default_workers,
        help=f"并行 NetMHCIIpan 子进程数（默认 min(8,CPU)={default_workers}）。设为 1 则顺序执行。",
    )
    p.add_argument(
        "--tmp_root",
        type=Path,
        default=None,
        help="NetMHCIIpan 每次子进程使用的临时目录父路径（需可写、空间充足）。"
        "默认: 环境变量 NETMHCIIpan_TMP_ROOT，否则系统 tempfile.gettempdir()（常为 /tmp）。"
        "若报 Cannot make tmpdir，请改到本地大盘如 /data1/.../nmhc_tmp",
    )
    args = p.parse_args()
    print(f"[INFO] bootstrap script: {Path(__file__).resolve()}", flush=True)

    tmp_root = (args.tmp_root or Path(os.environ.get("NETMHCIIpan_TMP_ROOT", tempfile.gettempdir()))).resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] NetMHCIIpan tmp_root={tmp_root}", flush=True)

    nm_root = args.netmhciipan_home.resolve()
    package_root, plat_dir, exe = resolve_netmhciipan_install(nm_root)
    if not (package_root / "data").is_dir():
        raise FileNotFoundError(
            f"Expected data directory at {package_root / 'data'}. "
            "若 --netmhciipan_home 指向 Linux_x86_64，请保证上一级目录含解压后的 data/；"
            "完整安装见 netMHCIIpan-4.3.readme。"
        )
    print(f"[INFO] NetMHCIIpan package_root={package_root} platform_dir={plat_dir} exe={exe}", flush=True)

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
    work = work[work[peptide_col].str.len().between(args.min_peptide_len, args.max_peptide_len)]
    work = work[work[peptide_col].apply(lambda s: all(c in STANDARD_AA for c in s))]

    if len(work) == 0:
        raise ValueError("No valid rows after filtering.")

    if use_panel:
        panel = load_allele_panel(args.allele_panel_csv)
        panel = [to_netmhciipan_allele(a) for a in panel]
        panel = [a for a in panel if a is not None]
        panel = list(dict.fromkeys(panel))
        if args.max_alleles > 0:
            panel = panel[: args.max_alleles]
        if not panel:
            raise ValueError("No allele left in panel.")
        if len(panel) > 400:
            print(
                f"[WARN] panel has {len(panel)} alleles (unusually many). "
                "If the CSV is multi-column without an 'Allele' header, every cell is treated as an allele.",
                flush=True,
            )
        print(f"[INFO] panel mode: {len(panel)} alleles, aggregate={args.aggregate}", flush=True)
        y_affinity_all = run_netmhciipan_predict_panel_parallel(
            plat_dir,
            exe,
            work[peptide_col].tolist(),
            panel,
            strict_allele=args.strict_alleles,
            panel_workers=args.panel_workers,
            tmp_root=tmp_root,
        )
    else:
        raw_n = len(work)
        work[allele_col] = work[allele_col].map(to_netmhciipan_allele)
        work = work.dropna(subset=[allele_col]).reset_index(drop=True)
        dropped = raw_n - len(work)
        if dropped > 0:
            print(f"[INFO] dropped rows with empty allele: {dropped}", flush=True)
        y_affinity_all = run_netmhciipan_predict_per_row_parallel(
            plat_dir,
            exe,
            work,
            peptide_col,
            allele_col,
            strict_allele=args.strict_alleles,
            panel_workers=args.panel_workers,
            tmp_root=tmp_root,
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
    per_run_path = args.out_dir / "NetMHCIIpan43_bootstrap_per_run.csv"
    summary_path = args.out_dir / "NetMHCIIpan43_bootstrap_summary.csv"
    txt_path = args.out_dir / "NetMHCIIpan43_bootstrap_summary.txt"
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
