import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch


def read_hla_sequences(base_dir: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    def read(fname: str, d: Dict[str, str]) -> None:
        path = os.path.join(base_dir, "dataset", fname)
        with open(path, encoding="utf-8", errors="replace") as in_file:
            for line_num, line in enumerate(in_file):
                if line_num == 0:
                    continue
                info = line.strip("\n").split("\t")
                if len(info) >= 2:
                    d[info[0]] = info[1]

    hla_a: Dict[str, str] = {}
    hla_b: Dict[str, str] = {}
    read("CLUATAL_OMEGA_A_chains_aligned_FLATTEN.txt", hla_a)
    read("CLUATAL_OMEGA_B_chains_aligned_FLATTEN.txt", hla_b)
    return hla_a, hla_b


def pred_to_ic50_nm(pred01: float) -> float:
    p = float(np.clip(pred01, 1e-9, 1.0 - 1e-9))
    return float(np.power(50000.0, 1.0 - p))


def main() -> None:
    p = argparse.ArgumentParser(description="DeepSeqPanII batch IC50 for one allele pair")
    p.add_argument("--code_dir", type=str, required=True, help="DeepSeqPanII/code_and_dataset directory")
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model file path relative to code_dir (e.g. ../Models/benchmark_weekly/model_bd2013.pytorch)",
    )
    p.add_argument("--hla_a", type=str, required=True)
    p.add_argument("--hla_b", type=str, required=True)
    p.add_argument("--peptides_txt", type=str, required=True, help="One peptide sequence per line")
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--progress_every", type=int, default=2000)
    args = p.parse_args()

    code_dir = os.path.abspath(args.code_dir)
    os.chdir(code_dir)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    from config_parser import Config  # noqa: E402
    from model import Model  # noqa: E402
    from seq_encoding import one_hot_PLUS_blosum_encode  # noqa: E402

    cfg_path = os.path.join(code_dir, "config_main.json")
    config = Config(cfg_path)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_full = os.path.join(code_dir, args.model_path)
    try:
        state_dict = torch.load(model_full, map_location=config.device, weights_only=False)
    except TypeError:
        state_dict = torch.load(model_full, map_location=config.device)

    model = Model(config)
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()

    hla_sequence_A, hla_sequence_B = read_hla_sequences(code_dir)
    if args.hla_a not in hla_sequence_A:
        raise KeyError(f"hla_a={args.hla_a!r} not in CLUATAL_OMEGA_A_chains_aligned_FLATTEN.txt")
    if args.hla_b not in hla_sequence_B:
        raise KeyError(f"hla_b={args.hla_b!r} not in CLUATAL_OMEGA_B_chains_aligned_FLATTEN.txt")

    hla_a_seq = hla_sequence_A[args.hla_a]
    hla_b_seq = hla_sequence_B[args.hla_b]
    hla_a_encoded, hla_a_mask, hla_a_len = one_hot_PLUS_blosum_encode(hla_a_seq, config.max_len_hla_A)
    hla_b_encoded, hla_b_mask, hla_b_len = one_hot_PLUS_blosum_encode(hla_b_seq, config.max_len_hla_B)
    hla_a_encoded = hla_a_encoded.to(config.device)
    hla_a_mask = hla_a_mask.to(config.device)
    hla_b_encoded = hla_b_encoded.to(config.device)
    hla_b_mask = hla_b_mask.to(config.device)

    peptides = []
    with open(args.peptides_txt, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                peptides.append(s.upper())

    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n = len(peptides)
    with open(args.out_csv, "w", encoding="utf-8") as out_f:
        out_f.write("peptide,ic50_nm\n")
        for i, peptide in enumerate(peptides, start=1):
            pep_enc, pep_mask, pep_len = one_hot_PLUS_blosum_encode(peptide, config.max_len_pep)
            pep_enc = pep_enc.to(config.device)
            pep_mask = pep_mask.to(config.device)

            with torch.no_grad():
                pred_ic50, _ = model(
                    torch.stack([hla_a_encoded], dim=0),
                    torch.stack([hla_a_mask], dim=0),
                    torch.tensor([hla_a_len], device=config.device),
                    torch.stack([hla_b_encoded], dim=0),
                    torch.stack([hla_b_mask], dim=0),
                    torch.tensor([hla_b_len], device=config.device),
                    torch.stack([pep_enc], dim=0),
                    torch.stack([pep_mask], dim=0),
                    torch.tensor([pep_len], device=config.device),
                )
            ic50 = pred_to_ic50_nm(pred_ic50.item())
            out_f.write(f"{peptide},{ic50:.8g}\n")

            if args.progress_every > 0 and i % args.progress_every == 0:
                print(f"[DeepSeqPanII worker] {i}/{n} peptides", flush=True)


if __name__ == "__main__":
    main()
