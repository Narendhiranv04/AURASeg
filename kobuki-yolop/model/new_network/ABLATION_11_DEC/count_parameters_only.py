import os
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# ------------------------------------------------------------
# CHECKPOINT PATHS (as you requested)
# ------------------------------------------------------------
base_dir = r'kobuki-yolop/model/new_network/ABLATION_11_DEC/runs'
v1_ckpt = os.path.join(base_dir, r'v1_base_spp/checkpoints/best.pth')
v2_ckpt = os.path.join(base_dir, r'v2_base_assplite/checkpoints/best.pth')
v3_ckpt = os.path.join(base_dir, r'v3_apud/checkpoints/best.pth')
v4_ckpt = os.path.join(base_dir, r'v4_rbrm/checkpoints/best.pth')

CHECKPOINTS = {
    "V1 (CSPDarknet + SPP + BasicDec)": v1_ckpt,
    "V2 (CSPDarknet + ASPP-Lite + BasicDec)": v2_ckpt,
    "V3 (V2 + APUD + DeepSup)": v3_ckpt,
    "V4 (V3 + RBRM)": v4_ckpt,
}

OUTPUT_TXT = "param_report.txt"

# Since these are YOUR checkpoints, fallback is fine.
AUTO_FALLBACK_TO_WEIGHTS_ONLY_FALSE = True


def fmt_m(n: int) -> str:
    return f"{n / 1e6:.2f}M"


def print_block(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def add_weights_only_safe_globals():
    """
    PyTorch 2.6+ restricts globals for weights_only=True.
    Your ckpts include numpy scalar + numpy dtype class objects in metadata.
    We allowlist the common ones.
    """
    try:
        from torch.serialization import add_safe_globals
    except Exception:
        return

    safe = []

    # numpy scalar constructor (numpy>=2 uses numpy._core)
    try:
        safe.append(np._core.multiarray.scalar)
    except Exception:
        pass

    # dtype factory
    try:
        safe.append(np.dtype)
    except Exception:
        pass

    # numpy>=2 dtype classes live in np.dtypes
    dtypes_mod = getattr(np, "dtypes", None)
    if dtypes_mod is not None:
        for name in [
            "Float64DType", "Float32DType", "Float16DType",
            "Int64DType", "Int32DType", "Int16DType", "Int8DType",
            "UInt64DType", "UInt32DType", "UInt16DType", "UInt8DType",
            "BoolDType",
        ]:
            cls = getattr(dtypes_mod, name, None)
            if cls is not None:
                safe.append(cls)

    # Apply allowlist
    if safe:
        add_safe_globals(safe)


def safe_load_checkpoint(path: str):
    """
    1) Try weights_only=True with safe globals allowlisted.
    2) If still blocked and AUTO_FALLBACK... is True, fallback to weights_only=False.
    """
    add_weights_only_safe_globals()

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        if not AUTO_FALLBACK_TO_WEIGHTS_ONLY_FALSE:
            raise

        print("  [WARN] weights_only=True failed due to restricted globals.")
        print("  [WARN] Falling back to weights_only=False (ONLY SAFE if checkpoint is trusted).")
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(ckpt_obj):
    """
    Handle common checkpoint layouts.
    """
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            return ckpt_obj["model_state_dict"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        # Sometimes the ckpt itself is already a state_dict
        if all(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj

    raise ValueError("Could not extract a state_dict from this checkpoint object.")


def is_probably_learnable_param_key(k: str) -> bool:
    """
    Count only learnable parameters (weights/bias).
    Exclude BN buffers/statistics and any fixed kernels.
    """
    if not (k.endswith(".weight") or k.endswith(".bias")):
        return False

    bad_substrings = [
        "running_mean",
        "running_var",
        "num_batches_tracked",
        "sobel_x",
        "sobel_y",
    ]
    return not any(b in k for b in bad_substrings)


def count_params_from_state_dict(state_dict: dict):
    total = 0
    by_mod = defaultdict(int)
    skipped = 0
    total_tensors = 0

    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            continue
        total_tensors += 1

        if is_probably_learnable_param_key(k):
            n = v.numel()
            total += n
            top = k.split(".")[0] if "." in k else k
            by_mod[top] += n
        else:
            skipped += 1

    return total, dict(by_mod), skipped, total_tensors


def main():
    print_block("PARAMETER COUNT (checkpoint-based, weights_only=True + allowlisted numpy + fallback)")
    lines = []

    header = f"{'Model':<38} | {'Params':>12} | {'(M)':>8} | {'#tensors':>8} | {'skipped':>8}"
    print(header)
    print("-" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    results = []

    for name, ckpt_path in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            row = f"{name:<38} | {'NOT FOUND':>12} | {'':>8} | {'':>8} | {'':>8}"
            print(row)
            lines.append(row)
            continue

        try:
            ckpt_obj = safe_load_checkpoint(ckpt_path)
            sd = extract_state_dict(ckpt_obj)
            total, by_mod, skipped, total_tensors = count_params_from_state_dict(sd)

            row = f"{name:<38} | {total:>12,d} | {fmt_m(total):>8} | {total_tensors:>8} | {skipped:>8}"
            print(row)
            lines.append(row)

            results.append((name, total, by_mod, ckpt_path))
        except Exception as e:
            row = f"{name:<38} | {'ERROR':>12} | {'':>8} | {'':>8} | {'':>8}"
            print(row)
            print(f"  -> {type(e).__name__}: {e}")
            lines.append(row)
            lines.append(f"  -> {type(e).__name__}: {e}")

    print_block("MODULE-WISE BREAKDOWN (top-level prefix sums)")
    lines.append("\n" + "=" * 90)
    lines.append("MODULE-WISE BREAKDOWN (top-level prefix sums)")
    lines.append("=" * 90)

    for name, total, by_mod, ckpt_path in results:
        print(f"\n{name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"Total params (est.): {total:,} ({fmt_m(total)})")

        lines.append(f"\n{name}")
        lines.append(f"Checkpoint: {ckpt_path}")
        lines.append(f"Total params (est.): {total:,} ({fmt_m(total)})")

        items = sorted(by_mod.items(), key=lambda x: x[1], reverse=True)
        for mod, n in items:
            print(f"  - {mod:<18}: {n:>12,d} ({fmt_m(n)})")
            lines.append(f"  - {mod:<18}: {n:>12,d} ({fmt_m(n)})")

    out_path = Path(OUTPUT_TXT).resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print_block("DONE")
    print(f"Saved report to: {out_path}")


if __name__ == "__main__":
    main()
