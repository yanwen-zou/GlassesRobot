#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path
from typing import List

import numpy as np


def ensure_4x4(arr: np.ndarray, path: Path) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"{path} does not contain a numpy array")
    if arr.shape != (4, 4):
        raise ValueError(f"Expected shape (4,4), got {arr.shape} in {path}")
    return arr


def parse_16_numbers(values: List[str]) -> np.ndarray:
    if len(values) != 16:
        raise ValueError("--set-matrix requires exactly 16 numbers (row-major)")
    nums = [float(v) for v in values]
    return np.array(nums, dtype=float).reshape(4, 4)


def main():
    parser = argparse.ArgumentParser(
        description="Edit elements of a 4x4 matrix stored in a .npy file (e.g., eih_camT.npy)."
    )
    parser.add_argument("path", help="Path to .npy file containing a 4x4 matrix")
    parser.add_argument(
        "--set",
        nargs=3,
        action="append",
        metavar=("ROW", "COL", "VALUE"),
        help="Set a single element at [ROW, COL] (0-based) to VALUE. Can be repeated.",
    )
    parser.add_argument(
        "--set-row",
        nargs=5,
        action="append",
        metavar=("ROW", "v0", "v1", "v2", "v3"),
        help="Replace an entire row (0-based) with four values.",
    )
    parser.add_argument(
        "--set-matrix",
        nargs=16,
        metavar=[f"m{i}{j}" for i in range(4) for j in range(4)],
        help="Replace the whole matrix with 16 numbers (row-major order).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak backup before writing.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the current matrix; do not modify.",
    )

    args = parser.parse_args()
    p = Path(args.path)
    if not p.exists():
        print(f"File not found: {p}", file=sys.stderr)
        sys.exit(1)

    try:
        arr = np.load(p, allow_pickle=False)
    except Exception as e:
        print(f"Failed to load {p}: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        arr = ensure_4x4(arr, p)
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(3)

    if args.print_only and not (args.set or args.set_row or args.set_matrix):
        with np.printoptions(linewidth=120):
            print(arr)
        return

    orig_dtype = arr.dtype

    if args.set_matrix is not None:
        new_m = parse_16_numbers(args.set_matrix)
        arr = new_m.astype(orig_dtype, copy=False)

    if args.set_row:
        for entry in args.set_row:
            r = int(entry[0])
            if r < 0 or r > 3:
                print(f"Row index out of range: {r}", file=sys.stderr)
                sys.exit(4)
            vals = [float(v) for v in entry[1:]]
            if len(vals) != 4:
                print("--set-row needs exactly four values", file=sys.stderr)
                sys.exit(4)
            arr[r, :] = np.array(vals, dtype=orig_dtype)

    if args.set:
        for r_str, c_str, v_str in args.set:
            r, c = int(r_str), int(c_str)
            if not (0 <= r <= 3 and 0 <= c <= 3):
                print(f"Index out of range: ({r}, {c})", file=sys.stderr)
                sys.exit(5)
            v = float(v_str)
            arr[r, c] = orig_dtype.type(v) if hasattr(orig_dtype, 'type') else orig_dtype(v)

    if not (args.set or args.set_row or args.set_matrix):
        print("No modifications specified. Use --print-only to just display the matrix.")
        with np.printoptions(linewidth=120):
            print(arr)
        return

    # Backup and save
    if not args.no_backup:
        bak = p.with_suffix(p.suffix + ".bak")
        try:
            shutil.copy2(p, bak)
            print(f"Backup written: {bak}")
        except Exception as e:
            print(f"Warning: failed to create backup: {e}")

    try:
        np.save(p, arr.astype(orig_dtype, copy=False))
        print(f"Saved updated matrix to: {p}")
    except Exception as e:
        print(f"Failed to save {p}: {e}", file=sys.stderr)
        sys.exit(6)

    with np.printoptions(linewidth=120):
        print("New matrix:")
        print(arr)


if __name__ == "__main__":
    main()

