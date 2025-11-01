#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np


def print_array(name: str, arr: np.ndarray, full: bool, limit: int | None):
    print(f"=== {name} ===")
    print(f"dtype: {arr.dtype}")
    print(f"shape: {arr.shape}")

    if not full and limit is None:
        # Default summary printing
        with np.printoptions(edgeitems=3, threshold=100, linewidth=120):
            print(arr)
        return

    if limit is not None and arr.size > limit:
        flat = arr.ravel()
        first = flat[: min(limit, 10)]
        last = flat[-min(limit, 10) :]
        print(f"showing first {len(first)} and last {len(last)} of {arr.size} elements:")
        with np.printoptions(linewidth=120):
            print(first)
            if arr.size > len(first) + len(last):
                print("...")
            print(last)
        return

    # Full printing
    with np.printoptions(threshold=np.inf, linewidth=120):
        print(arr)


def main():
    parser = argparse.ArgumentParser(
        description="Print information and contents of a .npy (or .npz) file."
    )
    parser.add_argument("path", help="Path to .npy or .npz file")
    parser.add_argument(
        "--full",
        default=True,
        help="Print entire array contents without truncation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For large arrays, show only first/last N elements (overrides summary)",
    )

    args = parser.parse_args()
    p = Path(args.path)
    if not p.exists():
        print(f"File not found: {p}", file=sys.stderr)
        sys.exit(1)

    try:
        if p.suffix.lower() == ".npz":
            with np.load(p, allow_pickle=False) as data:
                if not data.files:
                    print("Empty .npz archive")
                    return
                for key in data.files:
                    arr = data[key]
                    print_array(key, arr, args.full, args.limit)
        else:
            arr = np.load(p, allow_pickle=False)
            print_array(p.name, arr, args.full, args.limit)
    except Exception as e:
        print(f"Failed to load {p}: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

