#!/usr/bin/env sh
set -eu

# Downsample images in data/{cup,small_book,pot}/{left,right} to 1/10.
# Copies sampled images into a parallel tree under data_downsampled.

BASE_ROOT="data"
OUT_ROOT="data_downsampled"
FACTOR=10  # keep every Nth frame (0-based)

for category in cup small_book pot; do
  for side in left right; do
    src_dir="$BASE_ROOT/$category/$side"
    if [ ! -d "$src_dir" ]; then
      echo "[skip] $src_dir (missing)"
      continue
    fi

    out_dir="$OUT_ROOT/$category/$side"
    mkdir -p "$out_dir"
    rm -f "$out_dir"/*

    # Collect files sorted; if none, skip.
    files=$(find "$src_dir" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort)
    if [ -z "$files" ]; then
      echo "[skip] no images in $src_dir"
      continue
    fi

    idx=0       # counts all input frames
    kept=0      # counts saved frames; filenames start at 0
    old_ifs=$IFS
    IFS='
'
    for f in $files; do
      [ -e "$f" ] || continue
      if [ $((idx % FACTOR)) -eq 0 ]; then
        # keep and rename starting from 0: left000000.png, right000000.jpg, ...
        ext="${f##*.}"
        new_name=$(printf "%s%06d.%s" "$side" "$kept" "$ext")
        cp "$f" "$out_dir/$new_name"
        kept=$((kept + 1))
      fi
      idx=$((idx + 1))
    done
    IFS=$old_ifs

    echo "[done] $src_dir -> $out_dir (kept $kept / $idx)"
  done
done
