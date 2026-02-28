from pathlib import Path

ROOT = Path("data")   # 换成你的 data 也可以
TARGET_DIRS = ["zed_left", "zed_right"]
TARGET_DOC = "head_pos.txt"
CUT_FRONT = 36
CUT_BACK = 20


def _parse_indexed_rows(txt_path: Path):
    rows = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            tokens = raw.split()
            try:
                idx = int(round(float(tokens[0])))
            except ValueError:
                # Skip non-numeric header lines
                continue
            rows.append((idx, tokens))
    return rows


def _rename_with_mapping(files, index_map):
    temp_paths = []
    for f in files:
        stem = f.stem
        try:
            idx = int(stem)
        except ValueError:
            continue
        if idx not in index_map:
            f.unlink()
            continue
        tmp = f.with_name(f"__tmp_{index_map[idx]:06d}{f.suffix}")
        f.rename(tmp)
        temp_paths.append(tmp)
    for t in temp_paths:
        new_name = t.with_name(t.name.replace("__tmp_", ""))
        t.rename(new_name)


for timestamp_dir in ROOT.iterdir():
    if not timestamp_dir.is_dir():
        continue

    print(f"Processing: {timestamp_dir.name}")

    target_doc = timestamp_dir / TARGET_DOC
    if not target_doc.exists():
        print(f"  Skip {TARGET_DOC}, not found")
        continue

    rows = _parse_indexed_rows(target_doc)
    if not rows:
        print(f"  Skip {TARGET_DOC}, no valid rows")
        continue

    rows.sort(key=lambda x: x[0])
    keep_rows = rows[CUT_FRONT : len(rows) - CUT_BACK]
    if not keep_rows:
        print(f"  Skip {TARGET_DOC}, nothing to keep after trim")
        continue

    index_map = {old_idx: new_idx for new_idx, (old_idx, _) in enumerate(keep_rows)}

    # Update target doc with new indices
    with target_doc.open("w", encoding="utf-8") as f:
        for old_idx, tokens in keep_rows:
            tokens[0] = f"{index_map[old_idx]}"
            f.write(" ".join(tokens) + "\n")

    for sub in TARGET_DIRS:
        d = timestamp_dir / sub
        if not d.exists():
            print(f"  Skip {sub}, not found")
            continue

        files = sorted(d.glob("*.*"))
        if not files:
            continue

        _rename_with_mapping(files, index_map)

    print(f"  kept {len(keep_rows)}, removed {len(rows) - len(keep_rows)}")
