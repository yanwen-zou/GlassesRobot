"""Push a local LeRobot dataset to Hugging Face Hub.

Example:
  python baseline/openpi/scripts/push_lerobot_to_hub.py \
    --local-path data/book_openpi \
    --hub-repo-id your_hf_name/book_openpi
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@dataclass(frozen=True)
class Args:
    # Local path to LeRobot dataset root (contains meta/info.json).
    local_path: str
    # Target Hugging Face dataset repo id, e.g. "username/dataset_name".
    hub_repo_id: str
    # If true, create/update private dataset repo.
    private: bool = False
    # If true, only validate local dataset structure and exit.
    dry_run: bool = False


def main(args: Args) -> None:
    local_path = Path(args.local_path).expanduser().resolve()
    info_path = local_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Invalid LeRobot dataset path: missing {info_path}")

    print(f"Loading local LeRobot dataset from: {local_path}")
    dataset = LeRobotDataset(args.hub_repo_id, root=local_path)
    print(f"Loaded dataset length: {len(dataset)}")

    if args.dry_run:
        print("Dry run enabled. Skip pushing to hub.")
        return

    print(f"Pushing to Hub repo: {args.hub_repo_id} (private={args.private})")
    dataset.push_to_hub(private=args.private)
    print("Push completed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
