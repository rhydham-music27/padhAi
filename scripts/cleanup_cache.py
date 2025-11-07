from pathlib import Path
import argparse
import shutil

CACHED_DIRS = [
    Path("~/.cache/huggingface").expanduser(),
    Path("~/.cache/torch").expanduser(),
    Path(".cache").resolve(),
]


def cleanup(dry_run: bool = True) -> None:
    for d in CACHED_DIRS:
        if d.exists():
            if dry_run:
                print(f"[DRY RUN] Would remove: {d}")
            else:
                print(f"Removing: {d}")
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    print(f"Failed to remove {d}: {e}")
        else:
            print(f"Not found: {d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cleanup local caches (HF/Torch)")
    ap.add_argument("--apply", action="store_true", help="Actually delete caches (dangerous)")
    args = ap.parse_args()
    cleanup(dry_run=not args.apply)
