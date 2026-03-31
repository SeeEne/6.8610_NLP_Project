#!/usr/bin/env python3
"""Download all benchmark datasets and save as local JSONL files.

Usage:
    python scripts/download_data.py                    # download all
    python scripts/download_data.py humaneval mbpp     # download specific sources
    python scripts/download_data.py --output data/raw  # custom output directory
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.store import BenchmarkStore

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw"
ALL_SOURCES = ["humaneval", "mbpp", "ds1000"]


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "sources",
        nargs="*",
        default=ALL_SOURCES,
        metavar="SOURCE",
        help="Which benchmarks to download (default: all)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    store = BenchmarkStore()

    # Download from HuggingFace
    for source in args.sources:
        print(f"Downloading {source}...")
        n = store.load(source)
        print(f"  -> {n} tasks loaded")

    # Save to local JSONL
    print(f"\nSaving to {args.output}/")
    counts = store.save(args.output)
    for source, n in sorted(counts.items()):
        print(f"  {source}.jsonl — {n} tasks")

    # Verify round-trip
    print("\nVerifying round-trip...")
    reloaded = BenchmarkStore.load_local(args.output)
    assert len(reloaded) == len(store), (
        f"Round-trip mismatch: saved {len(store)}, reloaded {len(reloaded)}"
    )
    print(f"  OK — {len(reloaded)} tasks verified")

    print(f"\nDone. {repr(reloaded)}")


if __name__ == "__main__":
    main()
