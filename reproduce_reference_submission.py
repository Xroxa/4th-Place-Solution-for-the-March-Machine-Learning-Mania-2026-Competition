#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the exact archived winning submission CSV.")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"), help="Where to write the exact reference submission.")
    parser.add_argument("--reference", type=Path, default=Path("submission_reference.csv"), help="Bundled archived submission file.")
    args = parser.parse_args()

    if not args.reference.exists():
        raise FileNotFoundError(f"Reference submission not found: {args.reference}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.reference, args.output)
    print(f"Copied exact archived submission to {args.output}")


if __name__ == "__main__":
    main()
