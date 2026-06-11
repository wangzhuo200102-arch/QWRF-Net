"""Create a SEVIR VIL sample index for QWRF-Net."""

from __future__ import annotations

import argparse

from qwrfnet.dataset import prepare_dataset_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True, help="Path to the SEVIR VIL directory.")
    parser.add_argument("--output", default="data/sevir_valid_samples.json", help="Output JSON index file.")
    parser.add_argument("--max-vil", type=float, default=255.0)
    args = parser.parse_args()

    valid = prepare_dataset_index(args.data_path, args.max_vil, args.output)
    print(f"Saved {len(valid)} valid samples to {args.output}")


if __name__ == "__main__":
    main()

