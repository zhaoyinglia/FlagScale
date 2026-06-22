# Copyright (c) 2025, BAAI. All rights reserved.
# Convert Bagel T2I parquet data to WebDataset tar format for Energon.
#
# Input format:
#   - Parquet files with columns: image (bytes), caption (str), [metadata]
#
# Output format:
#   WebDataset tar shards, each sample contains:
#     - __key__: unique sample id
#     - jpg: raw image bytes
#     - txt: caption text
#     - json: metadata (optional)
#
# Usage:
#   python convert_t2i_to_wds.py \
#     --data_dirs /data/t2i/laion /data/t2i/internal \
#     --output_dir /data/t2i/wds \
#     --max_count 10000

import argparse
import io
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import webdataset as wds
from tqdm import tqdm


def convert_t2i_to_wds(
    data_dirs: list,
    output_dir: str,
    max_count: int = 10000,
    max_size: float = 3e9,
    image_column: str = "image",
    caption_column: str = "caption",
):
    """Convert T2I parquet data to WebDataset tar format.

    Each sample in the tar contains:
      - __key__: "{global_idx:09d}"
      - jpg: raw image bytes
      - txt: caption string
      - json: any additional metadata columns
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all parquet files
    parquet_files = []
    for data_dir in data_dirs:
        if os.path.isdir(data_dir):
            for f in sorted(os.listdir(data_dir)):
                if f.endswith('.parquet'):
                    parquet_files.append(os.path.join(data_dir, f))
        elif os.path.isfile(data_dir) and data_dir.endswith('.parquet'):
            parquet_files.append(data_dir)

    if not parquet_files:
        print("Error: no parquet files found")
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet files")

    global_idx = 0
    shard_pattern = os.path.join(output_dir, "t2i-%06d.tar")

    with wds.ShardWriter(shard_pattern, maxcount=max_count, maxsize=max_size) as sink:
        for pq_path in tqdm(parquet_files, desc="Processing parquet files"):
            try:
                table = pq.read_table(pq_path)
            except Exception as e:
                print(f"Warning: failed to read {pq_path}: {e}")
                continue

            columns = table.column_names
            num_rows = len(table)

            for row_idx in range(num_rows):
                # Extract image bytes
                image_data = table.column(image_column)[row_idx].as_py()
                if image_data is None:
                    continue

                # Handle different image storage formats
                if isinstance(image_data, dict):
                    # Some parquet formats store as {"bytes": ..., "path": ...}
                    image_bytes = image_data.get('bytes', None)
                elif isinstance(image_data, bytes):
                    image_bytes = image_data
                else:
                    continue

                if image_bytes is None or len(image_bytes) == 0:
                    continue

                # Extract caption
                caption = table.column(caption_column)[row_idx].as_py() or ""

                # Extract metadata (all other columns)
                metadata = {}
                for col in columns:
                    if col not in (image_column, caption_column):
                        val = table.column(col)[row_idx].as_py()
                        if val is not None:
                            metadata[col] = val

                sample = {
                    "__key__": f"{global_idx:09d}",
                    "jpg": image_bytes,
                    "txt": caption.encode("utf-8") if isinstance(caption, str) else caption,
                }
                if metadata:
                    sample["json"] = json.dumps(metadata).encode("utf-8")

                sink.write(sample)
                global_idx += 1

    print(f"Done. Wrote {global_idx} samples to {output_dir}")
    print(f"Run 'energon prepare {output_dir} --sample-type CrudeWebdataset' to finalize.")


def main():
    parser = argparse.ArgumentParser(description="Convert Bagel T2I parquet data to WebDataset tar format")
    parser.add_argument("--data_dirs", nargs="+", required=True,
                        help="Directories containing parquet files")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for tar shards")
    parser.add_argument("--max_count", type=int, default=10000,
                        help="Max samples per shard")
    parser.add_argument("--max_size", type=float, default=3e9,
                        help="Max shard size in bytes (default 3GB)")
    parser.add_argument("--image_column", default="image",
                        help="Column name for image data")
    parser.add_argument("--caption_column", default="caption",
                        help="Column name for caption text")
    args = parser.parse_args()

    convert_t2i_to_wds(
        data_dirs=args.data_dirs,
        output_dir=args.output_dir,
        max_count=args.max_count,
        max_size=args.max_size,
        image_column=args.image_column,
        caption_column=args.caption_column,
    )


if __name__ == "__main__":
    main()
