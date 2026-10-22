"""Create a non-destructive canonical target dataset excluding reviewed samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .targets import TARGET_DATASET_FORMAT


def curate_targets(
    source: str | Path,
    output: str | Path,
    *,
    exclude_lines: tuple[int, ...] = (),
    exclude_sample_ids: tuple[str, ...] = (),
) -> tuple[int, int]:
    source_path = Path(source)
    output_path = Path(output)
    line_set = set(exclude_lines)
    id_set = set(exclude_sample_ids)
    if any(line < 1 for line in line_set):
        raise ValueError("excluded line numbers must be positive")
    if len(line_set) != len(exclude_lines) or len(id_set) != len(exclude_sample_ids):
        raise ValueError("exclusion lines and sample IDs must be unique")

    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    kept: list[str] = []
    seen_ids: set[str] = set()
    excluded_ids: set[str] = set()
    excluded_lines: set[int] = set()
    excluded_count = 0
    for line_number, line in enumerate(source_lines, 1):
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source_path}:{line_number}: invalid JSON") from error
        if not isinstance(sample, dict) or sample.get("format") != TARGET_DATASET_FORMAT:
            raise ValueError(f"{source_path}:{line_number}: expected canonical target sample")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{source_path}:{line_number}: missing sample_id")
        if sample_id in seen_ids:
            raise ValueError(f"{source_path}:{line_number}: duplicate sample_id {sample_id!r}")
        seen_ids.add(sample_id)
        excluded = line_number in line_set or sample_id in id_set
        if excluded:
            excluded_count += 1
            excluded_ids.add(sample_id)
            if line_number in line_set:
                excluded_lines.add(line_number)
        else:
            kept.append(line)

    missing_lines = line_set - excluded_lines
    missing_ids = id_set - excluded_ids
    if missing_lines or missing_ids:
        missing = [*(f"line {line}" for line in sorted(missing_lines)),
                   *(f"sample {sample_id}" for sample_id in sorted(missing_ids))]
        raise ValueError("requested exclusions were not found: " + ", ".join(missing))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves the source and refuses overwrites.
    with output_path.open("x", encoding="utf-8") as stream:
        if kept:
            stream.write("\n".join(kept) + "\n")
    return len(kept), excluded_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="raw canonical target JSONL (never modified)")
    parser.add_argument("--output", required=True, type=Path, help="new curated JSONL destination")
    parser.add_argument("--exclude-line", action="append", type=int, default=[],
                        help="source line to omit; may be repeated")
    parser.add_argument("--exclude-sample-id", action="append", default=[],
                        help="sample ID to omit; may be repeated")
    args = parser.parse_args()
    if not args.exclude_line and not args.exclude_sample_id:
        parser.error("specify at least one --exclude-line or --exclude-sample-id")
    try:
        kept, excluded = curate_targets(
            args.source, args.output,
            exclude_lines=tuple(args.exclude_line),
            exclude_sample_ids=tuple(args.exclude_sample_id),
        )
        print(f"created {args.output}: kept {kept} samples; excluded {excluded}")
    except (OSError, ValueError) as error:
        print(f"curate_targets: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
