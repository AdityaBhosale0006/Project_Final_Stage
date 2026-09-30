"""
Stage 0 only: scan a PDF, detect every table on every page, and save
every detected crop + the manifest -- nothing more.

This intentionally stops BEFORE any vocab/schema matching. That logic
lives entirely in extract_page.py's classify_table() / _classify_and_prepare(),
a separate later stage that this script never imports or calls. Nothing
here discards a detected table for any reason (content similarity,
column count, banner match, etc.) -- every table detect_tables_on_page()
finds gets rendered and written to disk, full stop.

Usage:
    python detect_only.py path/to/file.pdf output_dir
    python detect_only.py path/to/file.pdf output_dir --dpi=600 --preview-dpi=150
"""

import sys
import json
from pathlib import Path

from detect_tables import detect_and_render


def main():
    if len(sys.argv) < 3:
        print("Usage: python detect_only.py <pdf_path> <output_dir> "
              "[--dpi=600] [--preview-dpi=300] [--min-lines=3] [--padding=6.0]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_dir = sys.argv[2]

    # simple flag parsing -- kept local to this script so it can't
    # silently swallow flags the way extract_page.py's CLI does
    kwargs = {}
    for arg in sys.argv[3:]:
        if arg.startswith("--dpi="):
            kwargs["dpi"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--preview-dpi="):
            kwargs["preview_dpi"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--min-lines="):
            kwargs["min_lines"] = int(arg.split("=", 1)[1])
        elif arg.startswith("--padding="):
            kwargs["padding_pt"] = float(arg.split("=", 1)[1])

    if not Path(pdf_path).exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[Stage 0] Detecting tables in {pdf_path} ...")
    manifest = detect_and_render(pdf_path, output_dir, **kwargs)

    by_page = {}
    for entry in manifest:
        by_page.setdefault(entry["page"], []).append(entry)

    print(f"\n{len(manifest)} table(s) detected across {len(by_page)} page(s):")
    for page, entries in sorted(by_page.items()):
        methods = {}
        for e in entries:
            methods[e["method"]] = methods.get(e["method"], 0) + 1
        method_str = ", ".join(f"{m}={c}" for m, c in sorted(methods.items()))
        print(f"  page {page}: {len(entries)} table(s) ({method_str})")

    tables_dir = Path(output_dir) / "tables"
    manifest_path = Path(output_dir) / "tables.json"
    print(f"\nEvery crop saved to: {tables_dir}/")
    print(f"Manifest saved to:   {manifest_path}")
    print("\nNo classification, no vocab/schema matching, no discarding "
          "happened here -- every detected table above is on disk.")


if __name__ == "__main__":
    main()
