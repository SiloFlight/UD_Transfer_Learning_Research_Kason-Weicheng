#!/usr/bin/env python3
from pathlib import Path
import ast
import json
import sys

def load_results(results_dir: Path) -> dict:
    """
    results_dir contains files named like: model_code-eval_code.txt
    Each file contains a printed Python dict.
    Returns: {model_code: {eval_code: metrics_dict}}
    """
    combined = {}
    for fp in results_dir.glob("*.txt"):
        name = fp.stem  # e.g., "en_ewt-en_ewt"
        # split model_code and eval_code (robust to extra '-' by splitting from the right)
        if "-" not in name:
            # skip files that don't match the pattern
            continue
        model_code, eval_code = name.rsplit("-", 1)

        try:
            text = fp.read_text().strip()
            # parse safely (handles printed dicts)
            metrics = ast.literal_eval(text)
            if not isinstance(metrics, dict):
                raise ValueError("File did not contain a dict")
        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: {e}", file=sys.stderr)
            continue

        combined.setdefault(model_code, {})[eval_code] = metrics

    return combined

if __name__ == "__main__":
    # allow optional custom directory: `python combine.py /path/to/results`
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    if not results_dir.is_dir():
        print(f"[ERROR] Results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    combined = load_results(results_dir)

    # pretty-print to stdout
    print(json.dumps(combined, indent=2, sort_keys=True))

    # also write to JSON file next to the folder
    out_path = results_dir.parent / "combined_results.json"
    out_path.write_text(json.dumps(combined, indent=2, sort_keys=True))
    print(f"[INFO] Wrote {out_path}")
