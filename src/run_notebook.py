"""Execute a notebook's code cells with live output.

nbconvert buffers everything until a cell finishes, so a long cell is
indistinguishable from a hang. This runs the same code in-process and prints
as it goes, with no cell timeout.

Usage:  python src/run_notebook.py notebooks/04_modeling_and_tuning.ipynb
        python src/run_notebook.py notebooks/04_modeling_and_tuning.ipynb 5
                                                         (start at cell 5)
"""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    path = Path(sys.argv[1]).resolve()
    start_at = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

    import matplotlib
    matplotlib.use("Agg")          # write figures to disk, open no windows

    ns = {"__name__": "__main__"}
    root = path.parent.parent
    sys.path.insert(0, str(root / "src"))

    print(f"{path.name}: {len(cells)} code cells\n", flush=True)
    began = time.perf_counter()

    for i, cell in enumerate(cells, 1):
        if i < start_at:
            print(f"[{i}/{len(cells)}] skipped", flush=True)
            continue
        source = "".join(cell["source"])
        head = next((l for l in source.splitlines() if l.strip()), "")[:60]
        print(f"[{i}/{len(cells)}] {head}", flush=True)
        t0 = time.perf_counter()
        try:
            exec(compile(source, f"{path.name}#cell{i}", "exec"), ns)
        except Exception:
            import traceback
            print(f"\n--- cell {i} failed ---", flush=True)
            traceback.print_exc()
            return 1
        print(f"      done in {time.perf_counter() - t0:.1f}s\n", flush=True)

    print(f"finished in {time.perf_counter() - began:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
