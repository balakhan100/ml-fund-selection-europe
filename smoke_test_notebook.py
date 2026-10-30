"""
End-to-end smoke test: generate synthetic raw data, then execute every code cell of
notebooks/fund_selection.ipynb in order (quick model grids). No Jupyter needed.

    python tests/smoke_test_notebook.py
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="fundml_smoke_"))
    os.environ.update({"FUNDML_DATA_DIR": str(tmp / "data"), "FUNDML_OUTPUT_DIR": str(tmp / "outputs"),
                       "FUNDML_QUICK": "1"})
    sys.path.insert(0, str(ROOT / "tests"))
    from synthetic_data import make
    make(tmp / "data")

    os.chdir(ROOT)
    nb = json.loads((ROOT / "notebooks" / "fund_selection.ipynb").read_text())
    ns = {"__name__": "__main__"}
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, cell in enumerate(code_cells, 1):
        t = time.time()
        exec(compile("".join(cell["source"]), f"<cell {i}>", "exec"), ns)
        print(f"[cell {i}/{len(code_cells)} ok, {time.time() - t:.1f}s]")
    print("\nsmoke test passed; outputs in", tmp / "outputs")
