import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "so_survey_2025.csv"
INITIAL = ROOT / "data" / "initial"
INCOMING = ROOT / "data" / "incoming"
MANIFEST = ROOT / "data" / "manifest.csv"
SOURCE = "Stack Overflow Developer Survey 2025 (ODbL), so_survey_2025.csv"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    params = yaml.safe_load((ROOT / "params.yaml").read_text())["split"]
    # читаем всё как строки, чтобы порции совпадали с сырым файлом по значениям
    df = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    rng = np.random.default_rng(params["seed"])
    order = rng.permutation(len(df))
    n_initial = int(round(len(df) * params["initial_share"]))
    parts = {INITIAL / "initial.csv": order[:n_initial]}
    for i, idx in enumerate(np.array_split(order[n_initial:], params["n_batches"]), 1):
        parts[INCOMING / f"batch_{i:02d}.csv"] = idx

    INITIAL.mkdir(parents=True, exist_ok=True)
    INCOMING.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, idx in parts.items():
        df.iloc[np.sort(idx)].to_csv(path, index=False)
        rows.append({"file": str(path.relative_to(ROOT)), "rows": len(idx),
                     "md5": md5(path), "source": SOURCE})
        print(f"{path.relative_to(ROOT)}: {len(idx)} строк")
    pd.DataFrame(rows).to_csv(MANIFEST, index=False)
    print("всего:", sum(r["rows"] for r in rows), "из", len(df))


if __name__ == "__main__":
    main()
