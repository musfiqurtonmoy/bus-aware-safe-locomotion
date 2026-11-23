"""Compare two eval CSVs of the same run episode by episode (CHANGELOG #24).

    python scripts/compare_eval.py results/filter_ws_s0/eval_pre_eps.csv results/filter_ws_s0/eval.csv

Rows are matched on (test_mode, soc, episode). Every numeric column present in both files is
compared; columns present in only one file (e.g. counters added later) are listed, not
compared. Prints, per compared column, the number of episodes that differ and the largest
absolute difference, then a one-line verdict. Exit code 0 if all compared values are equal,
1 otherwise. Reads only; writes nothing.
"""
import sys

import numpy as np
import pandas as pd

KEYS = ["test_mode", "soc", "episode"]


def main(a_path, b_path):
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    for name, d in (("first", a), ("second", b)):
        if d.duplicated(KEYS).any():
            sys.exit(f"{name} file has duplicate {KEYS} rows")
    m = a.merge(b, on=KEYS, how="outer", suffixes=("_a", "_b"), indicator=True)
    unmatched = m[m["_merge"] != "both"]
    m = m[m["_merge"] == "both"]
    common = [c for c in a.columns if c in b.columns and c not in KEYS
              and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    only_a = sorted(set(a.columns) - set(b.columns))
    only_b = sorted(set(b.columns) - set(a.columns))
    print(f"episodes: first {len(a)}, second {len(b)}, matched {len(m)}, unmatched {len(unmatched)}")
    print(f"columns only in first: {only_a}")
    print(f"columns only in second: {only_b}")
    n_diff_total = 0
    for c in common:
        x, y = m[f"{c}_a"].to_numpy(float), m[f"{c}_b"].to_numpy(float)
        same = (x == y) | (np.isnan(x) & np.isnan(y))
        n = int((~same).sum())
        n_diff_total += n
        if n:
            print(f"  {c:16s} differs in {n:3d} episodes, max |diff| {np.nanmax(np.abs(x - y)):.3e}")
    print(f"compared {len(common)} numeric columns over {len(m)} episodes: "
          + ("ALL EQUAL" if n_diff_total == 0 and len(unmatched) == 0 else f"{n_diff_total} differing values"))
    return 0 if n_diff_total == 0 and len(unmatched) == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
