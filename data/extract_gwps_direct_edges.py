"""
One-time preprocessing for the GWPS gene network.

graph_data_gwps_rev1.csv is a 415 MB long-format 1428x1428 gene-gene effect table.
The DIRECT causal edges are the rows with path_length == 1 (non-diagonal); their
weight is the direct-effect coefficient G_hat (= G[parent, child] in the model
Y = YG + Xbeta + gamma). This script streams the big file once and writes the
small direct-edge list so GwpsGraph never has to touch the 415 MB file at runtime.

Output: data/gwps_direct_edges.csv  with columns  Exposure,Outcome,G_hat
Run:    python data/extract_gwps_direct_edges.py
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "graph_data_gwps_rev1.csv")
OUT = os.path.join(HERE, "gwps_direct_edges.csv")


def main():
    with open(SRC, newline="") as fin, open(OUT, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["Exposure", "Outcome", "G_hat"])
        kept = 0
        for row in reader:
            if row["path_length"] == "1" and row["Exposure"] != row["Outcome"]:
                writer.writerow([row["Exposure"], row["Outcome"], row["G_hat"]])
                kept += 1
    print(f"wrote {kept} direct edges to {OUT}")


if __name__ == "__main__":
    main()
