"""One-time preprocessing: streams the 415 MB graph_data_gwps_rev1.csv once to extract direct edges (path_length == 1)
into data/gwps_direct_edges.csv, so GwpsGraph never touches the big file at runtime."""

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
