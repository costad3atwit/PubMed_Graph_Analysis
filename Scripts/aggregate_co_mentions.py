"""
Aggregate co-mentions by (drug_ui, disease_ui).
Includes drug_name and disease_name for readability.
"""

import csv
from collections import defaultdict

# ---------- CONFIG ----------
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "Data"
FIGURES_DIR = PROJECT_ROOT / "Figures"
LOGS_DIR = PROJECT_ROOT / "Logs"
GRAPHS_DIR = PROJECT_ROOT / "Graphs"

CO_FILE = DATA_DIR / "co-mentions.csv"
PAPERS_FILE = DATA_DIR / "papers.csv"
OUT_FILE = DATA_DIR / "aggregated.csv"

# ---------- STEP 1: LOAD PAPER YEARS ----------
paper_year = {}
with open(PAPERS_FILE, encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        pmid = row.get("pmid", "").strip()
        year = row.get("year", "").strip()
        if pmid and year.isdigit():
            paper_year[pmid] = int(year)

# ---------- STEP 2: AGGREGATE ----------
agg = defaultdict(lambda: {
    "count": 0,
    "years": set(),
    "drug_name": "",
    "disease_name": ""
})

with open(CO_FILE, encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        drug = row["drug_ui"].strip()
        dis = row["disease_ui"].strip()
        drug_name = row.get("drug_name", "").strip()
        dis_name = row.get("disease_name", "").strip()
        pmid = row["pmid"].strip()
        if not (drug and dis and pmid):
            continue
        key = (drug, dis)
        agg[key]["count"] += 1
        agg[key]["drug_name"] = drug_name
        agg[key]["disease_name"] = dis_name
        y = paper_year.get(pmid)
        if y:
            agg[key]["years"].add(y)

# ---------- STEP 3: SORT AND WRITE ----------
sorted_agg = sorted(
    agg.items(),
    key=lambda kv: (kv[1]["count"], max(kv[1]["years"]) if kv[1]["years"] else 0),
    reverse=True
)

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "drug_ui",
        "drug_name",
        "disease_ui",
        "disease_name",
        "paper_count",
        "first_paper",
        "last_paper"
    ])
    for (drug, dis), data in sorted_agg:
        years = sorted(data["years"])
        first_y = years[0] if years else ""
        last_y = years[-1] if years else ""
        w.writerow([
            drug,
            data["drug_name"],
            dis,
            data["disease_name"],
            data["count"],
            first_y,
            last_y
        ])

print(f"Wrote {len(sorted_agg)} aggregated pairs to {OUT_FILE}")
