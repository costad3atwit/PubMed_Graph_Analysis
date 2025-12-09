"""
Detect drug—disease co-mentions in PubMed papers.
NOW WITH TERM SPECIFICITY FILTERING to avoid overly-general terms.

Inputs:
  - papers.csv
  - terms.csv

Output:
  - co-mentions.csv: pmid,title,drug_ui,drug_name,disease_ui,disease_name,where_found
"""

import csv, re, string
import ahocorasick

# ---------- CONFIG ----------
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "Data"
FIGURES_DIR = PROJECT_ROOT / "Figures"
LOGS_DIR = PROJECT_ROOT / "Logs"
GRAPHS_DIR = PROJECT_ROOT / "Graphs"

TERMS_FILE = DATA_DIR / "terms.csv"
PAPERS_FILE = DATA_DIR / "papers.csv"
OUT_FILE = DATA_DIR / "co-mentions.csv"

# ---------- CONFIG: TERM SPECIFICITY FILTERING ----------
MIN_TREE_DEPTH = 2  # Minimum dots in tree number (e.g., C10.228 = depth 2)
                    # This filters out very general terms like "Disease" (C = depth 0)

# Optional: Blacklist specific known-generic terms by name (case-insensitive)
BLACKLIST_TERMS = {
    'disease',
    'diseases', 
    'disorder',
    'disorders',
    'syndrome',
    'syndromes',
    'pharmaceutical preparations',
    'therapeutic use'
}

# ---------- HELPERS ----------
def normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()

def detect_type(tree_nums):
    for tn in tree_nums.split(","):
        if tn.strip().startswith("C"):
            return "disease"
    return "drug"

def get_max_tree_depth(tree_nums):
    """
    Calculate maximum tree depth (number of dots) for a term.
    Higher depth = more specific term.
    Examples:
      C           -> depth 0 (very general: "Disease")
      C10         -> depth 1 (general: "Nervous System Diseases")  
      C10.228     -> depth 2 (specific: "Central Nervous System Diseases")
      C10.228.140 -> depth 3 (very specific: "Alzheimer Disease")
    """
    if not tree_nums:
        return 0
    
    max_depth = 0
    for tree_num in tree_nums.split(","):
        tree_num = tree_num.strip()
        depth = tree_num.count(".")
        max_depth = max(max_depth, depth)
    
    return max_depth

def passes_specificity_filter(term_name, tree_nums):
    """
    Check if a term is specific enough to include.
    Returns True if term should be INCLUDED, False if too general.
    """
    # Check blacklist first
    if term_name.lower() in BLACKLIST_TERMS:
        return False
    
    # Check tree depth
    depth = get_max_tree_depth(tree_nums)
    if depth < MIN_TREE_DEPTH:
        return False
    
    return True

# ---------- STEP 1: LOAD TERMS WITH FILTERING ----------
print("=" * 70)
print("DETECTING CO-MENTIONS WITH TERM SPECIFICITY FILTERING")
print("=" * 70)
print(f"\nFiltering criteria:")
print(f"  - Minimum tree depth: {MIN_TREE_DEPTH}")
print(f"  - Blacklist size: {len(BLACKLIST_TERMS)} terms")

terms = []
mesh_info = {}  # maps MeSH ID → (term, type)
filtered_out = {"blacklist": 0, "tree_depth": 0, "kept": 0}

with open(TERMS_FILE, encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        mesh_id = (row.get("id") or "").strip()
        if not mesh_id:
            continue
        term_name = (row.get("term") or "").strip()
        if not term_name:
            continue
        
        tree_nums = row.get("tree_nums", "")
        typ = detect_type(tree_nums)
        
        # Apply specificity filter
        if not passes_specificity_filter(term_name, tree_nums):
            # Track why it was filtered
            if term_name.lower() in BLACKLIST_TERMS:
                filtered_out["blacklist"] += 1
            else:
                filtered_out["tree_depth"] += 1
            continue  # Skip this term
        
        # Term passed filter - include it
        filtered_out["kept"] += 1
        mesh_info[mesh_id] = (term_name, typ)

        base = [term_name]
        entries_field = row.get("entries", "") or ""
        entries = re.split(r"[\n;]", entries_field)
        for e in entries:
            e = e.strip()
            if not e:
                continue
            base.append(e.split("|", 1)[0].strip())
        for term in base:
            if term:
                terms.append((normalize(term), mesh_id, typ))

print(f"\nTerm filtering results:")
print(f"  ✓ Kept: {filtered_out['kept']:,} terms")
print(f"  ✗ Filtered by blacklist: {filtered_out['blacklist']:,} terms")
print(f"  ✗ Filtered by tree depth: {filtered_out['tree_depth']:,} terms")
print(f"  Total filtered: {filtered_out['blacklist'] + filtered_out['tree_depth']:,} terms")

# Build automaton
A = ahocorasick.Automaton()
for surface, mid, typ in terms:
    if surface not in A:
        A.add_word(surface, (mid, typ))
A.make_automaton()
print(f"\nLoaded {len(A):,} unique term surfaces into matcher.")

# ---------- STEP 2: SCAN PAPERS ----------
print("\nScanning papers for co-mentions...")

def find_mentions(text):
    diseases, drugs = set(), set()
    for _, (mid, typ) in A.iter(normalize(text)):
        if typ == "disease":
            diseases.add(mid)
        else:
            drugs.add(mid)
    return diseases, drugs

rows_out = []

with open(PAPERS_FILE, encoding="utf-8") as pf:
    rdr = csv.DictReader(pf)
    for paper in rdr:
        pmid = paper.get("pmid", "")
        title = paper.get("title", "") or ""
        abstract = paper.get("abstract", "") or ""

        d_t, dr_t = find_mentions(title)
        d_a, dr_a = find_mentions(abstract)

        diseases = d_t | d_a
        drugs = dr_t | dr_a
        if not diseases or not drugs:
            continue

        for dis in diseases:
            for drug in drugs:
                in_title = dis in d_t or drug in dr_t
                in_abs = dis in d_a or drug in dr_a
                if in_title and in_abs:
                    where = "both"
                elif in_title:
                    where = "title"
                else:
                    where = "abstract"

                drug_name = mesh_info.get(drug, ("", ""))[0]
                disease_name = mesh_info.get(dis, ("", ""))[0]
                rows_out.append([
                    pmid,
                    title.strip(),
                    drug,
                    drug_name,
                    dis,
                    disease_name,
                    where
                ])

print(f"\n✓ Found {len(rows_out):,} co-mentions.")

# ---------- STEP 3: WRITE OUTPUT ----------
with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "pmid",
        "title",
        "drug_ui",
        "drug_name",
        "disease_ui",
        "disease_name",
        "where_found"
    ])
    w.writerows(rows_out)

print(f"✓ Wrote {len(rows_out):,} rows to {OUT_FILE.name}")
print("=" * 70)