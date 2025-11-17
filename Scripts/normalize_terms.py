"""
Normalize terms using BioWordVec embeddings to merge synonyms.
This script clusters similar biomedical terms based on cosine similarity
of their word embeddings and creates a canonical mapping to reduce redundancy.

Inputs:
  - terms.csv
  - aggregated.csv
  - BioWordVec_PubMed_MIMICIII_d200.vec.bin

Outputs:
  - term_embeddings.csv: terms with their embedding coverage
  - synonym_clusters.csv: clusters of similar terms
  - canonical_mapping.csv: mapping from original to canonical term IDs
  - aggregated_canonical.csv: re-aggregated co-mentions using canonical terms
"""

import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import time
import re

# ---------- CONFIG ----------
DATA_DIR = Path("C:/Users/sirda/Dropbox (Personal)/Documents/Fall '25/Data Mining/PubMed Project/Data")
TERMS_FILE = DATA_DIR / "terms.csv"
AGG_FILE = DATA_DIR / "aggregated.csv"
EMBEDDINGS_FILE = DATA_DIR / "BioWordVec_PubMed_MIMICIII_d200.vec.bin"

# Output files
TERM_EMB_OUT = DATA_DIR / "term_embeddings.csv"
CLUSTERS_OUT = DATA_DIR / "synonym_clusters.csv"
MAPPING_OUT = DATA_DIR / "canonical_mapping.csv"
CANONICAL_AGG_OUT = DATA_DIR / "aggregated_canonical.csv"

# Clustering parameters
SIMILARITY_THRESHOLD = 0.85  # Cosine similarity threshold for clustering
DISTANCE_THRESHOLD = 1 - SIMILARITY_THRESHOLD  # Convert to distance

print("=" * 60)
print("TERM NORMALIZATION USING BIOWORDVEC EMBEDDINGS")
print("=" * 60)

start_time = time.time()

# ---------- STEP 1: LOAD BIOWORDVEC ----------
print("\n[1/7] Loading BioWordVec embeddings...")
print(f"      File: {EMBEDDINGS_FILE}")
print("      This may take 2-3 minutes for the 13GB file...")

try:
    model = KeyedVectors.load_word2vec_format(str(EMBEDDINGS_FILE), binary=True)
    print(f"      ✓ Loaded successfully!")
    print(f"      Vocabulary size: {len(model):,}")
    print(f"      Vector dimensions: {model.vector_size}")
except Exception as e:
    print(f"      ✗ Error loading embeddings: {e}")
    print("      Make sure BioWordVec_PubMed_MIMICIII_d200.vec.bin is in the Data directory")
    exit(1)

# ---------- STEP 2: LOAD TERMS ----------
print("\n[2/7] Loading terms from terms.csv...")
terms_df = pd.read_csv(TERMS_FILE)
print(f"      Loaded {len(terms_df):,} terms")

# ---------- HELPER FUNCTIONS ----------
def normalize_term(term):
    """
    Normalize term to match BioWordVec format.
    BioWordVec uses lowercase with underscores.
    """
    if pd.isna(term):
        return ""
    # Convert to lowercase, strip whitespace
    normalized = str(term).lower().strip()
    # Replace spaces with underscores
    normalized = normalized.replace(" ", "_")
    # Remove extra punctuation that might interfere
    normalized = re.sub(r'[^\w_-]', '', normalized)
    return normalized

def extract_aliases(entry):
    """
    Extract aliases from the entries field.
    Handles both newline and pipe-separated formats.
    """
    if pd.isna(entry):
        return []
    aliases = []
    # Split by newline first
    for line in str(entry).split("\n"):
        # Then split by pipe and take first element
        parts = line.split("|")
        if parts and parts[0].strip():
            aliases.append(parts[0].strip())
    return aliases

def get_embedding(term, model):
    """
    Try to get embedding for a term with multiple fallback strategies.
    """
    if not term:
        return None
    
    # Strategy 1: Direct lookup (with underscores)
    normalized = normalize_term(term)
    if normalized in model:
        return model[normalized]
    
    # Strategy 2: Try without underscores (single word)
    no_underscore = normalized.replace("_", "")
    if no_underscore in model:
        return model[no_underscore]
    
    # Strategy 3: Try with spaces instead of underscores
    with_spaces = normalized.replace("_", " ")
    if with_spaces in model:
        return model[with_spaces]
    
    # Strategy 4: Try first word only for multi-word terms
    if "_" in normalized:
        first_word = normalized.split("_")[0]
        if first_word in model and len(first_word) > 2:
            return model[first_word]
    
    return None

# ---------- STEP 3: NORMALIZE AND MAP TERMS ----------
print("\n[3/7] Normalizing terms and finding embeddings...")

# Add normalized column
terms_df["normalized"] = terms_df["term"].apply(normalize_term)

# Extract aliases
print("      Extracting aliases from entries field...")
terms_df["aliases"] = terms_df["entries"].apply(extract_aliases)

# Map terms to embeddings
term_embeddings = {}
coverage_stats = {"found": 0, "not_found": 0, "found_via_alias": 0}

for idx, row in terms_df.iterrows():
    mesh_id = row["id"]
    main_term = row["term"]
    
    # Try main term first
    emb = get_embedding(main_term, model)
    if emb is not None:
        term_embeddings[mesh_id] = {
            "vector": emb,
            "canonical_name": main_term,
            "normalized": normalize_term(main_term),
            "found_via": "main_term"
        }
        coverage_stats["found"] += 1
        continue
    
    # Try aliases if main term not found
    aliases = row["aliases"] if isinstance(row["aliases"], list) else []
    found = False
    for alias in aliases:
        emb = get_embedding(alias, model)
        if emb is not None:
            term_embeddings[mesh_id] = {
                "vector": emb,
                "canonical_name": alias,
                "normalized": normalize_term(alias),
                "found_via": "alias"
            }
            coverage_stats["found"] += 1
            coverage_stats["found_via_alias"] += 1
            found = True
            break
    
    if not found:
        coverage_stats["not_found"] += 1

print(f"      ✓ Found embeddings for {coverage_stats['found']:,}/{len(terms_df):,} terms")
print(f"        - Via main term: {coverage_stats['found'] - coverage_stats['found_via_alias']:,}")
print(f"        - Via alias: {coverage_stats['found_via_alias']:,}")
print(f"        - Not found: {coverage_stats['not_found']:,}")
print(f"        Coverage: {100 * coverage_stats['found'] / len(terms_df):.1f}%")

# Save embedding coverage report
emb_report = []
for mesh_id, emb_data in term_embeddings.items():
    emb_report.append({
        "mesh_id": mesh_id,
        "canonical_name": emb_data["canonical_name"],
        "normalized": emb_data["normalized"],
        "found_via": emb_data["found_via"]
    })
emb_df = pd.DataFrame(emb_report)
emb_df.to_csv(TERM_EMB_OUT, index=False)
print(f"      Saved embedding coverage to {TERM_EMB_OUT.name}")

# ---------- STEP 4: SEPARATE BY TYPE ----------
print("\n[4/7] Separating terms by type (disease vs drug)...")

# Detect type based on tree numbers
def detect_type(tree_nums):
    if pd.isna(tree_nums):
        return "unknown"
    tree_str = str(tree_nums)
    for tn in tree_str.split(","):
        if tn.strip().startswith("C"):
            return "disease"
    return "drug"

terms_df["type"] = terms_df["tree_nums"].apply(detect_type)

# Create a dictionary for fast type lookup
type_dict = dict(zip(terms_df["id"], terms_df["type"]))

# Split by type for separate clustering
disease_ids = [mid for mid in term_embeddings.keys() 
               if type_dict.get(mid, "unknown") == "disease"]
drug_ids = [mid for mid in term_embeddings.keys() 
            if type_dict.get(mid, "unknown") == "drug"]
unknown_ids = [mid for mid in term_embeddings.keys()
               if type_dict.get(mid, "unknown") == "unknown"]

print(f"      Diseases with embeddings: {len(disease_ids):,}")
print(f"      Drugs with embeddings: {len(drug_ids):,}")
if unknown_ids:
    print(f"      Unknown type (will cluster with drugs): {len(unknown_ids):,}")
    # Add unknowns to drugs for clustering
    drug_ids.extend(unknown_ids)

# ---------- STEP 5: CLUSTER SIMILAR TERMS ----------
print("\n[5/7] Clustering similar terms...")

def cluster_terms(term_ids, label_prefix):
    """
    Cluster terms based on cosine similarity of embeddings.
    Returns canonical mapping and cluster information.
    """
    if len(term_ids) < 2:
        print(f"      Skipping {label_prefix}: too few terms")
        return {}, []
    
    print(f"      Clustering {len(term_ids):,} {label_prefix}...")
    
    # Create embedding matrix
    vectors = np.array([term_embeddings[mid]["vector"] for mid in term_ids])
    
    # Compute pairwise similarities
    print(f"        Computing similarity matrix...")
    sim_matrix = cosine_similarity(vectors)
    
    # Convert to distance matrix
    distance_matrix = 1 - sim_matrix
    
    # Hierarchical clustering
    print(f"        Clustering with threshold {SIMILARITY_THRESHOLD}...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD,
        metric="precomputed",
        linkage="average"
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    # Group into clusters
    clusters = {}
    for idx, label in enumerate(labels):
        mesh_id = term_ids[idx]
        clusters.setdefault(label, []).append(mesh_id)
    
    # Create canonical mapping
    canonical_mapping = {}
    cluster_info = []
    
    for cluster_id, members in clusters.items():
        # Choose canonical term (shortest name as heuristic)
        canonical = min(members, 
                       key=lambda x: len(term_embeddings[x]["canonical_name"]))
        
        # Map all members to canonical
        for member in members:
            canonical_mapping[member] = canonical
        
        # Record cluster info
        if len(members) > 1:  # Only record clusters with multiple members
            cluster_info.append({
                "cluster_id": f"{label_prefix}_{cluster_id}",
                "canonical_id": canonical,
                "canonical_name": term_embeddings[canonical]["canonical_name"],
                "member_count": len(members),
                "member_ids": ";".join(members),
                "member_names": ";".join([term_embeddings[m]["canonical_name"] for m in members])
            })
    
    print(f"        ✓ Created {len(clusters):,} clusters")
    print(f"          - {len([c for c in clusters.values() if len(c) > 1]):,} clusters with synonyms")
    print(f"          - {len([c for c in clusters.values() if len(c) == 1]):,} singleton clusters")
    
    return canonical_mapping, cluster_info

# Cluster diseases and drugs separately
disease_mapping, disease_clusters = cluster_terms(disease_ids, "disease")
drug_mapping, drug_clusters = cluster_terms(drug_ids, "drug")

# Combine mappings
all_canonical_mapping = {**disease_mapping, **drug_mapping}
all_clusters = disease_clusters + drug_clusters

print(f"\n      Total canonical mappings: {len(all_canonical_mapping):,}")
print(f"      Total synonym clusters: {len(all_clusters):,}")

# Save cluster information
if all_clusters:
    cluster_df = pd.DataFrame(all_clusters).sort_values("member_count", ascending=False)
    cluster_df.to_csv(CLUSTERS_OUT, index=False)
    print(f"      Saved clusters to {CLUSTERS_OUT.name}")
    
    # Show top clusters
    print(f"\n      Top 5 largest synonym clusters:")
    for idx, row in cluster_df.head(5).iterrows():
        print(f"        - {row['canonical_name']} ({row['member_count']} synonyms)")

# Save canonical mapping
mapping_rows = []
for orig_id, canon_id in all_canonical_mapping.items():
    mapping_rows.append({
        "original_id": orig_id,
        "canonical_id": canon_id,
        "original_name": term_embeddings[orig_id]["canonical_name"],
        "canonical_name": term_embeddings[canon_id]["canonical_name"],
        "is_synonym": orig_id != canon_id
    })

mapping_df = pd.DataFrame(mapping_rows)
mapping_df.to_csv(MAPPING_OUT, index=False)
print(f"      Saved canonical mapping to {MAPPING_OUT.name}")

synonyms_count = len(mapping_df[mapping_df["is_synonym"] == True])
print(f"      Terms mapped to different canonical form: {synonyms_count:,}")

# ---------- STEP 6: APPLY TO AGGREGATED CO-MENTIONS ----------
print("\n[6/7] Applying canonical mapping to aggregated co-mentions...")

agg_df = pd.read_csv(AGG_FILE)
print(f"      Loaded {len(agg_df):,} drug-disease pairs")

# Apply canonical mapping (keep original if not in mapping)
agg_df["drug_ui_canonical"] = agg_df["drug_ui"].map(all_canonical_mapping).fillna(agg_df["drug_ui"])
agg_df["disease_ui_canonical"] = agg_df["disease_ui"].map(all_canonical_mapping).fillna(agg_df["disease_ui"])

# Get canonical names
agg_df["drug_name_canonical"] = agg_df["drug_ui_canonical"].apply(
    lambda x: term_embeddings[x]["canonical_name"] if x in term_embeddings else ""
)
agg_df["disease_name_canonical"] = agg_df["disease_ui_canonical"].apply(
    lambda x: term_embeddings[x]["canonical_name"] if x in term_embeddings else ""
)

# Re-aggregate by canonical IDs
print("      Re-aggregating by canonical terms...")
canonical_agg = agg_df.groupby(["drug_ui_canonical", "disease_ui_canonical"]).agg({
    "paper_count": "sum",
    "first_paper": "min",
    "last_paper": "max",
    "drug_name_canonical": "first",
    "disease_name_canonical": "first"
}).reset_index()

# Rename columns for consistency
canonical_agg = canonical_agg.rename(columns={
    "drug_ui_canonical": "drug_ui",
    "disease_ui_canonical": "disease_ui",
    "drug_name_canonical": "drug_name",
    "disease_name_canonical": "disease_name"
})

# Sort by paper count
canonical_agg = canonical_agg.sort_values("paper_count", ascending=False)

print(f"      ✓ Original pairs: {len(agg_df):,}")
print(f"      ✓ After merging synonyms: {len(canonical_agg):,}")
print(f"      ✓ Reduction: {len(agg_df) - len(canonical_agg):,} pairs ({100 * (len(agg_df) - len(canonical_agg)) / len(agg_df):.1f}%)")

canonical_agg.to_csv(CANONICAL_AGG_OUT, index=False)
print(f"      Saved to {CANONICAL_AGG_OUT.name}")

# ---------- STEP 7: SUMMARY ----------
print("\n[7/7] Summary")
print("=" * 60)

runtime = time.time() - start_time
print(f"Runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")
print(f"\nOutput files created in {DATA_DIR}:")
print(f"  1. {TERM_EMB_OUT.name} - embedding coverage report")
print(f"  2. {CLUSTERS_OUT.name} - synonym clusters")
print(f"  3. {MAPPING_OUT.name} - canonical term mappings")
print(f"  4. {CANONICAL_AGG_OUT.name} - normalized co-mentions")

print(f"\nTop 10 drug-disease pairs (after normalization):")
print(canonical_agg.head(10)[["drug_name", "disease_name", "paper_count"]].to_string(index=False))

print("\n" + "=" * 60)
print("✓ Term normalization complete!")
print("=" * 60)
print("\nNext steps:")
print("  - Review synonym_clusters.csv to validate groupings")
print("  - Use aggregated_canonical.csv for graph construction")
print("  - Adjust SIMILARITY_THRESHOLD if needed and re-run")