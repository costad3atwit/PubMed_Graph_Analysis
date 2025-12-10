"""
Optimized Frequency Analysis using Aho-Corasick for fast term matching.
This version is 100-1000x faster than regex-based approach for large term sets.

Key optimization: Uses Aho-Corasick automaton to find all terms in a single pass
through the text, rather than running separate regex for each term.
"""

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from pathlib import Path
import string
import re
import time
import ahocorasick

start_time = time.time()

print("=" * 70)
print("OPTIMIZED FREQUENCY ANALYSIS (Aho-Corasick)")
print("=" * 70)

# === File paths ===
DATA_DIR = Path("Data")
FIGURES_DIR = Path("Figures")
FIGURES_DIR.mkdir(exist_ok=True)

TERMS_PATH = DATA_DIR / "terms.csv"
PAPERS_PATH = DATA_DIR / "papers.csv"
CO_MENTIONS_PATH = DATA_DIR / "co-mentions.csv"
AGGREGATED_PATH = DATA_DIR / "aggregated.csv"

print(f"\nReading files from: {DATA_DIR}")

# === Helper function for normalization ===
def normalize(text: str) -> str:
    """Normalize text for matching (same as detect_co_mentions.py)"""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()

# === Load data ===
print("\n[1/6] Loading data files...")
try:
    terms_df = pd.read_csv(TERMS_PATH)
    papers_df = pd.read_csv(PAPERS_PATH)
    print(f"       Loaded {len(terms_df):,} terms")
    print(f"       Loaded {len(papers_df):,} papers")
except FileNotFoundError as e:
    print(f"       File not found: {e}")
    exit(1)

# === Detect title column ===
title_col = None
for c in papers_df.columns:
    if "title" in c.lower():
        title_col = c
        break

if title_col is None:
    raise KeyError("No column containing 'title' was found in papers.csv")

print(f"       Using column: '{title_col}'")

# === Build Aho-Corasick automaton ===
print("\n[2/6] Building Aho-Corasick automaton for fast matching...")

# Extract all term variants (main term + aliases)
all_terms = []
term_to_id = {}  # Map normalized term to MeSH ID

for _, row in terms_df.iterrows():
    mesh_id = str(row.get("id", "")).strip()
    if not mesh_id or mesh_id == "nan":
        continue
    
    # Add main term
    main_term = str(row.get("term", "")).strip()
    if main_term and main_term != "nan":
        normalized = normalize(main_term)
        if normalized:
            all_terms.append((normalized, mesh_id, main_term))
            term_to_id[normalized] = (mesh_id, main_term)
    
    # Add aliases
    entries_field = row.get("entries", "") or ""
    if pd.notna(entries_field):
        entries = re.split(r"[\n;]", str(entries_field))
        for e in entries:
            e = e.strip()
            if not e:
                continue
            # Take first part before pipe
            alias = e.split("|", 1)[0].strip()
            if alias:
                normalized = normalize(alias)
                if normalized and normalized not in term_to_id:
                    all_terms.append((normalized, mesh_id, alias))
                    term_to_id[normalized] = (mesh_id, main_term)

print(f"       Extracted {len(all_terms):,} term variants")
print(f"       Unique normalized terms: {len(term_to_id):,}")

# Build automaton
A = ahocorasick.Automaton()
for normalized, mesh_id, display_name in all_terms:
    if normalized not in A:
        A.add_word(normalized, (mesh_id, display_name))

A.make_automaton()
print(f"       Built Aho-Corasick automaton with {len(A):,} patterns")

# === Count term frequencies in titles ===
print("\n[3/6] Scanning paper titles for term frequencies...")

term_counts = Counter()
mesh_id_to_name = {}  # Track the display name for each MeSH ID

# Process each paper title individually to count occurrences
for idx, title in enumerate(papers_df[title_col].dropna().astype(str), 1):
    normalized_title = normalize(title)
    
    # Find all terms in this title
    found_in_title = set()  # Track which MeSH IDs appear (avoid double-counting)
    
    for end_idx, (mesh_id, display_name) in A.iter(normalized_title):
        found_in_title.add(mesh_id)
        mesh_id_to_name[mesh_id] = display_name
    
    # Increment count for each unique term found
    for mesh_id in found_in_title:
        term_counts[mesh_id] += 1
    
    if idx % 10000 == 0:
        print(f"      Processed {idx:,} / {len(papers_df):,} papers...")

print(f"       Found {len(term_counts):,} unique terms in titles")

# === Convert to DataFrame with display names ===
print("\n[4/6] Preparing frequency data...")

freq_data = []
for mesh_id, count in term_counts.items():
    display_name = mesh_id_to_name.get(mesh_id, mesh_id)
    freq_data.append({
        "mesh_id": mesh_id,
        "term": display_name,
        "count": count
    })

freq_df = pd.DataFrame(freq_data)
freq_df = freq_df.sort_values("count", ascending=False)

print(f"\nTop 20 most frequent terms in titles:")
print("─" * 70)
for idx, row in freq_df.head(20).iterrows():
    print(f"  {row['term'][:50]:50s} {row['count']:6,} papers")
print("─" * 70)

# Save frequency data
freq_output = DATA_DIR / "term_frequencies.csv"
freq_df.to_csv(freq_output, index=False)
print(f"\n       Saved frequency data to {freq_output.name}")

# === Generate visualizations ===
print("\n[5/6] Generating visualizations...")

# Visualization 1: Top 20 terms bar chart
if not freq_df.empty and len(freq_df) >= 20:
    plt.figure(figsize=(12, 8))
    top_20 = freq_df.head(20)
    plt.barh(top_20["term"][::-1], top_20["count"][::-1], color='steelblue', edgecolor='black')
    plt.xlabel("Number of Papers", fontsize=12)
    plt.ylabel("Term", fontsize=11)
    plt.title("Top 20 MeSH Terms in Paper Titles", fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_20_terms_in_titles.png", dpi=300, bbox_inches='tight')
    print(f"       Saved: top_20_terms_in_titles.png")
    plt.close()

# Visualization 2: Word cloud
if not freq_df.empty:
    # Create word cloud from term names and counts
    wc_data = {row['term']: row['count'] for _, row in freq_df.head(100).iterrows()}
    wc = WordCloud(width=1200, height=600, background_color="white",
                   colormap="viridis").generate_from_frequencies(wc_data)
    
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud: Top MeSH Terms in Paper Titles", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "term_wordcloud_titles.png", dpi=300, bbox_inches='tight')
    print(f"       Saved: term_wordcloud_titles.png")
    plt.close()

# Visualization 3: Frequency distribution
if not freq_df.empty:
    plt.figure(figsize=(12, 6))
    plt.hist(freq_df['count'], bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel("Number of Papers Mentioning Term", fontsize=12)
    plt.ylabel("Number of Terms", fontsize=12)
    plt.title("Distribution of Term Frequencies", fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "term_frequency_distribution.png", dpi=300, bbox_inches='tight')
    print(f"       Saved: term_frequency_distribution.png")
    plt.close()

# === Refined data analysis (co-mentions) ===
print("\n[6/6] Analyzing refined data (post-filtering co-mentions)...")

try:
    co_mentions_df = pd.read_csv(CO_MENTIONS_PATH)
    aggregated_df = pd.read_csv(AGGREGATED_PATH)
    print(f"       Loaded {len(co_mentions_df):,} co-mentions")
    print(f"       Loaded {len(aggregated_df):,} aggregated pairs")
    
    # Drug and disease frequency analysis
    drug_counts = Counter(co_mentions_df['drug_name'].dropna())
    disease_counts = Counter(co_mentions_df['disease_name'].dropna())
    
    print(f"\n      Found {len(drug_counts):,} unique drugs")
    print(f"      Found {len(disease_counts):,} unique diseases")
    
    print(f"\n      Top 10 most mentioned drugs:")
    for drug, count in drug_counts.most_common(10):
        print(f"        {drug}: {count:,}")
    
    print(f"\n      Top 10 most mentioned diseases:")
    for disease, count in disease_counts.most_common(10):
        print(f"        {disease}: {count:,}")
    
    # Visualization: Top drugs
    if drug_counts:
        drug_freq_df = pd.DataFrame(drug_counts.most_common(20), columns=["drug", "mentions"])
        
        plt.figure(figsize=(12, 8))
        plt.barh(drug_freq_df["drug"][::-1], drug_freq_df["mentions"][::-1], 
                color='steelblue', edgecolor='black')
        plt.xlabel("Number of Co-Mentions", fontsize=12)
        plt.ylabel("Drug", fontsize=11)
        plt.title("Top 20 Drugs by Co-Mention Frequency\n(After Specificity Filtering)", 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "top_20_drugs_by_comentions.png", dpi=300, bbox_inches='tight')
        print(f"       Saved: top_20_drugs_by_comentions.png")
        plt.close()
    
    # Visualization: Top diseases
    if disease_counts:
        disease_freq_df = pd.DataFrame(disease_counts.most_common(20), columns=["disease", "mentions"])
        
        plt.figure(figsize=(12, 8))
        plt.barh(disease_freq_df["disease"][::-1], disease_freq_df["mentions"][::-1], 
                color='darkred', edgecolor='black')
        plt.xlabel("Number of Co-Mentions", fontsize=12)
        plt.ylabel("Disease", fontsize=11)
        plt.title("Top 20 Diseases by Co-Mention Frequency\n(After Specificity Filtering)", 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "top_20_diseases_by_comentions.png", dpi=300, bbox_inches='tight')
        print(f"       Saved: top_20_diseases_by_comentions.png")
        plt.close()
    
    # Word clouds for refined data
    if drug_counts:
        drug_wc = WordCloud(width=1200, height=600, background_color="white", 
                           colormap="Blues").generate_from_frequencies(drug_counts)
        plt.figure(figsize=(14, 7))
        plt.imshow(drug_wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud: Drugs (Refined Co-Mentions)", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "drugs_wordcloud.png", dpi=300, bbox_inches='tight')
        print(f"       Saved: drugs_wordcloud.png")
        plt.close()
    
    if disease_counts:
        disease_wc = WordCloud(width=1200, height=600, background_color="white",
                              colormap="Reds").generate_from_frequencies(disease_counts)
        plt.figure(figsize=(14, 7))
        plt.imshow(disease_wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud: Diseases (Refined Co-Mentions)", fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "diseases_wordcloud.png", dpi=300, bbox_inches='tight')
        print(f"       Saved: diseases_wordcloud.png")
        plt.close()
    
    # Top associations
    print(f"\n      Top 15 Drug-Disease Associations (by paper count):")
    print("      " + "─" * 66)
    top_associations = aggregated_df.nlargest(15, 'paper_count')
    
    for idx, row in top_associations.iterrows():
        drug = row['drug_name'][:25]
        disease = row['disease_name'][:25]
        print(f"      {drug:25s} ↔ {disease:25s} ({row['paper_count']:4,} papers)")
    
    # Co-mention location analysis
    print(f"\n      Co-Mention Location Distribution:")
    location_counts = co_mentions_df['where_found'].value_counts()
    for location, count in location_counts.items():
        pct = count/len(co_mentions_df)*100
        print(f"        {location:10s}: {count:6,} ({pct:5.1f}%)")
    
    # Visualization: Location pie chart
    plt.figure(figsize=(10, 10))
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0.05, 0.05)
    plt.pie(location_counts.values, labels=location_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, explode=explode, textprops={'fontsize': 14})
    plt.title("Co-Mention Location Distribution\n(Title, Abstract, or Both)", 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comention_location_distribution.png", dpi=300, bbox_inches='tight')
    print(f"       Saved: comention_location_distribution.png")
    plt.close()
    
    # Distribution plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Paper count distribution
    ax1.hist(aggregated_df['paper_count'], bins=50, color='green', alpha=0.7, edgecolor='black')
    ax1.set_xlabel("Papers per Drug-Disease Pair", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Distribution of Evidence Strength", fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(alpha=0.3)
    
    # Time span distribution
    aggregated_df['time_span'] = aggregated_df['last_paper'] - aggregated_df['first_paper']
    ax2.hist(aggregated_df['time_span'].dropna(), bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax2.set_xlabel("Years Between First and Last Paper", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Research Timeline Spans", fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "evidence_and_timeline_distributions.png", dpi=300, bbox_inches='tight')
    print(f"       Saved: evidence_and_timeline_distributions.png")
    plt.close()
    
except FileNotFoundError as e:
    print(f"\n       Refined data files not found: {e}")
    print("      Run co-mention detection and aggregation scripts first")

# === Summary ===
runtime = time.time() - start_time

print("\n" + "=" * 70)
print(" FREQUENCY ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nRuntime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")
print(f"\nGenerated visualizations in: {FIGURES_DIR}")
print("\nOutput files:")
print(f"  - {freq_output.name} (term frequency data)")
print(f"  - top_20_terms_in_titles.png")
print(f"  - term_wordcloud_titles.png")
print(f"  - term_frequency_distribution.png")
print(f"  - top_20_drugs_by_comentions.png")
print(f"  - top_20_diseases_by_comentions.png")
print(f"  - drugs_wordcloud.png")
print(f"  - diseases_wordcloud.png")
print(f"  - comention_location_distribution.png")
print(f"  - evidence_and_timeline_distributions.png")
print("\n" + "=" * 70)