"""
Network Analysis for Drug-Disease Association Graph
Analyzes the bipartite network structure to extract insights about:
- Disease comorbidity implications (diseases sharing drugs)
- Drug polypharmacology (drugs treating multiple diseases)
- Network centrality and hub identification
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time

start_time = time.time()

print("=== Network Analysis Script Started ===")

# ---------- CONFIG ----------
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "Figures"
LOGS_DIR = PROJECT_ROOT / "Logs"
GRAPHS_DIR = PROJECT_ROOT / "Graphs"

CO_MENTIONS_PATH = DATA_DIR / "co-mentions.csv"
AGGREGATED_PATH = DATA_DIR / "aggregated.csv"

print(f"Reading files from: {DATA_DIR}")

# === Load data ===
try:
    co_mentions_df = pd.read_csv(CO_MENTIONS_PATH)
    aggregated_df = pd.read_csv(AGGREGATED_PATH)
    print(f"Loaded {len(co_mentions_df)} co-mentions and {len(aggregated_df)} aggregated pairs")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    print("Run the co-mention detection and aggregation scripts first.")
    exit(1)

# === Build bipartite network structure ===
print("\n=== Building Network Structure ===")

# Drug -> set of diseases
drug_to_diseases = defaultdict(set)
# Disease -> set of drugs
disease_to_drugs = defaultdict(set)
# Track names for readability
drug_names = {}
disease_names = {}

for _, row in aggregated_df.iterrows():
    drug_ui = row['drug_ui']
    disease_ui = row['disease_ui']
    drug_name = row['drug_name']
    disease_name = row['disease_name']
    
    drug_to_diseases[drug_ui].add(disease_ui)
    disease_to_drugs[disease_ui].add(drug_ui)
    drug_names[drug_ui] = drug_name
    disease_names[disease_ui] = disease_name

print(f"Network structure: {len(drug_to_diseases)} drugs, {len(disease_to_drugs)} diseases")

# === Drug Polypharmacology Analysis ===
print("\n=== Drug Polypharmacology Analysis ===")

# Calculate degree for each drug (number of diseases it's associated with)
drug_degrees = [(drug_ui, len(diseases), drug_names[drug_ui]) 
                for drug_ui, diseases in drug_to_diseases.items()]
drug_degrees.sort(key=lambda x: x[1], reverse=True)

print("Top 20 most versatile drugs (treat most diseases):")
for i, (drug_ui, degree, drug_name) in enumerate(drug_degrees[:20], 1):
    print(f"{i:2d}. {drug_name}: {degree} diseases")

# Plot top drugs by degree
top_drugs_df = pd.DataFrame(drug_degrees[:20], columns=['drug_ui', 'degree', 'drug_name'])

plt.figure(figsize=(12, 8))
plt.barh(top_drugs_df['drug_name'][::-1], top_drugs_df['degree'][::-1], color='steelblue')
plt.xlabel("Number of Associated Diseases", fontsize=12)
plt.ylabel("Drug", fontsize=12)
plt.title("Top 20 Drugs by Polypharmacology\n(Number of Disease Associations)", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "network_top_drugs_polypharmacology.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'network_top_drugs_polypharmacology.png'}")

# === Disease Comorbidity Implications ===
print("\n=== Disease Comorbidity Implications Analysis ===")
print("Computing disease-disease similarity based on shared drugs...")

# Get top N diseases by degree for manageable visualization
disease_degrees = [(disease_ui, len(drugs), disease_names[disease_ui]) 
                   for disease_ui, drugs in disease_to_drugs.items()]
disease_degrees.sort(key=lambda x: x[1], reverse=True)

# Use top 15 diseases for the heatmap
TOP_N = 15
top_diseases = [d[0] for d in disease_degrees[:TOP_N]]
top_disease_names = [d[2] for d in disease_degrees[:TOP_N]]

print(f"Analyzing comorbidity patterns for top {TOP_N} diseases...")

# Build similarity matrix: Jaccard similarity based on shared drugs
similarity_matrix = np.zeros((TOP_N, TOP_N))

for i, disease_i in enumerate(top_diseases):
    drugs_i = disease_to_drugs[disease_i]
    for j, disease_j in enumerate(top_diseases):
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            drugs_j = disease_to_drugs[disease_j]
            # Jaccard similarity: |intersection| / |union|
            intersection = len(drugs_i & drugs_j)
            union = len(drugs_i | drugs_j)
            similarity_matrix[i, j] = intersection / union if union > 0 else 0

# Create heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(similarity_matrix, 
            xticklabels=top_disease_names,
            yticklabels=top_disease_names,
            cmap='YlOrRd',
            cbar_kws={'label': 'Jaccard Similarity (Shared Drugs)'},
            square=True,
            linewidths=0.5,
            linecolor='gray')
plt.title(f"Disease Comorbidity Implications Heatmap\n(Top {TOP_N} Diseases by Drug Associations)", 
          fontsize=14, pad=20)
plt.xlabel("Disease", fontsize=11)
plt.ylabel("Disease", fontsize=11)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "network_disease_comorbidity_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'network_disease_comorbidity_heatmap.png'}")

# Print some interesting high-similarity pairs
print("\nHighest disease-disease similarities (excluding diagonal):")
similarity_pairs = []
for i in range(TOP_N):
    for j in range(i+1, TOP_N):
        if similarity_matrix[i, j] > 0.1:  # Threshold for meaningful similarity
            similarity_pairs.append((
                top_disease_names[i],
                top_disease_names[j],
                similarity_matrix[i, j]
            ))

similarity_pairs.sort(key=lambda x: x[2], reverse=True)
for disease_a, disease_b, sim in similarity_pairs[:15]:
    print(f"  {disease_a} ↔ {disease_b}: {sim:.3f}")

# === Network Degree Distributions ===
print("\n=== Degree Distribution Analysis ===")

drug_degree_values = [degree for _, degree, _ in drug_degrees]
disease_degree_values = [degree for _, degree, _ in disease_degrees]

print(f"Drug degree stats: mean={np.mean(drug_degree_values):.1f}, "
      f"median={np.median(drug_degree_values):.1f}, "
      f"max={np.max(drug_degree_values)}")
print(f"Disease degree stats: mean={np.mean(disease_degree_values):.1f}, "
      f"median={np.median(disease_degree_values):.1f}, "
      f"max={np.max(disease_degree_values)}")

# Plot degree distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Drug degree distribution
ax1.hist(drug_degree_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel("Number of Disease Associations", fontsize=11)
ax1.set_ylabel("Number of Drugs", fontsize=11)
ax1.set_title("Drug Degree Distribution", fontsize=12)
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3)

# Disease degree distribution
ax2.hist(disease_degree_values, bins=50, color='darkred', alpha=0.7, edgecolor='black')
ax2.set_xlabel("Number of Drug Associations", fontsize=11)
ax2.set_ylabel("Number of Diseases", fontsize=11)
ax2.set_title("Disease Degree Distribution", fontsize=12)
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "network_degree_distributions.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'network_degree_distributions.png'}")

# === Alzheimer's Disease Focus ===
print("\n=== Alzheimer's Disease Specific Analysis ===")

# Find Alzheimer's disease in the data
alzheimers_candidates = [
    (disease_ui, disease_name) 
    for disease_ui, disease_name in disease_names.items() 
    if 'alzheimer' in disease_name.lower()
]

if alzheimers_candidates:
    print(f"Found {len(alzheimers_candidates)} Alzheimer's-related disease terms:")
    for disease_ui, disease_name in alzheimers_candidates:
        num_drugs = len(disease_to_drugs[disease_ui])
        print(f"  {disease_name}: {num_drugs} drug associations")
    
    # Focus on the main Alzheimer's Disease term (usually the one with most connections)
    main_alzheimers = max(alzheimers_candidates, key=lambda x: len(disease_to_drugs[x[0]]))
    alzheimers_ui, alzheimers_name = main_alzheimers
    alzheimers_drugs = disease_to_drugs[alzheimers_ui]
    
    print(f"\nFocusing on: {alzheimers_name}")
    print(f"Number of associated drugs: {len(alzheimers_drugs)}")
    
    # Get drug details for Alzheimer's
    alzheimers_drug_details = []
    for drug_ui in alzheimers_drugs:
        drug_name = drug_names[drug_ui]
        # Count papers for this specific drug-disease pair
        papers = aggregated_df[
            (aggregated_df['drug_ui'] == drug_ui) & 
            (aggregated_df['disease_ui'] == alzheimers_ui)
        ]
        if not papers.empty:
            paper_count = papers.iloc[0]['paper_count']
            first_year = papers.iloc[0]['first_paper']
            last_year = papers.iloc[0]['last_paper']
            alzheimers_drug_details.append((drug_name, paper_count, first_year, last_year))
    
    alzheimers_drug_details.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 15 drugs associated with {alzheimers_name}:")
    for i, (drug_name, papers, first, last) in enumerate(alzheimers_drug_details[:15], 1):
        print(f"{i:2d}. {drug_name}: {papers} papers ({first}-{last})")
    
    # Plot Alzheimer's drug associations
    top_15_alzheimers = alzheimers_drug_details[:15]
    
    plt.figure(figsize=(12, 8))
    drug_names_plot = [d[0] for d in top_15_alzheimers]
    paper_counts_plot = [d[1] for d in top_15_alzheimers]
    
    plt.barh(drug_names_plot[::-1], paper_counts_plot[::-1], color='mediumseagreen')
    plt.xlabel("Number of Papers", fontsize=12)
    plt.ylabel("Drug", fontsize=12)
    plt.title(f"Top 15 Drugs Associated with {alzheimers_name}\n(by number of co-mentioning papers)", 
              fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "network_alzheimers_top_drugs.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'network_alzheimers_top_drugs.png'}")
else:
    print("No Alzheimer's disease terms found in the dataset.")

# === Summary Statistics ===
print("\n=== Network Summary Statistics ===")
print(f"Total nodes: {len(drug_to_diseases) + len(disease_to_drugs)}")
print(f"  - Drugs: {len(drug_to_diseases)}")
print(f"  - Diseases: {len(disease_to_drugs)}")
print(f"Total edges: {len(aggregated_df)}")
print(f"Average drug degree: {np.mean(drug_degree_values):.2f}")
print(f"Average disease degree: {np.mean(disease_degree_values):.2f}")
print(f"Network density: {len(aggregated_df) / (len(drug_to_diseases) * len(disease_to_drugs)) * 100:.3f}%")

# === Wrap up ===
runtime = time.time() - start_time
print(f"\n=== Network analysis complete. Runtime: {runtime:.2f} seconds ===")