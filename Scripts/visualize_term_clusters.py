"""
Visualize synonym clusters from term normalization.
Creates multiple visualizations to understand term groupings and embeddings.

Inputs:
  - synonym_clusters.csv
  - canonical_mapping.csv
  - term_embeddings.csv
  - BioWordVec_PubMed_MIMICIII_d200.vec.bin (for t-SNE visualization)

Outputs:
  - Multiple matplotlib figures showing cluster analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ---------- CONFIG ----------
DATA_DIR = Path("C:/Users/sirda/Dropbox (Personal)/Documents/Fall '25/Data Mining/PubMed Project/Data")
PLOT_DIR = Path("C:/Users/sirda/Dropbox (Personal)/Documents/Fall '25/Data Mining/PubMed Project/Figures")
CLUSTERS_FILE = DATA_DIR / "synonym_clusters.csv"
MAPPING_FILE = DATA_DIR / "canonical_mapping.csv"
EMBEDDINGS_FILE = DATA_DIR / "BioWordVec_PubMed_MIMICIII_d200.vec.bin"
TERM_EMB_FILE = DATA_DIR / "term_embeddings.csv"

print("=" * 60)
print("VISUALIZING SYNONYM CLUSTERS")
print("=" * 60)

# ---------- LOAD DATA ----------
print("\n[1/6] Loading data files...")

try:
    clusters_df = pd.read_csv(CLUSTERS_FILE)
    mapping_df = pd.read_csv(MAPPING_FILE)
    term_emb_df = pd.read_csv(TERM_EMB_FILE)
    print(f"      ✓ Loaded {len(clusters_df):,} clusters")
    print(f"      ✓ Loaded {len(mapping_df):,} mappings")
    print(f"      ✓ Loaded {len(term_emb_df):,} term embeddings")
except FileNotFoundError as e:
    print(f"      ✗ Error: {e}")
    print("      Make sure to run normalize_terms_embeddings.py first!")
    exit(1)

# ---------- VIZ 1: CLUSTER SIZE DISTRIBUTION ----------
print("\n[2/6] Creating cluster size distribution plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Histogram of cluster sizes
ax1.hist(clusters_df['member_count'], bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Number of Terms per Cluster', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Cluster Sizes', fontsize=14, fontweight='bold')
ax1.axvline(clusters_df['member_count'].median(), color='red', 
            linestyle='--', label=f'Median: {clusters_df["member_count"].median():.0f}')
ax1.legend()
ax1.grid(alpha=0.3)

# Box plot by cluster type
clusters_df['type'] = clusters_df['cluster_id'].apply(lambda x: x.split('_')[0])
sns.boxplot(data=clusters_df, x='type', y='member_count', ax=ax2)
ax2.set_xlabel('Cluster Type', fontsize=12)
ax2.set_ylabel('Number of Terms per Cluster', fontsize=12)
ax2.set_title('Cluster Sizes by Type', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "cluster_size_distribution.png", dpi=300, bbox_inches='tight')
print(f"      ✓ Saved to {PLOT_DIR / 'cluster_size_distribution.png'}")
plt.close()

# ---------- VIZ 2: TOP CLUSTERS ----------
print("\n[3/6] Creating top clusters visualization...")

# Get top 20 largest clusters
top_clusters = clusters_df.nlargest(20, 'member_count')

fig, ax = plt.subplots(figsize=(14, 10))

# Create horizontal bar chart
y_pos = np.arange(len(top_clusters))
colors = ['#1f77b4' if 'disease' in cid else '#ff7f0e' 
          for cid in top_clusters['cluster_id']]

bars = ax.barh(y_pos, top_clusters['member_count'], color=colors, alpha=0.7, edgecolor='black')

# Customize
ax.set_yticks(y_pos)
ax.set_yticklabels([name[:50] + '...' if len(name) > 50 else name 
                     for name in top_clusters['canonical_name']], fontsize=10)
ax.set_xlabel('Number of Synonymous Terms', fontsize=12)
ax.set_title('Top 20 Largest Synonym Clusters', fontsize=14, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#1f77b4', label='Disease'),
                   Patch(facecolor='#ff7f0e', label='Drug')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(PLOT_DIR / "top_clusters.png", dpi=300, bbox_inches='tight')
print(f"      ✓ Saved to {PLOT_DIR / 'top_clusters.png'}")
plt.close()

# ---------- VIZ 3: CLUSTER EXAMPLES ----------
print("\n[4/6] Creating detailed cluster examples...")

# Select diverse examples
example_clusters = clusters_df.nlargest(10, 'member_count')

fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.flatten()

for idx, (_, cluster) in enumerate(example_clusters.iterrows()):
    if idx >= 10:
        break
    
    ax = axes[idx]
    
    # Parse member names
    members = cluster['member_names'].split(';')
    canonical = cluster['canonical_name']
    
    # Show up to 10 members
    display_members = members[:10]
    if len(members) > 10:
        display_members.append(f"... and {len(members) - 10} more")
    
    # Create text display
    y_positions = np.arange(len(display_members))
    
    # Color canonical differently
    colors = ['red' if m == canonical else 'black' for m in display_members]
    
    ax.barh(y_positions, [1] * len(display_members), alpha=0.0)  # Invisible bars for structure
    
    for y, member, color in zip(y_positions, display_members, colors):
        label = member[:60] + '...' if len(member) > 60 else member
        weight = 'bold' if color == 'red' else 'normal'
        ax.text(0.05, y, label, fontsize=9, va='center', weight=weight, color=color)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(display_members) - 0.5)
    ax.axis('off')
    
    cluster_type = cluster['cluster_id'].split('_')[0].title()
    ax.set_title(f"{cluster_type}: {canonical[:50]}\n({cluster['member_count']} synonyms)", 
                 fontsize=11, fontweight='bold', loc='left')

plt.suptitle('Example Synonym Clusters (Canonical in Red)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(PLOT_DIR / "cluster_examples.png", dpi=300, bbox_inches='tight')
print(f"      ✓ Saved to {PLOT_DIR / 'cluster_examples.png'}")
plt.close()

# ---------- VIZ 4: MAPPING STATISTICS ----------
print("\n[5/6] Creating mapping statistics visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4a: Synonym vs non-synonym counts
ax1 = axes[0, 0]
synonym_counts = mapping_df['is_synonym'].value_counts()
colors_pie = ['#2ecc71', '#e74c3c']
ax1.pie(synonym_counts, labels=['Same as canonical', 'Mapped to synonym'], 
        autopct='%1.1f%%', colors=colors_pie, startangle=90)
ax1.set_title('Terms Requiring Synonym Mapping', fontsize=12, fontweight='bold')

# 4b: Top canonical terms (most terms mapped to them)
ax2 = axes[0, 1]
top_canonical = mapping_df[mapping_df['is_synonym'] == True]['canonical_name'].value_counts().head(15)
ax2.barh(range(len(top_canonical)), top_canonical.values, color='steelblue', alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(top_canonical)))
ax2.set_yticklabels([name[:40] + '...' if len(name) > 40 else name for name in top_canonical.index], fontsize=9)
ax2.set_xlabel('Number of Terms Mapped to This Canonical', fontsize=10)
ax2.set_title('Most Common Canonical Terms', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

# 4c: Embedding coverage by type
ax3 = axes[1, 0]
coverage_by_type = term_emb_df.groupby('found_via').size()
ax3.bar(coverage_by_type.index, coverage_by_type.values, color=['#3498db', '#e67e22'], 
        alpha=0.7, edgecolor='black')
ax3.set_ylabel('Number of Terms', fontsize=10)
ax3.set_title('Embedding Coverage Strategy', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for i, v in enumerate(coverage_by_type.values):
    ax3.text(i, v + 50, str(v), ha='center', fontweight='bold')

# 4d: Summary statistics table
ax4 = axes[1, 1]
ax4.axis('off')

stats = [
    ['Metric', 'Value'],
    ['─' * 40, '─' * 15],
    ['Total terms', f"{len(mapping_df):,}"],
    ['Terms with synonyms', f"{len(clusters_df):,}"],
    ['Total clusters', f"{clusters_df['member_count'].sum():,}"],
    ['Avg cluster size', f"{clusters_df['member_count'].mean():.1f}"],
    ['Largest cluster', f"{clusters_df['member_count'].max():,}"],
    ['Disease clusters', f"{len(clusters_df[clusters_df['cluster_id'].str.contains('disease')]):,}"],
    ['Drug clusters', f"{len(clusters_df[clusters_df['cluster_id'].str.contains('drug')]):,}"],
    ['Embedding coverage', f"{100 * len(term_emb_df) / len(mapping_df):.1f}%"],
]

table = ax4.table(cellText=stats, cellLoc='left', loc='center',
                  colWidths=[0.7, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(2, len(stats)):
    for j in range(2):
        table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

plt.suptitle('Synonym Mapping Statistics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOT_DIR / "mapping_statistics.png", dpi=300, bbox_inches='tight')
print(f"      ✓ Saved to {PLOT_DIR / 'mapping_statistics.png'}")
plt.close()

# ---------- VIZ 5: EMBEDDING SPACE (t-SNE) ----------
print("\n[6/6] Creating embedding space visualization (this may take a minute)...")

try:
    # Load embeddings model
    print("      Loading BioWordVec for t-SNE visualization...")
    model = KeyedVectors.load_word2vec_format(str(EMBEDDINGS_FILE), binary=True)
    
    # Sample terms for visualization (too many will be cluttered)
    n_sample = min(500, len(term_emb_df))
    sample_terms = term_emb_df.sample(n_sample, random_state=42)
    
    # Get vectors
    vectors = []
    labels = []
    types = []
    
    for _, row in sample_terms.iterrows():
        normalized = row['normalized']
        if normalized in model:
            vectors.append(model[normalized])
            labels.append(row['canonical_name'][:30])
            
            # Determine type from cluster_id if available
            matching_cluster = clusters_df[clusters_df['canonical_id'] == row['mesh_id']]
            if len(matching_cluster) > 0:
                types.append(matching_cluster.iloc[0]['cluster_id'].split('_')[0])
            else:
                types.append('unknown')
    
    if len(vectors) > 50:  # Only create if we have enough points
        X = np.array(vectors)
        
        # Apply t-SNE
        print(f"      Running t-SNE on {len(vectors)} term vectors...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)//2))
        X_embedded = tsne.fit_transform(X)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: Colored by type
        colors_map = {'disease': '#e74c3c', 'drug': '#3498db', 'unknown': '#95a5a6'}
        for type_name in set(types):
            mask = np.array(types) == type_name
            ax1.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                       c=colors_map.get(type_name, '#95a5a6'),
                       label=type_name.title(), alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        ax1.set_title('Term Embeddings (t-SNE) - Colored by Type', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(alpha=0.3)
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        
        # Plot 2: Density
        from scipy.stats import gaussian_kde
        xy = X_embedded.T
        z = gaussian_kde(xy)(xy)
        scatter = ax2.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                             c=z, s=50, cmap='viridis', alpha=0.6, 
                             edgecolors='black', linewidth=0.5)
        ax2.set_title('Term Embeddings (t-SNE) - Density', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax2, label='Density')
        ax2.grid(alpha=0.3)
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "embedding_space_tsne.png", dpi=300, bbox_inches='tight')
        print(f"      ✓ Saved to {PLOT_DIR / 'embedding_space_tsne.png'}")
        plt.close()
        
        # Create an annotated version with top clusters
        fig, ax = plt.subplots(figsize=(14, 10))
        
        for type_name in set(types):
            mask = np.array(types) == type_name
            ax.scatter(X_embedded[mask, 0], X_embedded[mask, 1], 
                      c=colors_map.get(type_name, '#95a5a6'),
                      label=type_name.title(), alpha=0.4, s=30, edgecolors='black', linewidth=0.3)
        
        # Annotate a few interesting points (from largest clusters)
        top_cluster_terms = clusters_df.nlargest(10, 'member_count')['canonical_name'].values
        for i, label in enumerate(labels):
            if any(term in label for term in top_cluster_terms):
                ax.annotate(label, (X_embedded[i, 0], X_embedded[i, 1]), 
                          fontsize=8, alpha=0.7, 
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        ax.set_title('Term Embeddings with Top Cluster Annotations', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "embedding_space_annotated.png", dpi=300, bbox_inches='tight')
        print(f"      ✓ Saved to {PLOT_DIR / 'embedding_space_annotated.png'}")
        plt.close()
    else:
        print(f"      ⚠ Skipping t-SNE: not enough vectors ({len(vectors)})")

except Exception as e:
    print(f"      ⚠ Could not create t-SNE visualization: {e}")
    print("      This is optional - other visualizations are complete")

# ---------- SUMMARY ----------
print("\n" + "=" * 60)
print("✓ Visualization complete!")
print("=" * 60)
print(f"\nAll plots saved to: {PLOT_DIR}")
print("\nGenerated visualizations:")
print("  1. cluster_size_distribution.png - How cluster sizes are distributed")
print("  2. top_clusters.png - The 20 largest synonym groups")
print("  3. cluster_examples.png - Detailed view of top 10 clusters")
print("  4. mapping_statistics.png - Overall statistics dashboard")
print("  5. embedding_space_tsne.png - 2D projection of embedding space")
print("  6. embedding_space_annotated.png - Annotated with top clusters")
print("\n" + "=" * 60)