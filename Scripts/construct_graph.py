"""
Construct drug-disease association graph from normalized co-mentions.
Creates a bipartite network where drugs and diseases are nodes,
and edges represent co-mention relationships with paper counts.

Inputs:
  - aggregated_canonical.csv (normalized co-mentions)
  - canonical_mapping.csv (for term metadata)
  - terms.csv (for additional node attributes)

Outputs:
  - graph.graphml (full network in GraphML format)
  - graph_filtered.graphml (filtered by minimum paper count)
  - graph_stats.txt (network statistics)
  - Interactive HTML visualizations
"""

import pandas as pd
import networkx as nx

import json
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import pyvis for interactive visualization
try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    print("Note: pyvis not installed. Install with 'pip install pyvis' for interactive visualizations")

# ---------- CONFIG ----------
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "Data"
FIGURES_DIR = PROJECT_ROOT / "Figures"
LOGS_DIR = PROJECT_ROOT / "Logs"
GRAPHS_DIR = PROJECT_ROOT / "Graphs"

AGG_FILE = DATA_DIR / "aggregated_canonical.csv"
MAPPING_FILE = DATA_DIR / "canonical_mapping.csv"
TERMS_FILE = DATA_DIR / "terms.csv"

FULL_GRAPH = GRAPHS_DIR / "drug_disease_graph.graphml"
FILTERED_GRAPH = GRAPHS_DIR / "drug_disease_graph_filtered.graphml"
STATS_FILE = GRAPHS_DIR / "graph_statistics.txt"
VIZ_HTML = GRAPHS_DIR / "interactive_graph.html"
VIZ_FILTERED_HTML = GRAPHS_DIR / "interactive_graph_filtered.html"

# Filtering parameters
MIN_PAPERS = 5  # Minimum number of papers for an edge to be included in filtered graph
TOP_N_EDGES = 1000  # For interactive visualization (to avoid overwhelming browser)

print("=" * 70)
print("CONSTRUCTING DRUG-DISEASE ASSOCIATION GRAPH")
print("=" * 70)

# ---------- STEP 1: LOAD DATA ----------
print("\n[1/7] Loading data files...")

try:
    agg_df = pd.read_csv(AGG_FILE)
    mapping_df = pd.read_csv(MAPPING_FILE)
    terms_df = pd.read_csv(TERMS_FILE)
    
    print(f"      ✓ Loaded {len(agg_df):,} drug-disease associations")
    print(f"      ✓ Loaded {len(mapping_df):,} term mappings")
    print(f"      ✓ Loaded {len(terms_df):,} MeSH terms")
except FileNotFoundError as e:
    print(f"      ✗ Error: {e}")
    print("      Make sure to run normalize_terms_embeddings.py first!")
    exit(1)

# ---------- STEP 2: PREPARE NODE ATTRIBUTES ----------
print("\n[2/7] Preparing node attributes...")

# Create node attribute dictionaries
def detect_type(tree_nums):
    if pd.isna(tree_nums):
        return "unknown"
    for tn in str(tree_nums).split(","):
        if tn.strip().startswith("C"):
            return "disease"
    return "drug"

terms_df["type"] = terms_df["tree_nums"].apply(detect_type)

# Build node info dictionary
node_info = {}
for _, row in terms_df.iterrows():
    node_info[row["id"]] = {
        "name": row["term"],
        "type": row["type"],
        "tree_nums": row.get("tree_nums", ""),
        "mesh_id": row["id"]
    }

print(f"      ✓ Prepared attributes for {len(node_info):,} nodes")

# ---------- STEP 3: BUILD FULL GRAPH ----------
print("\n[3/7] Building full bipartite graph...")

G = nx.Graph()

# Add nodes with attributes
drugs = set()
diseases = set()

for mesh_id, info in node_info.items():
    G.add_node(
        mesh_id,
        name=info["name"],
        type=info["type"],
        tree_nums=info["tree_nums"],
        label=info["name"]  # For visualization
    )
    if info["type"] == "disease":
        diseases.add(mesh_id)
    else:
        drugs.add(mesh_id)

print(f"      ✓ Added {G.number_of_nodes():,} nodes")
print(f"        - Diseases: {len(diseases):,}")
print(f"        - Drugs: {len(drugs):,}")

# Add edges from co-mentions
edges_added = 0
edges_skipped = 0

for _, row in agg_df.iterrows():
    drug_id = row["drug_ui"]
    disease_id = row["disease_ui"]
    
    # Only add edge if both nodes exist in graph
    if drug_id in G and disease_id in G:
        G.add_edge(
            drug_id,
            disease_id,
            weight=int(row["paper_count"]),
            paper_count=int(row["paper_count"]),
            first_year=int(row["first_paper"]) if pd.notna(row["first_paper"]) else None,
            last_year=int(row["last_paper"]) if pd.notna(row["last_paper"]) else None,
            drug_name=row.get("drug_name", ""),
            disease_name=row.get("disease_name", "")
        )
        edges_added += 1
    else:
        edges_skipped += 1

print(f"      ✓ Added {edges_added:,} edges")
if edges_skipped > 0:
    print(f"      ⚠ Skipped {edges_skipped:,} edges (nodes not in graph)")

# ---------- STEP 4: COMPUTE GRAPH STATISTICS ----------
print("\n[4/7] Computing graph statistics...")

stats = {}

# Basic stats
stats["nodes"] = G.number_of_nodes()
stats["edges"] = G.number_of_edges()
stats["density"] = nx.density(G)

# Connected components
stats["connected_components"] = nx.number_connected_components(G)
largest_cc = max(nx.connected_components(G), key=len)
stats["largest_component_size"] = len(largest_cc)
stats["largest_component_pct"] = 100 * len(largest_cc) / G.number_of_nodes()

# Node type counts
node_types = Counter([G.nodes[n]["type"] for n in G.nodes()])
stats["disease_nodes"] = node_types["disease"]
stats["drug_nodes"] = node_types["drug"]

# Degree statistics
degrees = dict(G.degree())
stats["avg_degree"] = sum(degrees.values()) / len(degrees)
stats["max_degree"] = max(degrees.values())
stats["min_degree"] = min(degrees.values())

# Edge weight statistics
edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
stats["avg_papers_per_edge"] = sum(edge_weights) / len(edge_weights)
stats["max_papers_per_edge"] = max(edge_weights)
stats["total_papers"] = sum(edge_weights)

# Top nodes by degree
top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
stats["top_10_nodes"] = [(node, deg, G.nodes[node]["name"]) for node, deg in top_nodes]

print(f"      ✓ Computed statistics")
print(f"        - Nodes: {stats['nodes']:,}")
print(f"        - Edges: {stats['edges']:,}")
print(f"        - Density: {stats['density']:.6f}")
print(f"        - Largest component: {stats['largest_component_pct']:.1f}% of nodes")

# ---------- STEP 5: CREATE FILTERED GRAPH ----------
print(f"\n[5/7] Creating filtered graph (minimum {MIN_PAPERS} papers per edge)...")

G_filtered = G.copy()

# Remove edges with fewer than MIN_PAPERS
edges_to_remove = [(u, v) for u, v, data in G_filtered.edges(data=True) 
                   if data["weight"] < MIN_PAPERS]
G_filtered.remove_edges_from(edges_to_remove)

# Remove isolated nodes
isolated = list(nx.isolates(G_filtered))
G_filtered.remove_nodes_from(isolated)

print(f"      ✓ Filtered graph created")
print(f"        - Removed {len(edges_to_remove):,} edges")
print(f"        - Removed {len(isolated):,} isolated nodes")
print(f"        - Resulting graph: {G_filtered.number_of_nodes():,} nodes, {G_filtered.number_of_edges():,} edges")

# ---------- STEP 6: SAVE GRAPHS ----------
print("\n[6/7] Saving graph files...")

# Save full graph
nx.write_graphml(G, FULL_GRAPH)
print(f"      ✓ Saved full graph to {FULL_GRAPH.name}")

# Save filtered graph
nx.write_graphml(G_filtered, FILTERED_GRAPH)
print(f"      ✓ Saved filtered graph to {FILTERED_GRAPH.name}")

# Save statistics
with open(STATS_FILE, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("DRUG-DISEASE ASSOCIATION GRAPH STATISTICS\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("FULL GRAPH\n")
    f.write("-" * 70 + "\n")
    f.write(f"Nodes: {stats['nodes']:,}\n")
    f.write(f"  - Disease nodes: {stats['disease_nodes']:,}\n")
    f.write(f"  - Drug nodes: {stats['drug_nodes']:,}\n")
    f.write(f"Edges: {stats['edges']:,}\n")
    f.write(f"Density: {stats['density']:.6f}\n")
    f.write(f"Average degree: {stats['avg_degree']:.2f}\n")
    f.write(f"Max degree: {stats['max_degree']:,}\n")
    f.write(f"Connected components: {stats['connected_components']:,}\n")
    f.write(f"Largest component: {stats['largest_component_size']:,} nodes ({stats['largest_component_pct']:.1f}%)\n")
    f.write(f"\nTotal papers represented: {stats['total_papers']:,}\n")
    f.write(f"Average papers per edge: {stats['avg_papers_per_edge']:.1f}\n")
    f.write(f"Max papers for single edge: {stats['max_papers_per_edge']:,}\n")
    
    f.write(f"\n\nFILTERED GRAPH (≥{MIN_PAPERS} papers)\n")
    f.write("-" * 70 + "\n")
    f.write(f"Nodes: {G_filtered.number_of_nodes():,}\n")
    f.write(f"Edges: {G_filtered.number_of_edges():,}\n")
    f.write(f"Density: {nx.density(G_filtered):.6f}\n")
    
    f.write("\n\nTOP 10 MOST CONNECTED NODES\n")
    f.write("-" * 70 + "\n")
    for i, (node, degree, name) in enumerate(stats["top_10_nodes"], 1):
        node_type = G.nodes[node]["type"].title()
        f.write(f"{i:2d}. {name[:50]:50s} [{node_type}] (degree: {degree:,})\n")
    
    f.write("\n\nTOP 10 STRONGEST ASSOCIATIONS (by paper count)\n")
    f.write("-" * 70 + "\n")
    top_edges = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)[:10]
    for i, (u, v, data) in enumerate(top_edges, 1):
        drug_name = G.nodes[u]["name"] if G.nodes[u]["type"] == "drug" else G.nodes[v]["name"]
        disease_name = G.nodes[v]["name"] if G.nodes[v]["type"] == "disease" else G.nodes[u]["name"]
        f.write(f"{i:2d}. {drug_name[:30]:30s} ↔ {disease_name[:30]:30s} ({data['weight']:,} papers)\n")

print(f"      ✓ Saved statistics to {STATS_FILE.name}")

# ---------- STEP 7: CREATE INTERACTIVE VISUALIZATIONS ----------
if PYVIS_AVAILABLE:
    print("\n[7/7] Creating interactive visualizations...")
    
    # Function to create interactive viz
    def create_interactive_viz(graph, output_file, max_edges=None):
        # Take only top edges if graph is too large
        if max_edges and graph.number_of_edges() > max_edges:
            print(f"      Graph has {graph.number_of_edges():,} edges. Taking top {max_edges:,} by weight...")
            sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
            edges_to_keep = [(u, v) for u, v, _ in sorted_edges[:max_edges]]
            
            viz_graph = nx.Graph()
            viz_graph.add_edges_from(edges_to_keep)
            
            # Copy node attributes
            for node in viz_graph.nodes():
                viz_graph.nodes[node].update(graph.nodes[node])
            
            # Copy edge attributes
            for u, v in viz_graph.edges():
                viz_graph[u][v].update(graph[u][v])
            
            # Remove isolated nodes
            isolated = list(nx.isolates(viz_graph))
            viz_graph.remove_nodes_from(isolated)
            
            graph = viz_graph
        
        # Create pyvis network
        net = Network(
            height="900px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#000000",
            notebook=False,
            cdn_resources='in_line'
        )
        
        # Set physics options for better layout
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100
          }
        }
        """)
        
        # Add nodes with colors and sizes
        for node in graph.nodes():
            node_data = graph.nodes[node]
            node_type = node_data["type"]
            
            # Color by type
            color = "#e74c3c" if node_type == "disease" else "#3498db"
            
            # Size by degree
            degree = graph.degree(node)
            size = min(10 + degree * 2, 50)  # Scale size but cap at 50
            
            # Create title (tooltip)
            title = f"{node_data['name']}\nType: {node_type.title()}\nConnections: {degree}"
            
            net.add_node(
                node,
                label=node_data["name"][:30],  # Truncate long labels
                title=title,
                color=color,
                size=size,
                shape="dot"
            )
        
        # Add edges
        for u, v, data in graph.edges(data=True):
            weight = data["weight"]
            
            # Scale edge width by paper count
            width = min(1 + weight / 5, 10)  # Scale but cap at 10
            
            # Create title (tooltip)
            title = f"{data.get('drug_name', 'Drug')} ↔ {data.get('disease_name', 'Disease')}\nPapers: {weight}"
            
            net.add_edge(u, v, value=width, title=title)
        
        # Save with UTF-8 encoding fix for Windows
        try:
            # Generate HTML
            html = net.generate_html()
            # Write with explicit UTF-8 encoding
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"      ✓ Saved interactive visualization to {output_file.name}")
            print(f"        Open in browser to explore")
        except Exception as e:
            # Fallback: try the standard save method
            print(f"      Trying alternative save method...")
            net.save_graph(str(output_file))
            print(f"      ✓ Saved interactive visualization to {output_file.name}")
    
    # Create visualization for full graph
    try:
        create_interactive_viz(G, VIZ_HTML, max_edges=TOP_N_EDGES)
    except Exception as e:
        print(f"      ⚠ Could not create full graph visualization: {e}")
    
    # Create visualization for filtered graph
    try:
        create_interactive_viz(G_filtered, VIZ_FILTERED_HTML, max_edges=TOP_N_EDGES)
    except Exception as e:
        print(f"      ⚠ Could not create filtered graph visualization: {e}")

else:
    print("\n[7/7] Skipping interactive visualizations (pyvis not installed)")
    print("      Install with: pip install pyvis")

# ---------- SUMMARY ----------
print("\n" + "=" * 70)
print("✓ Graph construction complete!")
print("=" * 70)
print(f"\nFiles saved to: {GRAPH_DIR}")
print("\nGenerated files:")
print(f"  1. {FULL_GRAPH.name} - Full network (NetworkX GraphML)")
print(f"  2. {FILTERED_GRAPH.name} - Filtered network (≥{MIN_PAPERS} papers)")
print(f"  3. {STATS_FILE.name} - Detailed statistics")
if PYVIS_AVAILABLE:
    print(f"  4. {VIZ_HTML.name} - Interactive visualization (full)")
    print(f"  5. {VIZ_FILTERED_HTML.name} - Interactive visualization (filtered)")

print("\n" + "=" * 70)
print("Next steps:")
print("  - Open the HTML files in a web browser to explore interactively")
print("  - Load GraphML files in Cytoscape or Gephi for advanced analysis")
print("  - Use NetworkX to query and analyze the graph programmatically")
print("=" * 70)