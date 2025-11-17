import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from pathlib import Path
import re
import time

start_time = time.time()

print("=== Frequency Analysis Script Started ===")

# === File paths ===
DATA_DIR = Path("C:/Users/sirda/Dropbox (Personal)/Documents/Fall '25/Data Mining/PubMed Project/Data")
TERMS_PATH = DATA_DIR / "terms.csv"
PAPERS_PATH = DATA_DIR / "papers.csv"

print(f"Reading files from: {DATA_DIR}")

# === Load data ===
try:
    terms_df = pd.read_csv(TERMS_PATH)
    papers_df = pd.read_csv(PAPERS_PATH)
    print(f"Loaded {len(terms_df)} terms and {len(papers_df)} papers")
except FileNotFoundError as e:
    print(f"File not found: {e}")
    exit(1)

# === Detect correct title column automatically ===
title_col = None
for c in papers_df.columns:
    if "title" in c.lower():
        title_col = c
        break

if title_col is None:
    raise KeyError("No column containing 'title' was found in papers.csv")

print(f"Detected title column: '{title_col}'")

# === Prepare term list ===
terms = [str(t).lower().strip() for t in terms_df["term"].dropna()]
print(f"Prepared {len(terms)} terms for frequency matching")

# === Combine all paper titles ===
titles = papers_df[title_col].dropna().astype(str)
titles_text = " ".join(titles).lower()
print(f"Total combined title text length: {len(titles_text):,} characters")

# === Frequency Analysis ===
term_counts = Counter()
print("Counting term frequencies...")

for i, term in enumerate(terms, start=1):
    matches = re.findall(rf"\b{re.escape(term)}\b", titles_text)
    if matches:
        term_counts[term] = len(matches)
    if i % 500 == 0:
        print(f"Processed {i} / {len(terms)} terms...")

print(f"Found {len(term_counts)} terms that appear in titles")

# === Convert to DataFrame ===
freq_df = pd.DataFrame(term_counts.items(), columns=["term", "count"])
freq_df = freq_df.sort_values("count", ascending=False)
print("Top 5 most frequent terms:")
print(freq_df.head(5).to_string(index=False))

# === Plot top 20 terms ===
if not freq_df.empty:
    plt.figure(figsize=(10, 6))
    plt.barh(freq_df.head(20)["term"][::-1], freq_df.head(20)["count"][::-1])
    plt.xlabel("Frequency")
    plt.ylabel("Term")
    plt.title("Top 20 Terms in Paper Titles")
    plt.tight_layout()
    plt.show()

    # === Generate Word Cloud ===
    wc = WordCloud(width=1000, height=600, background_color="white").generate_from_frequencies(term_counts)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Terms in Paper Titles")
    plt.show()
else:
    print("No matching terms found to visualize.")

# === REFINED DATA ANALYSIS (using co-mentions after specificity filtering) ===
print("\n=== Analyzing Refined Data (Post-Filtering) ===")

CO_MENTIONS_PATH = DATA_DIR / "co-mentions.csv"
AGGREGATED_PATH = DATA_DIR / "aggregated.csv"

try:
    co_mentions_df = pd.read_csv(CO_MENTIONS_PATH)
    aggregated_df = pd.read_csv(AGGREGATED_PATH)
    print(f"Loaded {len(co_mentions_df)} co-mentions and {len(aggregated_df)} aggregated pairs")
    
    # === Drug frequency analysis ===
    drug_counts = Counter(co_mentions_df['drug_name'].dropna())
    disease_counts = Counter(co_mentions_df['disease_name'].dropna())
    
    print(f"\nFound {len(drug_counts)} unique drugs and {len(disease_counts)} unique diseases")
    print("\nTop 10 most mentioned drugs:")
    for drug, count in drug_counts.most_common(10):
        print(f"  {drug}: {count}")
    
    print("\nTop 10 most mentioned diseases:")
    for disease, count in disease_counts.most_common(10):
        print(f"  {disease}: {count}")
    
    # === Plot top drugs ===
    drug_freq_df = pd.DataFrame(drug_counts.most_common(20), columns=["drug", "mentions"])
    
    plt.figure(figsize=(12, 7))
    plt.barh(drug_freq_df["drug"][::-1], drug_freq_df["mentions"][::-1], color='steelblue')
    plt.xlabel("Number of Co-Mentions", fontsize=12)
    plt.ylabel("Drug", fontsize=12)
    plt.title("Top 20 Drugs by Co-Mention Frequency (After Specificity Filtering)", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # === Plot top diseases ===
    disease_freq_df = pd.DataFrame(disease_counts.most_common(20), columns=["disease", "mentions"])
    
    plt.figure(figsize=(12, 7))
    plt.barh(disease_freq_df["disease"][::-1], disease_freq_df["mentions"][::-1], color='darkred')
    plt.xlabel("Number of Co-Mentions", fontsize=12)
    plt.ylabel("Disease", fontsize=12)
    plt.title("Top 20 Diseases by Co-Mention Frequency (After Specificity Filtering)", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # === Word clouds for refined data ===
    if drug_counts:
        drug_wc = WordCloud(width=1200, height=600, background_color="white", 
                           colormap="Blues").generate_from_frequencies(drug_counts)
        plt.figure(figsize=(12, 6))
        plt.imshow(drug_wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud: Drugs (Refined Data)", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    if disease_counts:
        disease_wc = WordCloud(width=1200, height=600, background_color="white",
                              colormap="Reds").generate_from_frequencies(disease_counts)
        plt.figure(figsize=(12, 6))
        plt.imshow(disease_wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud: Diseases (Refined Data)", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # === Analysis from aggregated data (showing strongest associations) ===
    print("\n=== Top Drug-Disease Associations (by paper count) ===")
    top_associations = aggregated_df.nlargest(15, 'paper_count')
    
    for idx, row in top_associations.iterrows():
        print(f"{row['drug_name']} ↔ {row['disease_name']}: {row['paper_count']} papers "
              f"({row['first_paper']}-{row['last_paper']})")
    
    # === Co-mention Location Analysis ===
    print("\n=== Co-Mention Location Analysis ===")
    location_counts = co_mentions_df['where_found'].value_counts()
    print("Co-mentions by location:")
    for location, count in location_counts.items():
        print(f"  {location}: {count} ({count/len(co_mentions_df)*100:.1f}%)")
    
    # Pie chart for co-mention locations
    plt.figure(figsize=(8, 8))
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0.05, 0.05)
    plt.pie(location_counts.values, labels=location_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors, explode=explode, textprops={'fontsize': 12})
    plt.title("Co-Mention Location Distribution\n(Title, Abstract, or Both)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()
    
    # === Distribution plots ===
    plt.figure(figsize=(14, 5))
    
    # Paper count distribution
    plt.subplot(1, 2, 1)
    plt.hist(aggregated_df['paper_count'], bins=50, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel("Papers per Drug-Disease Pair", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.title("Distribution of Evidence Strength", fontsize=12)
    plt.yscale('log')
    
    # Time span distribution
    aggregated_df['time_span'] = aggregated_df['last_paper'] - aggregated_df['first_paper']
    plt.subplot(1, 2, 2)
    plt.hist(aggregated_df['time_span'].dropna(), bins=30, color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel("Years Between First and Last Paper", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.title("Research Timeline Spans", fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
except FileNotFoundError as e:
    print(f"\nRefined data files not found: {e}")
    print("Run the co-mention detection and aggregation scripts first to generate refined data.")

# === Wrap up ===
runtime = time.time() - start_time
print(f"\n=== Script complete. Runtime: {runtime:.2f} seconds ===")