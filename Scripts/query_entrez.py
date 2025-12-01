# fetch_pubmed_quarters.py
from Bio import Entrez, Medline
import csv, sys, time, datetime as dt
from urllib.error import HTTPError, URLError

# ---------- CONFIG ----------
EMAIL      = "costad3@wit.edu"
API_KEY    = "c44aa473bb61316f789ea8d5f3b35a73f208"
# IMPORTANT: Update BASE_TERM to match your actual search
BASE_TERM  = ('hasabstract[text] AND medline[sb]')
START_DATE = "2015/10/01"
END_DATE   = "2025/10/03"
BATCH      = 200
MAX_RETRY  = 6
BASE_DELAY = 0.2  # Increased from 0.12 for broader searches
OUT_CSV    = "papers.csv"
MAX_RESULTS_PER_CHUNK = 10000  # PubMed's hard limit

Entrez.email = EMAIL
if API_KEY:
    Entrez.api_key = API_KEY

# ---------- HELPERS ----------
def quarters_between(start_str, end_str):
    """Yield (mindate, maxdate) per quarter between start and end (inclusive)."""
    def parse(s): y,m,d = map(int, s.split("/")); return dt.date(y,m,d)
    start, end = parse(start_str), parse(end_str)
    qm = ((start.month - 1)//3)*3 + 1
    cur = dt.date(start.year, qm, 1)
    while cur <= end:
        q_end_month = cur.month + 2
        year2 = cur.year + (q_end_month-1)//12
        month2 = ((q_end_month-1) % 12) + 1
        if month2 == 12:
            last_day = dt.date(year2, month2, 31)
        else:
            last_day = dt.date(year2, month2+1, 1) - dt.timedelta(days=1)
        mind = max(cur, start)
        maxd = min(last_day, end)
        yield (mind.strftime("%Y/%m/%d"), maxd.strftime("%Y/%m/%d"))
        nm = cur.month + 3
        ny = cur.year + (nm-1)//12
        cur = dt.date(ny, ((nm-1)%12)+1, 1)

def months_in(mindate, maxdate):
    """Split a date range into months."""
    y1,m1,_ = map(int, mindate.split("/"))
    y2,m2,_ = map(int, maxdate.split("/"))
    cur = dt.date(y1,m1,1)
    end = dt.date(y2,m2,1)
    while cur <= end:
        ny, nm = (cur.year + (cur.month==12)), (1 if cur.month==12 else cur.month+1)
        month_end = dt.date(ny, nm, 1) - dt.timedelta(days=1)
        md = cur.strftime("%Y/%m/%d")
        mx = min(month_end, dt.datetime.strptime(maxdate,"%Y/%m/%d").date()).strftime("%Y/%m/%d")
        yield (md, mx)
        cur = dt.date(ny, nm, 1)

def days_in(mindate, maxdate):
    """Split a date range into individual days."""
    start = dt.datetime.strptime(mindate, "%Y/%m/%d").date()
    end = dt.datetime.strptime(maxdate, "%Y/%m/%d").date()
    cur = start
    while cur <= end:
        yield (cur.strftime("%Y/%m/%d"), cur.strftime("%Y/%m/%d"))
        cur += dt.timedelta(days=1)

def esearch_count(term, mindate, maxdate):
    """Get count and history for a date range."""
    h = Entrez.esearch(db="pubmed", term=term, usehistory="y",
                       datetype="pdat", mindate=mindate, maxdate=maxdate, retmax=0)
    rec = Entrez.read(h)
    return int(rec["Count"]), rec["WebEnv"], rec["QueryKey"]

def efetch_retry(webenv, qk, retstart, retmax):
    """Fetch with exponential backoff retry."""
    delay = BASE_DELAY
    for attempt in range(1, MAX_RETRY+1):
        try:
            return Entrez.efetch(db="pubmed", rettype="medline", retmode="text",
                                 webenv=webenv, query_key=qk,
                                 retstart=retstart, retmax=retmax)
        except (HTTPError, URLError) as e:
            print(f"[{attempt}] efetch @ {retstart} -> {e}", file=sys.stderr)
            if attempt == MAX_RETRY: raise
            time.sleep(delay)
            delay = min(delay*2, 15)

def harvest_slice(webenv, qk, count, seen, writer):
    """Fetch and write results for a date range (must be ≤10k results)."""
    # Safety check - use 9999 to stay safely under the 10k limit
    if count > 9999:
        count = 9999
    
    for start in range(0, count, BATCH):
        h = efetch_retry(webenv, qk, start, BATCH)
        recs = Medline.parse(h)
        wrote = 0
        for rec in recs:
            pmid = rec.get("PMID","")
            if not pmid or pmid in seen:
                continue
            seen.add(pmid)
            year = (rec.get("DP","") or "").split(" ")[0][:4]
            writer.writerow([
                pmid,
                year,
                rec.get("JT",""),
                rec.get("TI",""),
                rec.get("AB",""),
                ";".join(rec.get("PT", []))
            ])
            wrote += 1
        h.close()
        time.sleep(BASE_DELAY)
        print(f"    wrote {wrote} (retstart={start})", file=sys.stderr)

def process_date_range(d_start, d_end, seen, writer, depth=0):
    """
    Recursively process a date range, subdividing if necessary.
    depth: 0=quarter, 1=month, 2=day
    """
    indent = "  " * depth
    range_labels = ["Quarter", "Month", "Day"]
    label = range_labels[depth] if depth < 3 else "Range"
    
    count, webenv, qk = esearch_count(BASE_TERM, d_start, d_end)
    print(f"{indent}{label} {d_start}..{d_end} -> {count:,} results", file=sys.stderr)
    
    # If under limit, harvest directly
    if count <= MAX_RESULTS_PER_CHUNK:
        harvest_slice(webenv, qk, count, seen, writer)
        return
    
    # Subdivide based on depth
    if depth == 0:  # Quarter -> Months
        print(f"{indent}  Splitting into months...", file=sys.stderr)
        for m_start, m_end in months_in(d_start, d_end):
            process_date_range(m_start, m_end, seen, writer, depth=1)
    
    elif depth == 1:  # Month -> Days
        print(f"{indent}  Splitting into days...", file=sys.stderr)
        for day_start, day_end in days_in(d_start, d_end):
            process_date_range(day_start, day_end, seen, writer, depth=2)
    
    else:  # Day level - can't subdivide further
        if count > MAX_RESULTS_PER_CHUNK:
            print(f"{indent}  Day has {count:,} results, taking first 9,999", file=sys.stderr)
            harvest_slice(webenv, qk, 9999, seen, writer)
        else:
            harvest_slice(webenv, qk, count, seen, writer)

# ---------- MAIN ----------
def main():
    # Check for existing data
    seen = set()
    try:
        with open(OUT_CSV, encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                seen.add(r["pmid"])
        append_mode = True
        print(f"Resuming: {len(seen):,} PMIDs already fetched", file=sys.stderr)
    except FileNotFoundError:
        append_mode = False
        print("Starting fresh fetch", file=sys.stderr)

    out = open(OUT_CSV, "a", newline="", encoding="utf-8")
    w = csv.writer(out)
    if not append_mode:
        w.writerow(["pmid","year","journal","title","abstract","pub_types"])

    # Process each quarter
    try:
        for q_start, q_end in quarters_between(START_DATE, END_DATE):
            process_date_range(q_start, q_end, seen, w, depth=0)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved.", file=sys.stderr)
    finally:
        out.close()
        print(f"\nTotal unique PMIDs fetched: {len(seen):,}", file=sys.stderr)

if __name__ == "__main__":
    main()