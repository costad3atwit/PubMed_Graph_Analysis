# fetch_pubmed_quarters.py
from Bio import Entrez, Medline
import csv, sys, time, datetime as dt
from urllib.error import HTTPError, URLError

# ---------- CONFIG ----------
EMAIL      = "costad3@wit.edu"
API_KEY    = "c44aa473bb61316f789ea8d5f3b35a73f208"
BASE_TERM  = ('("Alzheimer Disease"[MeSH Terms] OR Alzheimer*[Title/Abstract]) '
              'AND hasabstract[text] AND medline[sb]')
START_DATE = "2015/10/01"  # last 10 years up to today
END_DATE   = "2025/10/03"  # today as of time of implementation
BATCH      = 200
MAX_RETRY  = 6
BASE_DELAY = 0.12
OUT_CSV    = "papers.csv"

Entrez.email = EMAIL
if API_KEY:
    Entrez.api_key = API_KEY

# ---------- HELPERS ----------
def quarters_between(start_str, end_str):
    """Yield (mindate, maxdate) per quarter between start and end (inclusive)."""
    def parse(s): y,m,d = map(int, s.split("/")); return dt.date(y,m,d)
    start, end = parse(start_str), parse(end_str)
    # Snap to first day of quarter for the start
    qm = ((start.month - 1)//3)*3 + 1
    cur = dt.date(start.year, qm, 1)
    # Iterate quarters
    while cur <= end:
        q_end_month = cur.month + 2
        year2 = cur.year + (q_end_month-1)//12
        month2 = ((q_end_month-1) % 12) + 1
        # last day of that month
        if month2 == 12:
            last_day = dt.date(year2, month2, 31)
        else:
            last_day = dt.date(year2, month2+1, 1) - dt.timedelta(days=1)
        mind = max(cur, start)
        maxd = min(last_day, end)
        yield (mind.strftime("%Y/%m/%d"), maxd.strftime("%Y/%m/%d"))
        # advance 3 months
        nm = cur.month + 3
        ny = cur.year + (nm-1)//12
        cur = dt.date(ny, ((nm-1)%12)+1, 1)

def esearch_count(term, mindate, maxdate):
    h = Entrez.esearch(db="pubmed", term=term, usehistory="y",
                       datetype="pdat", mindate=mindate, maxdate=maxdate, retmax=0)
    rec = Entrez.read(h)
    return int(rec["Count"]), rec["WebEnv"], rec["QueryKey"]

def efetch_retry(webenv, qk, retstart, retmax):
    delay = BASE_DELAY
    for attempt in range(1, MAX_RETRY+1):
        try:
            return Entrez.efetch(db="pubmed", rettype="medline", retmode="text",
                                 webenv=webenv, query_key=qk,
                                 retstart=retstart, retmax=retmax)
        except (HTTPError, URLError) as e:
            print(f"[{attempt}] efetch @ {retstart} -> {e}", file=sys.stderr)
            if attempt == MAX_RETRY: raise
            time.sleep(delay); delay = min(delay*2, 15)

def months_in(mindate, maxdate):
    """If a quarter still exceeds 10k, split into months."""
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

# ---------- MAIN ----------
def main():
    # De-dup using existing CSV if present
    seen = set()
    try:
        with open(OUT_CSV, encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                seen.add(r["pmid"])
        append_mode = True
    except FileNotFoundError:
        append_mode = False

    out = open(OUT_CSV, "a", newline="", encoding="utf-8")
    w = csv.writer(out)
    if not append_mode:
        w.writerow(["pmid","year","journal","title","abstract","pub_types"])

    for q_start, q_end in quarters_between(START_DATE, END_DATE):
        # Quarter-level search
        count, webenv, qk = esearch_count(BASE_TERM, q_start, q_end)
        print(f"Quarter {q_start}..{q_end} -> {count}", file=sys.stderr)

        # If a quarter still exceeds 10k, split into months
        if count > 10000:
            for m_start, m_end in months_in(q_start, q_end):
                m_count, m_we, m_qk = esearch_count(BASE_TERM, m_start, m_end)
                print(f"  Month {m_start}..{m_end} -> {m_count}", file=sys.stderr)
                harvest_slice(m_we, m_qk, m_count, seen, w)
        else:
            harvest_slice(webenv, qk, count, seen, w)

    out.close()
    print("Done.", file=sys.stderr)

def harvest_slice(webenv, qk, count, seen, writer):
    # page through this slice
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
        time.sleep(BASE_DELAY)  # courteous pacing
        print(f"    wrote {wrote} (retstart={start})", file=sys.stderr)

if __name__ == "__main__":
    main()
