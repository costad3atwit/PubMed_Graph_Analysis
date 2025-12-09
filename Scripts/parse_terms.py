import time, csv
import pandas as pd
import os

# ---------- CONFIG ----------
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "Data"
FIGURES_DIR = PROJECT_ROOT / "Figures"
LOGS_DIR = PROJECT_ROOT / "Logs"
GRAPHS_DIR = PROJECT_ROOT / "Graphs"

DATA_PATH = DATA_DIR / "d2025.bin"
DELIMITER = "*NEWRECORD"

print("Data path resolved at: " + str(DATA_PATH))


def read_records(file_path, delim):
    """Generator that yields record strings between delimiters."""
    record_lines = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == delim:
                # If we already have content collected, yield it
                if record_lines:
                    yield "\n".join(record_lines)
                    record_lines = []
            else:
                record_lines.append(line)
        # Yield the last record if file doesn't end with a delimiter
        if record_lines:
            yield "\n".join(record_lines)

def parse_record(record_text):
    """Parse a record's text into a structured dictionary."""
    parsed = {}
    
    for line in record_text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("UI = "):
            parsed["id"] = line[len("UI = "):].strip()
        elif line.startswith("MH = "):
            parsed["term"] = line[len("MH = "):].strip()
        elif line.startswith("ENTRY = "):
            parsed.setdefault("entries", []).append(line[len("ENTRY = "):].strip().split("|"))
        elif line.startswith("PRINT ENTRY = "):
            parsed.setdefault("entries", []).append(line[len("PRINT ENTRY = "):].strip().split("|"))
        elif line.startswith("MN = "):
            parsed.setdefault("tree_nums", []).append(line[len("MN = "):].strip())
        elif line.startswith("ST = "):
            parsed.setdefault("semantic_types", []).append(line[len("ST = "):].strip())

    return parsed

def pretty_print_record(record_dict, indent=2):
    """Nicely format a parsed record dictionary."""
    for key, value in record_dict.items():
        if key == "entries" and isinstance(value, list):
            print(f"{key}:")
            for sublist in value:
                print(" " * indent + "- " + " | ".join(sublist))
        elif isinstance(value, list):
            print(f"{key}: {', '.join(value)}")
        else:
            print(f"{key}: {value}")




terms_csv = PROJECT_ROOT / "Data" / "terms.csv"
header_written = os.path.exists(terms_csv)

for record_text in read_records(DATA_PATH, DELIMITER):
    parsed = parse_record(record_text)

    # Filter: skip if no tree_nums or none start with C/D
    if "tree_nums" not in parsed:
        continue
    if not any(str(num).startswith(("C", "D")) for num in parsed["tree_nums"]):
        continue

    # Flatten lists for CSV
    if "entries" in parsed:
        parsed["entries"] = "\n".join([" | ".join(sublist) for sublist in parsed["entries"]])
    for key in ["tree_nums", "semantic_types"]:
        if key in parsed and isinstance(parsed[key], list):
            parsed[key] = ", ".join(parsed[key])

    # Append this record directly to CSV
    df = pd.DataFrame([parsed])
    df.to_csv(terms_csv, mode="a", index=False, header=not header_written, encoding="utf-8")
    header_written = True
