"""
Master Pipeline Runner for Drug-Disease Association Graph Project

This script runs the entire pipeline in sequence:
1. parse_terms.py - Parse MeSH terms from d2025.bin
2. test_entrez.py - Fetch PubMed papers (can be skipped if already done)
3. detect_co_mentions_filtered.py - Detect co-mentions with specificity filtering
4. aggregate_co_mentions.py - Aggregate co-mentions by drug-disease pairs
5. normalize_terms_embeddings.py - Normalize synonyms using BioWordVec
6. construct_graph.py - Build the final graph
7. frequency_EDA.py - (Optional) Generate frequency analysis and visualizations
8. network_analysis.py - (Optional) Analyze network structure and comorbidity
9. visualize_clusters.py - (Optional) Visualize synonym clusters

Usage:
    python run_pipeline.py [options]
    
Options:
    --skip-fetch        Skip PubMed fetching (use existing papers.csv)
    --skip-eda          Skip frequency EDA visualizations
    --skip-network      Skip network analysis visualizations
    --skip-viz          Skip cluster visualization
    --skip-all-viz      Skip all visualization steps (EDA, network, and clusters)
    --auto              Run without prompts for optional steps
    --from-step N       Start from step N (1-9)
"""

import subprocess
import sys
import time

import argparse
import logging
from datetime import datetime

# ---------- CONFIG ----------
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "Data"
FIGURES_DIR = PROJECT_ROOT / "Figures"
LOGS_DIR = PROJECT_ROOT / "Logs"
GRAPHS_DIR = PROJECT_ROOT / "Graphs"

# Create directories if they don't exist already (they should)
FIGURES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
GRAPHS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

def setup_logging():
    """Setup logging with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"pipeline_run_{timestamp}.log"
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

# Pipeline steps in order
PIPELINE_STEPS = [
    {
        "number": 1,
        "name": "Parse MeSH Terms",
        "script": "parse_terms.py",
        "description": "Extract diseases and drugs from MeSH descriptors",
        "optional": False
    },
    {
        "number": 2,
        "name": "Fetch PubMed Papers",
        "script": "query_entrez.py",
        "description": "Download Alzheimer's papers from PubMed (can take hours)",
        "optional": True
    },
    {
        "number": 3,
        "name": "Detect Co-mentions",
        "script": "detect_co_mentions.py",
        "description": "Find drug-disease co-mentions in papers (with specificity filtering)",
        "optional": False
    },
    {
        "number": 4,
        "name": "Aggregate Co-mentions",
        "script": "aggregate_co_mentions.py",
        "description": "Aggregate co-mentions by drug-disease pairs",
        "optional": False
    },
    {
        "number": 5,
        "name": "Normalize Terms",
        "script": "normalize_terms.py",
        "description": "Cluster synonyms using BioWordVec embeddings",
        "optional": False
    },
    {
        "number": 6,
        "name": "Construct Graph",
        "script": "construct_graph.py",
        "description": "Build the drug-disease association network",
        "optional": False
    },
    {
        "number": 7,
        "name": "Frequency EDA",
        "script": "frequency_EDA.py",
        "description": "Generate frequency analysis and visualizations (SLOW!)",
        "optional": True
    },
    {
        "number": 8,
        "name": "Network Analysis",
        "script": "network_analysis.py",
        "description": "Analyze network structure and generate comorbidity insights",
        "optional": True
    },
    {
        "number": 9,
        "name": "Visualize Clusters",
        "script": "visualize_term_clusters.py",
        "description": "Generate visualizations of synonym clusters",
        "optional": True
    }
]

# ---------- HELPER FUNCTIONS ----------
def print_header(text):
    """Print a formatted header"""
    logging.info("")
    logging.info("=" * 80)
    logging.info(text.center(80))
    logging.info("=" * 80)
    logging.info("")

def print_step_header(step):
    """Print step information"""
    logging.info("")
    logging.info("─" * 80)
    logging.info(f"STEP {step['number']}/9: {step['name'].upper()}")
    logging.info(f"Script: {step['script']}")
    logging.info(f"Description: {step['description']}")
    logging.info("─" * 80)
    logging.info("")

def run_script(script_path):
    """Run a Python script and stream output in real-time to log"""
    try:
        # Use Popen to stream output line by line (not run() which buffers!)
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            encoding='utf-8',
            bufsize=1,  # Line buffered - enables real-time streaming
            universal_newlines=True
        )
        
        # Stream output line by line in real-time
        for line in process.stdout:
            line = line.rstrip('\n')
            if line:  # Don't log empty lines
                logging.info(f"  {line}")
        
        # Wait for process to complete and get return code
        return_code = process.wait()
        
        if return_code == 0:
            return True
        else:
            logging.error(f"Script failed with return code {return_code}")
            return False
            
    except FileNotFoundError:
        logging.error(f"Script not found: {script_path}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def confirm_step(step):
    """Ask user to confirm running a step"""
    while True:
        response = input(f"Run step {step['number']} ({step['name']})? [Y/n]: ").strip().lower()
        if response in ['', 'y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            logging.info("Please enter 'y' or 'n'")

# ---------- MAIN PIPELINE ----------
def main():

    log_file = setup_logging()
    parser = argparse.ArgumentParser(description="Run the drug-disease graph pipeline")
    parser.add_argument("--skip-fetch", action="store_true", 
                       help="Skip PubMed fetching step (use existing papers.csv)")
    parser.add_argument("--skip-eda", action="store_true",
                       help="Skip frequency EDA visualizations (can be slow)")
    parser.add_argument("--skip-network", action="store_true",
                       help="Skip network analysis visualizations")
    parser.add_argument("--skip-viz", action="store_true",
                       help="Skip cluster visualization step")
    parser.add_argument("--skip-all-viz", action="store_true",
                       help="Skip all visualization steps (EDA, network, and clusters)")
    parser.add_argument("--from-step", type=int, metavar="N",
                       help="Start from step N (1-9)")
    parser.add_argument("--auto", action="store_true",
                       help="Run all steps automatically without prompts")
    
    args = parser.parse_args()
    
    # Handle --skip-all-viz convenience flag
    if args.skip_all_viz:
        args.skip_eda = True
        args.skip_network = True
        args.skip_viz = True
    

    logging.info("=" * 80)
    logging.info("DRUG-DISEASE ASSOCIATION GRAPH PIPELINE")
    logging.info("=" * 80)
    logging.info(f"Log file: {log_file}")
    print_header("DRUG-DISEASE ASSOCIATION GRAPH PIPELINE")
    
    logging.info("Pipeline Steps:")
    for step in PIPELINE_STEPS:
        optional_tag = " (optional)" if step['optional'] else ""
        logging.info(f"  {step['number']}. {step['name']}{optional_tag}")
    
    # Determine which steps to run
    start_step = args.from_step if args.from_step else 1
    if start_step < 1 or start_step > 9:
        logging.info(f"Error: --from-step must be between 1 and 9")
        return 1
    
    if start_step > 1:
        logging.info(f"\nStarting from step {start_step}")
    
    # Track timing
    total_start = time.time()
    step_times = []
    
    # Run each step
    for step in PIPELINE_STEPS:
        if step['number'] < start_step:
            continue
        
        # Check if step should be skipped
        if args.skip_fetch and step['number'] == 2:
            logging.info(f"\nSkipping Step {step['number']} (--skip-fetch specified)")
            continue
        
        if args.skip_eda and step['number'] == 7:
            logging.info(f"\nSkipping Step {step['number']} (--skip-eda specified)")
            continue
        
        if args.skip_network and step['number'] == 8:
            logging.info(f"\nSkipping Step {step['number']} (--skip-network specified)")
            continue
        
        if args.skip_viz and step['number'] == 9:
            logging.info(f"\nSkipping Step {step['number']} (--skip-viz specified)")
            continue
        
        # Show step information
        print_step_header(step)
        
        # Ask for confirmation if not in auto mode and step is optional
        if not args.auto and step['optional']:
            if not confirm_step(step):
                logging.info(f"Skipping step {step['number']}")
                continue
        
        # Check if script exists
        script_path = SCRIPT_DIR / step['script']
        
        # Run the script
        logging.info(f"Running {step['script']}...")
        step_start = time.time()
        
        success = run_script(script_path)
        
        step_time = time.time() - step_start
        step_times.append((step['name'], step_time))
        
        if success:
            logging.info(f"\n Step {step['number']} completed in {step_time:.1f} seconds")
        else:
            logging.info(f"\n Step {step['number']} FAILED")
            
            if step['optional']:
                logging.warning(f"  Continuing despite failure in optional step...")
            else:
                logging.error(f"  Pipeline stopped due to failure in required step")
                return 1
        
        # Pause between steps
        if step['number'] < 9:  # Don't pause after last step
            time.sleep(1)
    
    # Summary
    total_time = time.time() - total_start
    
    print_header("PIPELINE COMPLETED SUCCESSFULLY!")
    
    logging.info("Step Timing Summary:")
    logging.info("─" * 80)
    for step_name, step_time in step_times:
        logging.info(f"  {step_name:40s} {step_time:8.1f}s  ({step_time/60:6.1f} min)")
    logging.info("─" * 80)
    logging.info(f"  {'TOTAL':40s} {total_time:8.1f}s  ({total_time/60:6.1f} min)")
    logging.info("─" * 80)
    
    logging.info("\nOutput files are in the Data directory:")
    logging.info("  - terms.csv")
    logging.info("  - papers.csv")
    logging.info("  - co-mentions.csv")
    logging.info("  - aggregated.csv")
    logging.info("  - aggregated_canonical.csv")
    logging.info("  - graphs/drug_disease_graph.graphml")
    logging.info("  - graphs/interactive_graph.html")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.info("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.info(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
