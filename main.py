import sys
from pathlib import Path
from src.data_utils import load_json, publish_directory


CURRENT_DIR = Path("./current")
chunks_dir = CURRENT_DIR / "chunks"
experiments_dir = CURRENT_DIR / "experiments"
figures_dir = CURRENT_DIR / "figures"
FINAL_DIR = Path("./final")
QUESTIONS_PATH = Path("./data/questions.json")
NUM_RUNS = 5
MAIN_CHUNKING_METHOD = "recursive"

# Testing config
SMALL_RUNS = 1
SMALL_QUESTIONS = 1


# --- Main Function ---

def main(command, options):

    # Slow imports (moved to avoid waiting if bad usage)
    print("Importing libraries ...")
    from src.chunks import build_all_methods_chunks, load_all_chunks
    from src.experiments import run_all_experiments
    from src.figures import build_all_figures
    print("Libraries imported\n")

    all_chunks = None
    metrics = None

    # Chunks command
    if command in ["chunks", "all"]:

        all_chunks, all_vstores = build_all_methods_chunks(saving_dir=chunks_dir)


    # Experiments command
    if command in ["experiments", "all"]:

        # Load questions and set runs
        questions = load_json(QUESTIONS_PATH)
        num_runs = NUM_RUNS

        # Small experiment
        if "--small" in options:
            questions = questions[:SMALL_QUESTIONS]
            num_runs = SMALL_RUNS

        # If chunking command not called
        if all_chunks is None:

            # Load final chunks
            try:
                all_chunks, all_vstores = load_all_chunks(chunks_dir)

            except Exception as e:
                print(f"Problem while loading chunks from directory {chunks_dir}, for running experiments.\n",
                    "Use 'python main.py chunks' for creating chunks, or 'python main.py all' for running everything.\n",
                    "Exception:\n{e}")
                return 2
             
        # Run experiments
        _, metrics = run_all_experiments(all_chunks, all_vstores, questions, num_runs,
                                          saving_dir=experiments_dir)
        

    # Figures command
    if command in ["figures", "all"]:

        # If experiments command not called
        if metrics is None:

            # Load final metrics
            try:
                # Load from path
                metrics_path = experiments_dir / "all_metrics.json"
                metrics = load_json(metrics_path)

            except Exception as e:
                print(f"Problem while loading metrics from path {metrics_path}, in order to build figures.\n",
                    "Use 'python main.py experiments' for creating metrics, or 'python main.py all' for running everything.\n",
                    "Exception:\n{e}")
                return 2

        build_all_figures(metrics, MAIN_CHUNKING_METHOD, saving_dir=figures_dir)

    return 0


if __name__ == "__main__":

    # --- CLI ---

    args = sys.argv[1:]

    if args:
            
        command = args[0]
        options = args[1:]

        if command == "publish":
            publish_directory(input_dir=CURRENT_DIR, output_dir=FINAL_DIR)
            sys.exit(0)
            
        # Ensure appropiate commands
        elif command in {"all", "chunks", "experiments", "figures"}:
            
            # Ensure appropiate options
            if all(o in {"--small"} for o in options):
                
                # If correct usage, run main
                sys.exit(main(command, options))


    # If bad usage, show instructions
    print("""
        Usage: python main.py <command> [options]

        Commands:
            all          Run the full pipeline (chunking, experiments and figures)
            chunks       Generate chunks
            experiments  Run experiments and generate metrics
            figures      Generate plots
            publish      Promote current directory to final one

        Options:
            --small     Use a smaller dataset (for testing)
        """)
    sys.exit(1)
