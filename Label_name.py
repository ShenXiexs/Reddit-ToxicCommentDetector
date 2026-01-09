"""
Wrapper to run the unified pipeline 'rename-labels' subcommand.
Use `python pipeline.py rename-labels ...` for the primary entrypoint.
"""
import sys
from pipeline import main


if __name__ == "__main__":
    main(["rename-labels", *sys.argv[1:]])
