"""
Wrapper to run the unified pipeline 'restore' subcommand.
Use `python pipeline.py restore ...` for the primary entrypoint.
"""
import sys
from pipeline import main


if __name__ == "__main__":
    main(["restore", *sys.argv[1:]])
