"""
Wrapper to run the unified pipeline 'match' subcommand.
Use `python pipeline.py match ...` for the primary entrypoint.
"""
import sys
from pipeline import main


if __name__ == "__main__":
    main(["match", *sys.argv[1:]])
