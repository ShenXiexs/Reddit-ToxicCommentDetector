"""
Wrapper to run the unified pipeline 'predict' subcommand.
Use `python pipeline.py predict ...` for the primary entrypoint.
"""
import sys
from pipeline import main


if __name__ == "__main__":
    main(["predict", *sys.argv[1:]])
