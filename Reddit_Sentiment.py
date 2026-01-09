"""
Wrapper to run the unified pipeline 'sentiment' subcommand.
Use `python pipeline.py sentiment ...` for the primary entrypoint.
"""
import sys
from pipeline import main


if __name__ == "__main__":
    main(["sentiment", *sys.argv[1:]])
