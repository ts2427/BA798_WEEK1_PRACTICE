#!/usr/bin/env python
"""
Execute Jupyter notebook programmatically.
Runs all cells in the notebook and saves output.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

try:
    import nbconvert
    from nbconvert.preprocessors import ExecutePreprocessor
    import nbformat
except ImportError:
    print("Error: nbconvert not installed")
    print("Install with: pip install nbconvert")
    sys.exit(1)


def execute_notebook(notebook_path, output_path=None, timeout=600):
    """
    Execute a Jupyter notebook and save the result.

    Args:
        notebook_path: Path to .ipynb file
        output_path: Path to save executed notebook (optional)
        timeout: Execution timeout in seconds

    Returns:
        bool: True if successful, False otherwise
    """
    notebook_path = Path(notebook_path)

    if not notebook_path.exists():
        print(f"Error: Notebook not found: {notebook_path}")
        return False

    print(f"Executing notebook: {notebook_path}")
    print(f"Timeout: {timeout} seconds")
    print()

    try:
        # Load notebook
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Configure executor
        ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')

        # Execute notebook
        print("[EXECUTING NOTEBOOK]")
        print("-" * 60)

        nb, resources = ep.preprocess(nb, {'metadata': {'path': notebook_path.parent}})

        print("-" * 60)
        print("[EXECUTION COMPLETE]")
        print()

        # Save executed notebook if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                nbformat.write(nb, f)

            print(f"Executed notebook saved to: {output_path}")

        # Print summary
        print("\nExecution Summary:")
        print(f"  Notebook: {notebook_path.name}")
        print(f"  Cells: {len(nb.cells)}")
        print(f"  Status: SUCCESS")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return True

    except Exception as e:
        print(f"\nError executing notebook: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Execute Jupyter notebook and save results'
    )
    parser.add_argument(
        'notebook',
        help='Path to notebook file (.ipynb)'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output path for executed notebook (optional)',
        default=None
    )
    parser.add_argument(
        '-t', '--timeout',
        type=int,
        default=600,
        help='Execution timeout in seconds (default: 600)'
    )

    args = parser.parse_args()

    # Execute notebook
    success = execute_notebook(
        args.notebook,
        output_path=args.output,
        timeout=args.timeout
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
