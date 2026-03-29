"""
CLI entry point for the ``ream-compress`` command installed by ream-moe.

This module delegates to ``examples/compress_model.py``, which contains the
full argument parser and compression workflow.  It works out of the box for
editable installs (``pip install -e .``).  For non-editable installs the
examples directory is not packaged, so run the script directly instead::

    python examples/compress_model.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Entry point for the ``ream-compress`` CLI command."""
    examples_dir = Path(__file__).parent.parent / "examples"

    if not examples_dir.is_dir():
        print(
            "[ream-moe] Could not locate the 'examples/' directory.\n"
            "For non-editable installs, run the script directly:\n"
            "    python examples/compress_model.py --help",
            file=sys.stderr,
        )
        sys.exit(1)

    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))

    try:
        from compress_model import main as _main  # type: ignore[import]
    except ImportError as exc:
        print(
            f"[ream-moe] Failed to import examples/compress_model.py: {exc}\n"
            "Run it directly:  python examples/compress_model.py --help",
            file=sys.stderr,
        )
        sys.exit(1)

    _main()


if __name__ == "__main__":
    main()
