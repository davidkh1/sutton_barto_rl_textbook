#!/usr/bin/env python3
"""
Download and split the Sutton & Barto RL textbook PDF into chapters.

Reference: http://incompleteideas.net/book/the-book-2nd.html

Examples:
    python scripts/split_rlbook_to_chapters.py --dry-run
    python scripts/split_rlbook_to_chapters.py --chapters 2,3,4
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import urllib.request
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader, PdfWriter


PDF_URL = "http://incompleteideas.net/book/RLbook2020trimmed.pdf"
REPO_ROOT = Path(__file__).parent.parent
PDF_CACHE_DIR = REPO_ROOT / "textbook_chapters"  # Where to store the downloaded PDF


# The actual chapters in Sutton & Barto (2nd edition)
CHAPTER_TITLES = [
    "Introduction",                              # Chapter 1
    "Multi-armed Bandits",                       # Chapter 2
    "Finite Markov Decision Processes",          # Chapter 3
    "Dynamic Programming",                       # Chapter 4
    "Monte Carlo Methods",                       # Chapter 5
    "Temporal-Difference Learning",              # Chapter 6
    "n-step Bootstrapping",                      # Chapter 7
    "Planning and Learning with Tabular Methods",# Chapter 8
    "On-policy Prediction with Approximation",   # Chapter 9
    "On-policy Control with Approximation",      # Chapter 10
    "*Off-policy Methods with Approximation",    # Chapter 11 (has asterisk in PDF)
    "Off-policy Methods with Approximation",     # Chapter 11 (alternate)
    "Eligibility Traces",                        # Chapter 12
    "Policy Gradient Methods",                   # Chapter 13
    "Psychology",                                # Chapter 14
    "Neuroscience",                              # Chapter 15
    "Applications and Case Studies",             # Chapter 16
    "Frontiers",                                 # Chapter 17
]


def slugify_title(title: str) -> str:
    """Convert chapter title to a clean filename slug."""
    cleaned = title.strip().lower()
    cleaned = cleaned.replace("&", " and ")
    cleaned = cleaned.replace("'", "").replace("'", "")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def format_bytes(num_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    units = ["KiB", "MiB", "GiB"]
    size = float(num_bytes)
    for unit in units:
        size /= 1024.0
        if size < 1024.0:
            return f"{size:.1f} {unit}"
    return f"{size:.1f} TiB"


@contextlib.contextmanager
def suppress_output(enabled: bool = True) -> Iterator[None]:
    """Suppress stdout/stderr when enabled."""
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def download_pdf(url: str, output_path: Path, force: bool = False) -> Path:
    """Download PDF if not already present."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force:
        print(f"PDF already exists at {output_path}")
        return output_path

    print(f"Downloading PDF from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to {output_path}")
    return output_path


def get_chapter_boundaries(reader: PdfReader) -> list[tuple[int, str, int, int]]:
    """
    Extract chapter boundaries from PDF bookmarks/outlines.
    Returns list of (chapter_num, title, start_page, end_page) tuples.
    Page numbers are 0-indexed.
    """
    outline = reader.outline
    if not outline:
        raise ValueError("PDF has no outline/bookmarks")

    # Flatten the outline to get all bookmarks
    all_bookmarks: list[tuple[str, int]] = []

    def extract_bookmarks(items) -> None:
        for item in items:
            if isinstance(item, list):
                extract_bookmarks(item)
            else:
                title = item.title
                page_num = reader.get_destination_page_number(item)
                all_bookmarks.append((title, page_num))

    extract_bookmarks(outline)

    # Filter to only actual chapters and assign chapter numbers
    chapters: list[tuple[int, str, int]] = []
    seen_titles: set[str] = set()
    chapter_num = 1

    for title, page in all_bookmarks:
        if title in CHAPTER_TITLES and title not in seen_titles:
            clean_title = title.lstrip('*')
            if clean_title not in seen_titles:
                chapters.append((chapter_num, clean_title, page))
                seen_titles.add(clean_title)
                seen_titles.add(title)
                chapter_num += 1

    # Calculate end pages based on next chapter start
    total_pages = len(reader.pages)
    chapter_ranges: list[tuple[int, str, int, int]] = []

    # Find References page for last chapter end
    ref_page = total_pages - 1
    for title, page in all_bookmarks:
        if title == "References":
            ref_page = page - 1
            break

    for i, (num, title, start) in enumerate(chapters):
        if i + 1 < len(chapters):
            end = chapters[i + 1][2] - 1
        else:
            end = ref_page
        chapter_ranges.append((num, title, start, end))

    return chapter_ranges


def split_pdf(
    pdf_path: Path,
    repo_root: Path,
    chapters_filter: set[int] | None = None,
    dry_run: bool = False,
    quiet: bool = True,
) -> None:
    """Split PDF into chapters, placing each in chapterXX/docs/ at repo root."""
    print(f"Reading PDF from {pdf_path}...")
    with suppress_output(quiet):
        reader = PdfReader(pdf_path, strict=False)
    print(f"Total pages: {len(reader.pages)}")

    chapters = get_chapter_boundaries(reader)
    print(f"\nFound {len(chapters)} chapters:")

    for num, title, start, end in chapters:
        print(f"  {num:2d}. {title} (pages {start+1}-{end+1})")

    print(f"\nSplitting into chapter directories...")

    written_count = 0
    written_bytes = 0

    for num, title, start, end in chapters:
        if chapters_filter is not None and num not in chapters_filter:
            continue

        # Create chapter directory with docs subdirectory at repo root
        chapter_dir = repo_root / f"chapter{num:02d}" / "docs"

        # Create filename: ch02_multi_armed_bandits.pdf
        slug = slugify_title(title)
        output_filename = f"ch{num:02d}_{slug}.pdf"
        output_path = chapter_dir / output_filename
        page_count = end - start + 1

        if dry_run:
            print(f"  chapter{num:02d}/docs/{output_filename} ({page_count} pages)")
            continue

        chapter_dir.mkdir(parents=True, exist_ok=True)

        writer = PdfWriter()
        for page_num in range(start, end + 1):
            writer.add_page(reader.pages[page_num])

        with open(output_path, 'wb') as f:
            writer.write(f)

        size_bytes = output_path.stat().st_size
        written_count += 1
        written_bytes += size_bytes
        print(f"  chapter{num:02d}/docs/{output_filename} ({page_count} pages) - {format_bytes(size_bytes)}")

    if not dry_run:
        print(f"\nDone: wrote {written_count} chapter PDFs ({format_bytes(written_bytes)} total)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pdf-url",
        default=PDF_URL,
        help="Source PDF URL",
    )
    parser.add_argument(
        "--chapters",
        help="Comma-separated chapter numbers to extract (e.g., 2,3,4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print chapter info without writing files",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the PDF even if it exists",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show pypdf warnings while parsing",
    )
    args = parser.parse_args(argv)

    # Parse chapters filter
    chapters_filter: set[int] | None = None
    if args.chapters:
        try:
            chapters_filter = {int(c.strip()) for c in args.chapters.split(",") if c.strip()}
        except ValueError:
            print("--chapters must be a comma-separated list of integers", file=sys.stderr)
            return 1

    # Download PDF to cache directory
    pdf_path = PDF_CACHE_DIR / "RLbook2020trimmed.pdf"
    try:
        download_pdf(args.pdf_url, pdf_path, force=args.force_download)
    except Exception as e:
        print(f"Failed to download PDF: {e}", file=sys.stderr)
        return 1

    # Split into chapters at repo root
    try:
        split_pdf(
            pdf_path,
            REPO_ROOT,
            chapters_filter=chapters_filter,
            dry_run=args.dry_run,
            quiet=not args.verbose,
        )
    except Exception as e:
        print(f"Failed to split PDF: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
