import pytest
import sys
from pathlib import Path

from src.ingestion.chunker import TextChunker


# Add project root directory to sys.path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_chunker_splitting():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    sample_text = "Data point " * 40
    chunks = chunker.split(sample_text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)


# ---------- table-aware page parsing ----------

from src.ingestion.parser import page_to_text


def _line(y, *cells):
    """Builds word boxes for one visual line: cells are (x0, text) pairs."""
    return [(x, y, x + 8 * len(text), y + 9, text) for x, text in cells]


STATEMENT_PAGE = (
    _line(10, (40, "CONSOLIDATED"), (120, "STATEMENTS"), (200, "OF"), (220, "OPERATIONS"))
    + _line(30, (300, "2015"), (400, "2014"), (500, "2013"))
    + _line(50, (40, "Net"), (70, "sales"), (290, "$"), (300, "233,715"), (400, "182,795"), (500, "170,910"))
    + _line(70, (40, "Operating"), (110, "expenses:"))
    + _line(90, (55, "Research"), (120, "and"), (150, "development"), (300, "8,067"), (400, "6,041"), (500, "4,475"))
    + _line(110, (40, "Net"), (70, "income"), (300, "53,394"), (400, "39,510"), (500, "(1,234)"))
    + _line(130, (40, "Sales"), (90, "grew"), (130, "28%"), (170, "in"), (190, "2015."))
    + _line(150, (40, "https://www.sec.gov/Archives/x.htm"), (400, "43/80"))
)


def test_rows_carry_their_year_columns():
    text = page_to_text(STATEMENT_PAGE)
    assert "Net sales — 2015: 233,715; 2014: 182,795; 2013: 170,910" in text
    assert "Net income — 2015: 53,394; 2014: 39,510; 2013: (1,234)" in text


def test_subheading_applies_only_to_indented_rows():
    text = page_to_text(STATEMENT_PAGE)
    assert "Operating expenses: Research and development — 2015: 8,067" in text
    assert "Operating expenses: Net income" not in text


def test_prose_with_numbers_is_kept_and_print_noise_dropped():
    text = page_to_text(STATEMENT_PAGE)
    assert "Sales grew 28% in 2015." in text
    assert "sec.gov" not in text


def test_words_out_of_order_are_regrouped_into_lines():
    shuffled = list(reversed(STATEMENT_PAGE))
    assert page_to_text(shuffled) == page_to_text(STATEMENT_PAGE)
