import re
import pymupdf as fitz
from typing import List, Optional, Tuple
from src.ingestion.chunker import TextChunker

# A word is a (x0, y0, x1, y1, text) box from page.get_text("words")
Word = Tuple[float, float, float, float, str]

_VALUE_RE = re.compile(r"^\(?-?\d[\d,]*(\.\d+)?\)?%?$|^[—–-]$")
_YEAR_RE = re.compile(r"^(19|20)\d\d$")
# Browser-print and navigation noise found on every page of SEC filings saved as PDF
_NOISE_RE = re.compile(r"^(Table of Contents$|https?://|\d{1,2}/\d{1,2}/\d{4}, \d{1,2}:\d{2})")

LINE_Y_TOLERANCE = 3.0  # points; words closer than this vertically share a visual line
INDENT_TOLERANCE = 3.0  # points; a row must start this far right of a sub-heading to belong to it


def _group_lines(words: List[Word]) -> List[List[Word]]:
    """Groups word boxes into visual lines (same baseline), each sorted left to right."""
    lines: List[List[Word]] = []
    for word in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        y_mid = (word[1] + word[3]) / 2
        if lines and abs(y_mid - (lines[-1][0][1] + lines[-1][0][3]) / 2) <= LINE_Y_TOLERANCE:
            lines[-1].append(word)
        else:
            lines.append([word])
    return [sorted(line, key=lambda w: w[0]) for line in lines]


def _split_row(line: List[Word]) -> Tuple[str, List[Word]]:
    """Splits a line into (label, value words) if everything after the label is numeric."""
    tokens = [w for w in line if w[4] != "$"]
    first_value = next((i for i, w in enumerate(tokens) if _VALUE_RE.match(w[4])), None)
    if first_value is None:
        return " ".join(w[4] for w in tokens), []
    values = tokens[first_value:]
    if not all(_VALUE_RE.match(w[4]) for w in values):
        return " ".join(w[4] for w in tokens), []  # prose that happens to contain numbers
    return " ".join(w[4] for w in tokens[:first_value]), values


def _nearest_column(value: Word, columns: List[Tuple[str, float]]) -> str:
    x_mid = (value[0] + value[2]) / 2
    return min(columns, key=lambda col: abs(col[1] - x_mid))[0]


def page_to_text(words: List[Word]) -> str:
    """
    Rebuilds a page as text where every financial-table row carries its own column headers.

    PyMuPDF's find_tables() only detects the shaded rows of SEC statements, so white rows
    (Net sales, Net income, ...) were dropped and the year headers were lost. Here each row
    is matched to the year header above it by x-position, giving lines like
    'Net income — 2015: 53,394; 2014: 39,510; 2013: 37,037'.
    """
    out: List[str] = []
    columns: List[Tuple[str, float]] = []  # (year label, x-center) of the active table header
    group: Optional[Tuple[str, float]] = None  # (sub-heading like "Earnings per share:", its x0)
    pending_label = ""  # first half of a row label that wrapped onto two lines

    for line in _group_lines(words):
        text = " ".join(w[4] for w in line)
        if _NOISE_RE.match(text.strip()):
            continue

        if len(line) >= 2 and all(_YEAR_RE.match(w[4]) for w in line):
            columns = [(w[4], (w[0] + w[2]) / 2) for w in line]
            group, pending_label = None, ""
            out.append(text)
            continue

        label, values = _split_row(line)
        if columns and values and label:
            if label[0].islower() and pending_label:
                label = f"{pending_label} {label}"
            if group and line[0][0] > group[1] + INDENT_TOLERANCE:
                label = f"{group[0]} {label}"
            else:
                group = None  # row is back at the sub-heading's indent: that section ended
            cells = "; ".join(f"{_nearest_column(v, columns)}: {v[4]}" for v in values)
            out.append(f"{label} — {cells}")
            pending_label = ""
            continue

        if columns and label.endswith(":"):
            group = (label, line[0][0])
        elif columns and not values:
            pending_label = label
        out.append(text)

    return "\n".join(out)


class DocumentIngestor:
    def __init__(self):
        self.chunker = TextChunker()

    def parse_pdf(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        pages = [page_to_text([w[:5] for w in page.get_text("words")]) for page in doc]
        doc.close()
        return "\n\n".join(pages)

    def create_chunks(self, raw_text: str) -> List[str]:
        return self.chunker.split(raw_text)
