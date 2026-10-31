"""
Turns an EDGAR 10-K HTML (inline XBRL) document into plain text with tables kept readable.

EDGAR tables split one figure across several cells ('$' | '391,035' | ')') and position
columns with colspans, so reading cells left to right loses which year a value belongs to.
Each cell is placed on a column grid instead, and a value is assigned to the header cell whose
grid span overlaps it, giving the same row format as the PDF parser:

    Total net sales — 2024: 391,035; 2023: 383,285; 2022: 394,328
"""
import re
import warnings
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_VALUE_RE = re.compile(r"^\(?-?[\d,]*\.?\d+\)?%?$|^—$|^-$")
_BLOCK_TAGS = ["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "br"]

Cell = Tuple[int, int, str]  # (first grid column, last grid column, text)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _row_cells(tr: Tag) -> List[Cell]:
    """Non-empty cells with their grid position; '$' and lone ')' / '%' fragments are merged away."""
    cells, col = [], 0
    for td in tr.find_all(["td", "th"], recursive=False):
        span = int(td.get("colspan", 1) or 1)
        # '( 95,971 )' and '$ 1,234' inside one cell -> '(95,971)' and '1,234'
        text = re.sub(r"^\$\s*|(?<=\()\s+|\s+(?=\))", "", _clean(td.get_text(" ")))
        # Only glue onto figures: a '%' after a '2023' header cell is the next column's unit
        if text in (")", "%", ")%") and cells and _VALUE_RE.match(cells[-1][2]) \
                and not _YEAR_RE.fullmatch(cells[-1][2]):
            first, last, prev = cells[-1]
            cells[-1] = (first, col + span - 1, prev + text)  # '(1,234' + ')' -> '(1,234)'
        elif text and text != "$":
            cells.append((col, col + span - 1, text))
        col += span
    return cells


def _year_label(text: str) -> str:
    """'September 28, 2024' -> '2024'; headers with no or several years are kept as written."""
    years = _YEAR_RE.findall(text)
    return _YEAR_RE.search(text).group(0) if len(years) == 1 else text


def _is_header(cells: List[Cell]) -> bool:
    texts = [t for _, _, t in cells]
    if not texts or any(_VALUE_RE.match(t) and not _YEAR_RE.fullmatch(t) for t in texts):
        return False
    return any(_YEAR_RE.search(t) for t in texts)


def _column_for(value: Cell, header: List[Cell]) -> Optional[str]:
    for first, last, label in header:
        if value[0] <= last and value[1] >= first:
            return label
    return None


def table_to_text(table: Tag) -> str:
    lines: List[str] = []
    header: List[Cell] = []
    previous_was_header = False
    section = ""  # 'Net sales:' style sub-heading, prefixed to the rows under it
    for tr in table.find_all("tr"):
        cells = _row_cells(tr)
        if not cells:
            continue
        if _is_header(cells):
            # Drop a leading label cell ('Years ended') so only real columns remain
            new = [(f, l, _year_label(t)) for f, l, t in cells
                   if (_YEAR_RE.search(t) or f > cells[0][1]) and re.search(r"[A-Za-z0-9]", t)]
            if previous_was_header:
                # Two-row header ('2025 | 2024' under '2025 vs. 2024 Change'): keep upper cells
                # for the columns the lower row leaves uncovered
                new += [h for h in header if not any(h[0] <= l and h[1] >= f for f, l, _ in new)]
            header = sorted(new)
            previous_was_header = True
            continue
        previous_was_header = False

        label_cells = [c for c in cells if not _VALUE_RE.match(c[2])]
        values = [c for c in cells if _VALUE_RE.match(c[2])]
        label = " ".join(t for _, _, t in label_cells)

        if not values:
            if label.endswith(":"):
                section = label
            elif label:
                lines.append(label)
            continue

        full_label = f"{section} {label}".strip() if section and not label.lower().startswith("total") else label
        if label.lower().startswith("total"):
            section = ""
        columns = [_column_for(v, header) for v in values] if header else []
        if columns and all(columns):
            lines.append(f"{full_label} — " + "; ".join(f"{c}: {v[2]}" for c, v in zip(columns, values)))
        else:
            lines.append(f"{full_label}: " + " | ".join(v[2] for v in values))
    return "\n".join(lines)


def html_to_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    # The inline XBRL header is a hidden block of raw tagged facts, not readable content
    for hidden in soup.find_all(["ix:header", "script", "style"]):
        hidden.decompose()

    for table in soup.find_all("table"):
        rendered = soup.new_tag("pre")
        rendered.string = "\n" + table_to_text(table) + "\n"
        table.replace_with(rendered)

    # Break lines only at block elements; inline spans stay inside their sentence
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.append("\n")

    lines = [_clean(line) for line in soup.get_text("").split("\n")]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
