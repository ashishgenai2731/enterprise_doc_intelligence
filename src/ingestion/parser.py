import pymupdf as fitz
from typing import List
from src.ingestion.chunker import TextChunker


class DocumentIngestor:
    def __init__(self):
        self.chunker = TextChunker()

    def parse_pdf(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        extracted_text = ""

        for page in doc:
            tables = page.find_tables()
            if tables.tables:
                for table in tables:
                    # Call to_markdown() directly on the Table object
                    extracted_text += "\n" + table.to_markdown() + "\n"
            else:
                extracted_text += page.get_text("text") + "\n"

        doc.close()
        return extracted_text

    def create_chunks(self, raw_text: str) -> List[str]:
        return self.chunker.split(raw_text)