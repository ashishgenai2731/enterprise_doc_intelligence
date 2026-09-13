from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings

class TextChunker:
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, chunk_overlap: int = settings.CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split(self, text: str) -> List[str]:
        """Splits continuous text into overlapping semantic chunks."""
        return self.splitter.split_text(text)