from pathlib import Path
from app.ingestion.models import Document
from .base import BaseDocumentLoader


class TextLoader(BaseDocumentLoader):
    def load(self, path: str | Path) -> Document:
        p = Path(path)
        return Document(
            p.read_text(encoding="utf-8"), str(p), {"extension": p.suffix.lower()}
        )
