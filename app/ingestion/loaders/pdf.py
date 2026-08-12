from pathlib import Path
from app.ingestion.models import Document
from .base import BaseDocumentLoader


class PDFLoader(BaseDocumentLoader):
    def load(self, path: str | Path) -> Document:
        from pypdf import PdfReader

        p = Path(path)
        text = "\n".join(
            (page.extract_text() or "") for page in PdfReader(str(p)).pages
        )
        return Document(text, str(p), {"extension": p.suffix.lower()})
