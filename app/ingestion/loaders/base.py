from abc import ABC, abstractmethod
from pathlib import Path
from app.ingestion.models import Document
class BaseDocumentLoader(ABC):
    @abstractmethod
    def load(self, path: str|Path) -> Document: ...

