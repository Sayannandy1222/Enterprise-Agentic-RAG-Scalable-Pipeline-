from pathlib import Path
from .text import TextLoader
from .pdf import PDFLoader


class LoaderRouter:
    def __init__(self):
        self._loaders = {
            ".txt": TextLoader(),
            ".md": TextLoader(),
            ".rst": TextLoader(),
            ".pdf": PDFLoader(),
        }

    def load(self, path):
        p = Path(path)
        loader = self._loaders.get(p.suffix.lower())
        if not loader:
            raise ValueError(f"unsupported file type: {p.suffix}")
        return loader.load(p)
