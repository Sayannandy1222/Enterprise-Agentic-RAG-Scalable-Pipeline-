from dataclasses import dataclass, field


@dataclass
class Document:
    text: str
    source: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    text: str
    source: str = ""
    metadata: dict = field(default_factory=dict)
