from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    chunk_index: int
    chunking_strategy: str
    section: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
