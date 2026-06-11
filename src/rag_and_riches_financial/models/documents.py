from dataclasses import dataclass, field


@dataclass(frozen=True)
class FinancialDocument:
    doc_id: str
    doc_type: str
    source_name: str
    company: str
    date: str
    section: str
    title: str
    text: str
    tags: list[str] = field(default_factory=list)
